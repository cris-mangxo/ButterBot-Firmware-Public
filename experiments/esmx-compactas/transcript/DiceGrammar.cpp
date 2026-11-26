#include "DiceGrammar.h"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <span>

namespace EsMx::Kws {
namespace {

using namespace std::literals;

constexpr std::array Negation{
	"no"sv, "nunca"sv, "jamas"sv, "tampoco"sv, "ni"sv, "sin"sv, "nada"sv,
};
constexpr std::array Verbs{
	"tira"sv, "tirar"sv, "tiras"sv, "tirame"sv, "tirarme"sv, "tires"sv,
	"lanza"sv, "lanzar"sv, "lanzas"sv, "lanzame"sv, "lanzarme"sv, "lances"sv,
	"avienta"sv, "aventar"sv, "avientas"sv, "avientame"sv, "avientes"sv,
	"echa"sv, "echar"sv, "echas"sv, "echame"sv, "eche"sv, "eches"sv,
	"rueda"sv, "rodar"sv, "ruedas"sv,
};
constexpr std::array Articles{
	"el"sv, "un"sv, "ese"sv, "este"sv, "tu"sv, "mi"sv,
};
constexpr std::array Faces{"caras"sv, "lados"sv};
constexpr std::array<uint8_t, 14> AccentSecond{
	0xA1, 0xA9, 0xAD, 0xB3, 0xBA, 0xBC, 0xB1,
	0x81, 0x89, 0x8D, 0x93, 0x9A, 0x9C, 0x91,
};
constexpr std::string_view AccentBase = "aeiouunaeiouun";

struct Turn {
	std::array<char, MaxTranscriptBytes> normalized{};
	std::array<std::string_view, 24> words{};
	size_t count = 0;
};

char latinLetter(uint8_t second){
	const auto found = std::find(AccentSecond.begin(), AccentSecond.end(), second);
	return found == AccentSecond.end() ? '\0' : AccentBase[found - AccentSecond.begin()];
}

bool tokenize(std::string_view input, Turn& output){
	if(input.empty() || input.size() > MaxTranscriptBytes){
		return false;
	}
	size_t length = 0;
	bool turnEnded = false;
	for(size_t i = 0; i < input.size(); ++i){
		const auto byte = static_cast<uint8_t>(input[i]);
		char symbol = static_cast<char>(byte);
		if(byte == 0xC2 || byte == 0xC3){
			if(++i == input.size()){
				return false;
			}
			const auto suffix = static_cast<uint8_t>(input[i]);
			if(byte == 0xC3){
				symbol = latinLetter(suffix);
			}else{
				const bool boundary = length == 0 || output.normalized[length - 1] == ' ';
				symbol = (suffix == 0xA1 || suffix == 0xBF) && boundary ? ' ' : '\0';
			}
		}
		if(symbol >= 'A' && symbol <= 'Z'){
			symbol += 'a' - 'A';
		}
		if((symbol >= 'a' && symbol <= 'z') || (symbol >= '0' && symbol <= '9')){
			if(turnEnded){
				return false;
			}
			output.normalized[length++] = symbol;
		}else if(symbol == ' ' || symbol == '\t' || symbol == ',' ||
		         symbol == '.' || symbol == '!' || symbol == '?'){
			turnEnded |= (symbol == '.' || symbol == '!' || symbol == '?');
			if(length && output.normalized[length - 1] != ' '){
				output.normalized[length++] = ' ';
			}
		}else{
			// Quotes, colon, separated turns and unsupported codepoints fail closed.
			return false;
		}
	}
	const std::string_view words(output.normalized.data(), length);
	for(size_t start = 0; start < words.size();){
		if(output.count == output.words.size()){
			return false;
		}
		const size_t end = words.find(' ', start);
		output.words[output.count++] = words.substr(start, end - start);
		if(end == std::string_view::npos){
			break;
		}
		start = end + 1;
	}
	return output.count != 0;
}

class Words {
public:
	explicit Words(const Turn& turn): remaining(turn.words.data(), turn.count){}

	bool take(std::string_view word){
		if(remaining.empty() || remaining.front() != word){
			return false;
		}
		remaining = remaining.subspan(1);
		return true;
	}

	template<size_t N>
	bool oneOf(const std::array<std::string_view, N>& choices){
		if(remaining.empty() || std::find(choices.begin(), choices.end(), remaining.front()) == choices.end()){
			return false;
		}
		remaining = remaining.subspan(1);
		return true;
	}

	bool phrase(std::initializer_list<std::string_view> words){
		if(words.size() > remaining.size() || !std::equal(words.begin(), words.end(), remaining.begin())){
			return false;
		}
		remaining = remaining.subspan(words.size());
		return true;
	}

	bool empty() const { return remaining.empty(); }

private:
	std::span<const std::string_view> remaining;
};

bool politeness(Words& words){
	return words.phrase({"por", "favor"}) || words.phrase({"por", "fa"}) || words.take("porfa");
}

bool validPredicate(Words& words){
	words.phrase({"a", "ver"}) || words.take("oye") || words.take("ahorita");
	politeness(words);
	words.phrase({"te", "pido", "que"}) || words.phrase({"vamos", "a"}) ||
	    words.phrase({"quiero", "que"}) || words.phrase({"me", "puedes"}) ||
	    words.take("quiero") || words.take("puedes") || words.take("podrias");
	politeness(words);
	words.take("me");
	return words.phrase({"haz", "rodar"}) || words.phrase({"haz", "una", "tirada", "con"}) ||
	       words.oneOf(Verbs);
}

bool six(Words& words){
	return words.take("seis") || words.take("6");
}

bool validObject(Words& words){
	politeness(words);
	words.oneOf(Articles);
	if(words.take("dado")){
		if(!words.take("d6")){
			if(!words.take("de") || !six(words)){
				return false;
			}
			words.oneOf(Faces);
		}
	}else if(!words.take("d6")){
		return false;
	}
	politeness(words);
	return words.empty();
}
}

bool isDice6Request(std::string_view transcript){
	Turn turn;
	if(!tokenize(transcript, turn)){
		return false;
	}
	for(const auto word : std::span(turn.words.data(), turn.count)){
		if(std::find(Negation.begin(), Negation.end(), word) != Negation.end()){
			return false;
		}
	}
	Words words(turn);
	return validPredicate(words) && validObject(words);
}

Intent completeDice6(Intent baseline, Intent verified, std::string_view transcript){
	if(verified != Intent::Unknown || baseline != Intent::Unknown){
		return verified;
	}
	return isDice6Request(transcript) ? Intent::Dice6 : Intent::Unknown;
}

}
