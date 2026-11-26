#include "EsMxCommand.h"

#include <array>
#include <cstddef>

namespace {
using Action = BB::Action::Scenario;
using Die = DiceScenarioData::DiceType;
using EsMx::Command;

constexpr Command Butter{"pass_the_butter", {Action::PassTheButter, {}}};
constexpr Command YouButter{"you_pass_butter", {Action::YouPassButter, {}}};
constexpr Command Joke{"joke", {Action::Joke, {}}};
constexpr Command Fact{"fact", {Action::Fact, {}}};
constexpr Command Object{"whats_this", {Action::WhatsThis, {}}};
constexpr Command Time{"current_time", {Action::CurrentTime, {}}};
constexpr Command Dice{"dice", {Action::DiceRoll, DiceScenarioData{Die::None}}};
constexpr Command Dice6{"dice_d6", {Action::DiceRoll, DiceScenarioData{Die::D6}}};
constexpr Command Dice100{"dice_d100", {Action::DiceRoll, DiceScenarioData{Die::D100}}};
constexpr Command EightBall{"eight_ball", {Action::EightBall, {}}};

struct Alias {
	std::string_view text;
	const Command& command;
};

constexpr Alias Aliases[] = {
	{"pasame la mantequilla", Butter},
	{"dame la mantequilla", Butter},
	{"me pasas la mantequilla", Butter},
	{"tu pasa la mantequilla", YouButter},
	{"tu pasa mantequilla", YouButter},
	{"cuentame un chiste", Joke},
	{"echame un chiste", Joke},
	{"avientate un chiste", Joke},
	{"dime un chiste", Joke},
	{"dime un dato curioso", Fact},
	{"cuentame un dato curioso", Fact},
	{"dame un dato curioso", Fact},
	{"que es esto", Object},
	{"que ves", Object},
	{"que hora es", Time},
	{"dime la hora", Time},
	{"tira un dado", Dice},
	{"lanza un dado", Dice},
	{"tira un dado de seis", Dice6},
	{"tira un dado de 6", Dice6},
	{"lanza un dado de seis caras", Dice6},
	{"lanza un d6", Dice6},
	{"tira un dado de cien", Dice100},
	{"tira un dado de 100", Dice100},
	{"lanza un dado de cien caras", Dice100},
	{"lanza un d100", Dice100},
	{"bola de la suerte", EightBall},
	{"ayudame a decidir", EightBall},
	{"dame un consejo", EightBall},
};

constexpr std::size_t MaxTranscriptBytes = 128;
constexpr std::string_view EdgePunctuation[] = {"¿", "¡", "?", "!", "."};
constexpr std::pair<std::string_view, char> Accents[] = {
	{"á", 'a'}, {"é", 'e'}, {"í", 'i'}, {"ó", 'o'}, {"ú", 'u'}, {"ü", 'u'},
	{"Á", 'a'}, {"É", 'e'}, {"Í", 'i'}, {"Ó", 'o'}, {"Ú", 'u'}, {"Ü", 'u'},
};

bool isGap(char ch){
	return ch == ' ' || ch == '\t';
}

void trimGap(std::string_view& text){
	while(!text.empty() && isGap(text.front())){
		text.remove_prefix(1);
	}
	while(!text.empty() && isGap(text.back())){
		text.remove_suffix(1);
	}
}

void stripSentencePunctuation(std::string_view& text){
	trimGap(text);
	for(auto punctuation : EdgePunctuation){
		if(text.starts_with(punctuation)){
			text.remove_prefix(punctuation.size());
			break;
		}
	}
	for(auto punctuation : EdgePunctuation){
		if(text.ends_with(punctuation)){
			text.remove_suffix(punctuation.size());
			break;
		}
	}
	trimGap(text);
}

std::optional<std::string_view> normalize(std::string_view text, std::array<char, MaxTranscriptBytes>& buffer){
	if(text.size() > buffer.size()){
		return std::nullopt;
	}
	stripSentencePunctuation(text);
	std::size_t length = 0;
	while(!text.empty()){
		const char ch = text.front();
		if(isGap(ch)){
			if(length != 0 && buffer[length - 1] != ' '){
				buffer[length++] = ' ';
			}
		}else if(ch >= 'A' && ch <= 'Z'){
			buffer[length++] = ch - 'A' + 'a';
		}else if((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9')){
			buffer[length++] = ch;
		}else{
			bool folded = false;
			for(const auto& [accent, letter] : Accents){
				if(text.starts_with(accent)){
					buffer[length++] = letter;
					text.remove_prefix(accent.size() - 1);
					folded = true;
					break;
				}
			}
			if(!folded){
				return std::nullopt;
			}
		}
		text.remove_prefix(1);
	}
	return std::string_view(buffer.data(), length);
}
}

std::optional<EsMx::Command> EsMx::matchCommand(std::string_view transcript){
	std::array<char, MaxTranscriptBytes> buffer;
	const auto normalized = normalize(transcript, buffer);
	if(normalized){
		for(const auto& [text, command] : Aliases){
			if(text == *normalized){
				return command;
			}
		}
	}
	return std::nullopt;
}
