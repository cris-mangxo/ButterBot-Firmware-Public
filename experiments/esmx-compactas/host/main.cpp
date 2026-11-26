#include "Audio/EsMxKws/Compact.h"
#include "Dice6Guard.h"
#include "PhraseGate.h"
#include "Audio/EsMxKws/ScenarioForIntent.h"

#include <array>
#include <charconv>
#include <fstream>
#include <iostream>
#include <string_view>
#include <type_traits>

using namespace EsMx::Kws;

namespace {
static_assert(sizeof(Prototype) == sizeof(Signature) + 1);
static_assert(sizeof(Signature) == Frames * Bands);
static_assert(sizeof(Dice6Model) == 5876);
static_assert(sizeof(Dice6Anchors) == 964);
static_assert(sizeof(PhraseBank) == 9676);
std::array<int16_t, MaxSamples + 1> pcm;
bb::asr::LogMel frontend;

bool parseUnsigned(std::string_view text, uint32_t& value){
	const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
	return error == std::errc{} && end == text.data() + text.size();
}

template <typename Model>
bool load(const char* path, Model& model){
	std::ifstream stream(path, std::ios::binary);
	return stream.read(reinterpret_cast<char*>(&model), sizeof(model)) &&
	       stream.peek() == std::ifstream::traits_type::eof() && !stream.bad();
}

template <typename T, size_t N>
void writeDistances(const Signature& input, const std::array<T, N>& examples){
	std::cout << '[';
	for(size_t i = 0; i < N; ++i){
		if(i){
			std::cout << ',';
		}
		if constexpr(std::is_same_v<T, Signature>){
			std::cout << signatureDistance(input, examples[i]);
		}else{
			std::cout << signatureDistance(input, examples[i].signature);
		}
	}
	std::cout << ']';
}
}

int main(int argc, char** argv){
	const bool features = argc == 2 && std::string_view(argv[1]) == "features";
	const bool score = argc == 5 && std::string_view(argv[1]) == "score";
	const bool guard = argc == 6 && std::string_view(argv[1]) == "guard";
	const bool anchor = argc == 6 && std::string_view(argv[1]) == "anchor";
	const bool phrases = argc == 6 && std::string_view(argv[1]) == "phrases";
	const bool distances = argc == 3 && std::string_view(argv[1]) == "distances";
	if(!features && !score && !guard && !anchor && !phrases && !distances){
		return 2;
	}
	std::cin.read(reinterpret_cast<char*>(pcm.data()), sizeof(pcm));
	const auto bytes = static_cast<size_t>(std::cin.gcount());
	if(bytes % 2 != 0 || bytes > MaxSamples * 2 || !bb::asr::logmel_init(&frontend)){
		return 2;
	}
	Signature signature;
	if(!extract(frontend, std::span(pcm.data(), bytes / 2), signature)){
		return 3;
	}
	if(features){
		std::cout.write(reinterpret_cast<const char*>(&signature), sizeof(signature));
		return std::cout ? 0 : 2;
	}
	if(distances){
		PhraseBank bank;
		if(!load(argv[2], bank) || !validPhraseBank(bank)){
			return 2;
		}
		std::cout << "{\"dice\":";
		writeDistances(signature, bank.dice);
		std::cout << ",\"facts\":";
		writeDistances(signature, bank.facts);
		std::cout << ",\"deniedFacts\":";
		writeDistances(signature, bank.deniedFacts);
		std::cout << "}\n";
		return 0;
	}

	std::ifstream file(argv[2], std::ios::binary);
	if(!file){
		return 2;
	}
	std::array<Prototype, MaxTemplates + 1> model;
	file.read(reinterpret_cast<char*>(model.data()), sizeof(model));
	const auto modelBytes = static_cast<size_t>(file.gcount());
	Limits limits;
	if(!file.eof() || modelBytes == 0 || modelBytes % sizeof(Prototype) != 0 ||
	   modelBytes > MaxTemplates * sizeof(Prototype) || !parseUnsigned(argv[3], limits.maxCost) ||
	   !parseUnsigned(argv[4], limits.minGap)){
		return 2;
	}
	const auto result = classify(signature, std::span(model.data(), modelBytes / sizeof(Prototype)), limits);
	auto intent = result.intent;
	int64_t margin = 0;
	if(guard){
		Dice6Model guardModel;
		if(!load(argv[5], guardModel) || guardModel.minMargin <= 0){
			return 2;
		}
		const auto decision = guardDice6(signature, intent, guardModel);
		intent = decision.intent;
		margin = decision.margin;
	}
	if(anchor){
		Dice6Anchors anchors;
		if(!load(argv[5], anchors) || anchors.maxCost == 0){
			return 2;
		}
		const auto decision = confirmDice6(signature, intent, anchors);
		intent = decision.intent;
		margin = decision.margin;
	}
	if(phrases){
		PhraseBank bank;
		if(!load(argv[5], bank) || !validPhraseBank(bank)){
			return 2;
		}
		intent = verifyPhrase(signature, intent, bank);
	}
	const auto scenario = scenarioForIntent(intent);
	std::cout << "{\"label\":" << static_cast<int>(intent)
	          << ",\"cost\":" << result.cost << ",\"gap\":" << result.gap;
	if(guard || anchor){
		std::cout << ",\"d6_margin\":" << margin;
	}
	if(scenario){
		std::cout << ",\"scenario\":" << static_cast<int>(scenario->first)
		          << ",\"data\":" << static_cast<int>(scenario->second.raw);
	}
	std::cout << "}\n";
}
