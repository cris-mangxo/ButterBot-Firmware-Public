#include "Audio/EsMxKws/Compact.h"
#include "Audio/EsMxKws/ScenarioForIntent.h"

#include <array>
#include <charconv>
#include <fstream>
#include <iostream>
#include <string_view>

using namespace EsMx::Kws;

namespace {
static_assert(sizeof(Prototype) == sizeof(Signature) + 1);
static_assert(sizeof(Signature) == Frames * Bands);
std::array<int16_t, MaxSamples + 1> pcm;
bb::asr::LogMel frontend;

bool parseUnsigned(std::string_view text, uint32_t& value){
	const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
	return error == std::errc{} && end == text.data() + text.size();
}
}

int main(int argc, char** argv){
	const bool features = argc == 2 && std::string_view(argv[1]) == "features";
	const bool score = argc == 5 && std::string_view(argv[1]) == "score";
	if(!features && !score){
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
	const auto scenario = scenarioForIntent(result.intent);
	std::cout << "{\"label\":" << static_cast<int>(result.intent)
	          << ",\"cost\":" << result.cost << ",\"gap\":" << result.gap;
	if(scenario){
		std::cout << ",\"scenario\":" << static_cast<int>(scenario->first)
		          << ",\"data\":" << static_cast<int>(scenario->second.raw);
	}
	std::cout << "}\n";
}
