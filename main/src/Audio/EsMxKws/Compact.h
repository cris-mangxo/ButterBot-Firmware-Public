#pragma once

#include "logmel.h"
#include <array>
#include <span>

namespace EsMx::Kws {
constexpr int Frames = 24;
constexpr int Bands = 20;
constexpr size_t MaxSamples = 4 * bb::asr::kSampleRate;
constexpr size_t MaxTemplates = 24;

enum class Intent : uint8_t {
	Unknown, Butter, YouButter, Joke, Fact, Time, Dice6, Count
};

struct Signature {
	std::array<std::array<int8_t, Bands>, Frames> frames;
};

struct Prototype {
	Intent intent;
	Signature signature;
};

struct Limits {
	uint32_t maxCost;
	uint32_t minGap;
};

struct Result {
	Intent intent;
	uint32_t cost;
	uint32_t gap;
};

// The caller owns the PCM and initialized P31 front-end. No heap allocation.
bool extract(bb::asr::LogMel& frontend, std::span<const int16_t> pcm, Signature& output);
uint32_t signatureDistance(const Signature& input, const Signature& example);
Result classify(const Signature& input, std::span<const Prototype> model, Limits limits);
}
