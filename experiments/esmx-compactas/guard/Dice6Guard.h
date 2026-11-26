#pragma once

#include "Audio/EsMxKws/Compact.h"

namespace EsMx::Kws {

constexpr size_t GuardHidden = 12;
constexpr size_t GuardClasses = 4;
constexpr int32_t GuardDivisor = 256;

struct Dice6Model {
	// Class order: outside command, D6, another die, negation.
	std::array<std::array<int8_t, Frames * Bands>, GuardHidden> input;
	std::array<int32_t, GuardHidden> inputBias;
	std::array<std::array<int8_t, GuardHidden>, GuardClasses> output;
	std::array<int32_t, GuardClasses> outputBias;
	int32_t minMargin;
};

struct Dice6Decision {
	Intent intent;
	int64_t margin;
};

Dice6Decision guardDice6(const Signature& signature, Intent baseline, const Dice6Model& guard);

struct Dice6Anchors {
	std::array<Signature, 2> examples;
	uint32_t maxCost;
};

Dice6Decision confirmDice6(const Signature& signature, Intent baseline, const Dice6Anchors& anchors);
}
