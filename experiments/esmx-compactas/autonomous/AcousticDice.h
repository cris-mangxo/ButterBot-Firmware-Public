#pragma once

#include "Audio/EsMxKws/Compact.h"

namespace EsMx::Kws {

constexpr int DiceChannels = 24;
constexpr int DiceClasses = 4;
constexpr int DiceRegions = 3;

template<int Inputs, int Width>
struct DiceConvolution {
	std::array<int8_t, DiceChannels * Inputs * Width> weights;
	std::array<int32_t, DiceChannels> biases;
	uint32_t shift;
};

struct AcousticDice {
	DiceConvolution<Bands, 5> first;
	DiceConvolution<DiceChannels, 3> second;
	DiceConvolution<DiceChannels, 3> third;
	std::array<int8_t, DiceClasses * DiceRegions * DiceChannels> output;
	std::array<int32_t, DiceClasses> biases;
	int32_t minimumScore;
	int32_t minimumMargin;
};

using DiceScores = std::array<int32_t, DiceClasses>;
enum DiceClass : size_t { DiceOutside, DiceRequest, DiceOther, DiceNegation };
bool validAcousticDice(const AcousticDice& model);
DiceScores acousticDiceScores(const Signature& input, const AcousticDice& model);
bool proposesDice6(const DiceScores& scores, const AcousticDice& model);
Intent completeAcousticDice(const Signature& input, Intent baseline, Intent verified,
                           const AcousticDice& model);
}
