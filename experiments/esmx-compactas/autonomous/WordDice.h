#pragma once

#include "Audio/EsMxKws/Compact.h"

namespace EsMx::Kws {

constexpr int WordStride = 2 * bb::asr::kHop;
constexpr int WordWindow = 17;
constexpr int WordHidden = 24;
constexpr int WordLabels = 7;
constexpr size_t MaxWordFrames = 1 + (MaxSamples - bb::asr::kWin) / WordStride;

struct WordFeatures {
	std::array<std::array<int8_t, Bands>, MaxWordFrames> frames;
	size_t count;
};

struct WordBank {
	std::array<int8_t, WordHidden * Bands * WordWindow> input;
	std::array<int32_t, WordHidden> inputBias;
	uint32_t inputShift;
	std::array<int8_t, WordHidden * WordHidden> hidden;
	std::array<int32_t, WordHidden> hiddenBias;
	uint32_t hiddenShift;
	std::array<int8_t, WordLabels * WordHidden> output;
	std::array<int32_t, WordLabels> outputBias;
	int32_t wordMargin;
	int32_t vetoMargin;
};

bool validWordBank(const WordBank& bank);
bool extractWordFeatures(bb::asr::LogMel& frontend, std::span<const int16_t> pcm, WordFeatures& output);
std::array<int32_t, WordLabels> wordLogits(const WordFeatures& input, size_t time, const WordBank& bank);
bool acceptsWordDice(const WordFeatures& input, const WordBank& bank);
bool acceptsContrastedDice(const WordFeatures& input, const WordBank& bank);
bool confirmsSixAcrossBanks(const WordFeatures& input, const WordBank& primary, const WordBank& extra,
                            const WordBank& adapted);
enum class DiceConfirmation { SixOnly, OrderedRescue };

Intent confirmContinuousDice(Intent previous, Intent candidate, const WordFeatures& input,
                             const WordBank& bank,
                             DiceConfirmation confirmation = DiceConfirmation::SixOnly);
Intent completeWordDice(bb::asr::LogMel& frontend, std::span<const int16_t> pcm,
                        Intent baseline, Intent verified, const WordBank& model, WordFeatures& workspace,
                        std::span<const WordBank* const> supplements = {});
}
