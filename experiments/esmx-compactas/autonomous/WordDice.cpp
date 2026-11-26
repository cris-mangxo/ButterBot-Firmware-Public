#include "WordDice.h"

#include "Audio/EsMxKws/cuantizar.h"
#include <algorithm>

namespace EsMx::Kws {
namespace {
constexpr int32_t MaxBias = 1 << 24;

enum Label {Other, Verb, Die, Six, Denial, OtherNumber, Context};

bool validBias(int32_t value){
	return value >= -MaxBias && value <= MaxBias;
}

int8_t relu(int32_t value, uint32_t shift){
	const int32_t round = int32_t(1) << (shift - 1);
	return static_cast<int8_t>(std::min<int32_t>(127, (std::max<int32_t>(value, 0) + round) >> shift));
}

int token(const std::array<int32_t, WordLabels>& logits, const WordBank& bank){
	const auto best = std::max_element(logits.begin(), logits.end());
	const int selected = static_cast<int>(best - logits.begin());
	int32_t competitor = -MaxBias * 2;
	for(int label = 0; label < WordLabels; ++label){
		if(label != selected){
			competitor = std::max(competitor, logits[label]);
		}
	}
	const int32_t minimum = selected >= Denial ? bank.vetoMargin : bank.wordMargin;
	return *best - competitor >= minimum ? selected : Other;
}
}

bool validWordBank(const WordBank& bank){
	return bank.inputShift > 0 && bank.inputShift <= 24 && bank.hiddenShift > 0 && bank.hiddenShift <= 24 &&
	       std::all_of(bank.inputBias.begin(), bank.inputBias.end(), validBias) &&
	       std::all_of(bank.hiddenBias.begin(), bank.hiddenBias.end(), validBias) &&
	       std::all_of(bank.outputBias.begin(), bank.outputBias.end(), validBias) &&
	       bank.wordMargin > 0 && bank.wordMargin <= MaxBias &&
	       bank.vetoMargin > 0 && bank.vetoMargin <= bank.wordMargin;
}

bool extractWordFeatures(bb::asr::LogMel& frontend, std::span<const int16_t> pcm, WordFeatures& output){
	if(!frontend.ready || pcm.size() < bb::asr::kWin || pcm.size() > MaxSamples){
		return false;
	}
	output.count = 1 + (pcm.size() - bb::asr::kWin) / WordStride;
	for(size_t frame = 0; frame < output.count; ++frame){
		float mel[bb::asr::kBands];
		bb::asr::logmel_frame(&frontend, pcm.data() + frame * WordStride, mel);
		float mean = 0;
		for(float value : mel){
			mean += value;
		}
		mean /= bb::asr::kBands;
		for(int band = 0; band < Bands; ++band){
			output.frames[frame][band] = bb::asr::quantize_value(
			    (mel[2 * band] + mel[2 * band + 1]) * 0.5f - mean, 0.125, 0);
		}
	}
	return true;
}

std::array<int32_t, WordLabels> wordLogits(const WordFeatures& input, size_t time, const WordBank& bank){
	std::array<int8_t, WordHidden> first, hidden;
	for(int channel = 0; channel < WordHidden; ++channel){
		int32_t sum = bank.inputBias[channel];
		for(int k = 0; k < WordWindow; ++k){
			const int frame = static_cast<int>(time) + k - WordWindow / 2;
			if(frame < 0 || static_cast<size_t>(frame) >= input.count){
				continue;
			}
			for(int band = 0; band < Bands; ++band){
				sum += int32_t(input.frames[frame][band]) * bank.input[(channel * Bands + band) * WordWindow + k];
			}
		}
		first[channel] = relu(sum, bank.inputShift);
	}
	for(int channel = 0; channel < WordHidden; ++channel){
		int32_t sum = bank.hiddenBias[channel];
		for(int term = 0; term < WordHidden; ++term){
			sum += int32_t(first[term]) * bank.hidden[channel * WordHidden + term];
		}
		hidden[channel] = relu(sum, bank.hiddenShift);
	}
	auto scores = bank.outputBias;
	for(int label = 0; label < WordLabels; ++label){
		for(int term = 0; term < WordHidden; ++term){
			scores[label] += int32_t(hidden[term]) * bank.output[label * WordHidden + term];
		}
	}
	return scores;
}

bool acceptsWordDice(const WordFeatures& input, const WordBank& bank){
	constexpr std::array<int, 3> order{Verb, Die, Six};
	int previous = Other;
	int run = 0;
	size_t stage = 0;
	int lastToken = Other;
	size_t lastFrame = 0;
	for(size_t frame = 0; frame < input.count; ++frame){
		const auto current = token(wordLogits(input, frame, bank), bank);
		run = current == previous ? run + 1 : 1;
		previous = current;
		if(current == Other || run != 2){
			continue;
		}
		if(current >= Denial){
			return false;
		}
		if(current == lastToken && frame - lastFrame <= 10){
			lastFrame = frame;
			continue;
		}
		if(stage >= order.size() || current != order[stage] || (stage && frame - lastFrame > 50)){
			return false;
		}
		lastFrame = frame;
		lastToken = current;
		++stage;
	}
	return stage == order.size();
}

Intent completeWordDice(bb::asr::LogMel& frontend, std::span<const int16_t> pcm,
                        Intent baseline, Intent verified, const WordBank& model, WordFeatures& workspace){
	if(baseline != Intent::Unknown || verified != Intent::Unknown){
		return verified;
	}
	if(extractWordFeatures(frontend, pcm, workspace) && acceptsWordDice(workspace, model)){
		return Intent::Dice6;
	}
	return Intent::Unknown;
}
}
