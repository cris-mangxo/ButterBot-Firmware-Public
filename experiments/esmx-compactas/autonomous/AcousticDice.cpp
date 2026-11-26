#include "AcousticDice.h"

#include <algorithm>

namespace EsMx::Kws {
namespace {
constexpr int32_t MaxBias = 1 << 24;
constexpr int32_t MaxLimit = 1 << 26;
using Features = std::array<std::array<int8_t, DiceChannels>, Frames>;

bool validBias(int32_t bias){
	return bias >= -MaxBias && bias <= MaxBias;
}

template<int InputChannels, int Width>
bool validConv(const DiceConvolution<InputChannels, Width>& layer){
	return layer.shift > 0 && layer.shift <= 24 &&
	       std::all_of(layer.biases.begin(), layer.biases.end(), validBias);
}

template<int In, int Width>
void apply(const std::array<std::array<int8_t, In>, Frames>& input,
           const DiceConvolution<In, Width>& layer, int dilation, Features& output){
	const int32_t round = int32_t(1) << (layer.shift - 1);
	for(int time = 0; time < Frames; ++time){
		for(int channel = 0; channel < DiceChannels; ++channel){
			int32_t sum = layer.biases[channel];
			for(int kern = 0; kern < Width; ++kern){
				const int index = time + (kern - Width / 2) * dilation;
				if(index < 0 || index >= Frames){
					continue;
				}
				for(int source = 0; source < In; ++source){
					const auto weight = layer.weights[(channel * In + source) * Width + kern];
					sum += int32_t(input[index][source]) * weight;
				}
			}
			output[time][channel] = static_cast<int8_t>(
			    std::min<int32_t>(127, (std::max<int32_t>(0, sum) + round) >> layer.shift));
		}
	}
}
}

bool validAcousticDice(const AcousticDice& model){
	return validConv(model.first) && validConv(model.second) && validConv(model.third) &&
	       std::all_of(model.biases.begin(), model.biases.end(), validBias) &&
	       model.minimumScore >= -MaxLimit && model.minimumScore <= MaxLimit &&
	       model.minimumMargin > 0 && model.minimumMargin <= MaxLimit;
}

DiceScores acousticDiceScores(const Signature& input, const AcousticDice& model){
	Features ping, pong;
	apply<Bands, 5>(input.frames, model.first, 1, ping);
	apply<DiceChannels, 3>(ping, model.second, 2, pong);
	apply<DiceChannels, 3>(pong, model.third, 3, ping);
	constexpr int RegionFrames = Frames / DiceRegions;
	static_assert(RegionFrames * DiceRegions == Frames);
	std::array<int8_t, DiceRegions * DiceChannels> pooled{};
	for(int frame = 0; frame < Frames; ++frame){
		const int region = frame / RegionFrames;
		for(int channel = 0; channel < DiceChannels; ++channel){
			auto& peak = pooled[region * DiceChannels + channel];
			peak = std::max(peak, ping[frame][channel]);
		}
	}
	DiceScores scores = model.biases;
	for(int label = 0; label < DiceClasses; ++label){
		for(size_t i = 0; i < pooled.size(); ++i){
			scores[label] += int32_t(pooled[i]) * model.output[label * pooled.size() + i];
		}
	}
	return scores;
}

Intent completeAcousticDice(const Signature& input, Intent baseline, Intent verified,
                           const AcousticDice& model){
	if(baseline != Intent::Unknown || verified != Intent::Unknown){
		return verified;
	}
	const auto scores = acousticDiceScores(input, model);
	const auto alternative = std::max({scores[0], scores[2], scores[3]});
	return scores[1] >= model.minimumScore && scores[1] - alternative >= model.minimumMargin
	       ? Intent::Dice6 : Intent::Unknown;
}
}
