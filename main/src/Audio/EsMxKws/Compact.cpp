#include "Compact.h"
#include "cuantizar.h"

#include <algorithm>
#include <limits>

namespace EsMx::Kws {
namespace {
constexpr int Hop = bb::asr::kHop;
constexpr int Window = bb::asr::kWin;
constexpr int MelBands = bb::asr::kBands;
static_assert(MelBands == 2 * Bands);
constexpr uint32_t Infinite = std::numeric_limits<uint32_t>::max() / 4;
constexpr uint32_t FloorEnergy = 180 * 180;
constexpr int Warp = 3;

uint32_t energy(std::span<const int16_t> pcm){
	uint64_t sum = 0;
	for(const int sample : pcm){
		sum += sample * sample;
	}
	return static_cast<uint32_t>(sum / pcm.size());
}

uint32_t distance(const Signature& input, const Signature& example){
	std::array<uint32_t, Frames + 1> previous, current;
	previous.fill(Infinite);
	previous[0] = 0;
	for(int i = 1; i <= Frames; ++i){
		current.fill(Infinite);
		for(int j = std::max(1, i - Warp); j <= std::min(Frames, i + Warp); ++j){
			unsigned cost = 0;
			for(int band = 0; band < Bands; ++band){
				const int difference = int(input.frames[i - 1][band]) - int(example.frames[j - 1][band]);
				cost += difference < 0 ? -difference : difference;
			}
			current[j] = cost + std::min({previous[j], previous[j - 1], current[j - 1]});
		}
		previous = current;
	}
	return previous[Frames];
}
}

bool extract(bb::asr::LogMel& frontend, std::span<const int16_t> pcm, Signature& output){
	if(!frontend.ready || pcm.size() > MaxSamples || pcm.size() < Window){
		return false;
	}
	const size_t windows = bb::asr::logmel_num_frames(pcm.size());
	uint32_t peak = 0;
	for(size_t i = 0; i < windows; ++i){
		peak = std::max(peak, energy(pcm.subspan(i * Hop, Window)));
	}
	if(peak < FloorEnergy){
		return false;
	}
	const uint32_t gate = std::max(FloorEnergy, peak / 128);
	size_t begin = 0;
	size_t end = windows;
	while(begin < end && energy(pcm.subspan(begin * Hop, Window)) < gate){
		++begin;
	}
	while(end > begin && energy(pcm.subspan((end - 1) * Hop, Window)) < gate){
		--end;
	}
	const size_t voiced = end - begin;
	if(voiced < Frames){
		return false;
	}
	for(int frame = 0; frame < Frames; ++frame){
		const size_t first = begin + voiced * frame / Frames;
		const size_t last = begin + voiced * (frame + 1) / Frames;
		float bins[Bands]{};
		for(size_t i = first; i < last; ++i){
			float mel[MelBands];
			bb::asr::logmel_frame(&frontend, pcm.data() + i * Hop, mel);
			float mean = 0;
			for(float value : mel){
				mean += value;
			}
			mean /= MelBands;
			for(int band = 0; band < Bands; ++band){
				bins[band] += (mel[2 * band] + mel[2 * band + 1]) * 0.5f - mean;
			}
		}
		for(int band = 0; band < Bands; ++band){
			output.frames[frame][band] = bb::asr::quantize_value(bins[band] / (last - first), 0.125, 0);
		}
	}
	return true;
}

Result classify(const Signature& input, std::span<const Prototype> model, Limits limits){
	constexpr auto count = static_cast<size_t>(Intent::Count);
	std::array<uint32_t, count> costs;
	costs.fill(Infinite);
	if(model.size() > MaxTemplates){
		return {Intent::Unknown, Infinite, 0};
	}
	for(const Prototype& example : model){
		const auto label = static_cast<size_t>(example.intent);
		if(label >= count){
			return {Intent::Unknown, Infinite, 0};
		}
		costs[label] = std::min(costs[label], distance(input, example.signature));
	}
	if(std::find(costs.begin(), costs.end(), Infinite) != costs.end()){
		return {Intent::Unknown, Infinite, 0};
	}
	const auto best = std::min_element(costs.begin(), costs.end());
	const auto label = static_cast<Intent>(best - costs.begin());
	const uint32_t cost = *best;
	*best = Infinite;
	const uint32_t second = *std::min_element(costs.begin(), costs.end());
	if(label == Intent::Unknown || cost > limits.maxCost || second - cost < limits.minGap || second == cost){
		return {Intent::Unknown, cost, second - cost};
	}
	return {label, cost, second - cost};
}
}
