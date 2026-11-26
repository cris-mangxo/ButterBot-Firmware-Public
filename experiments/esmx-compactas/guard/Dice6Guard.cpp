#include "Dice6Guard.h"

#include <algorithm>

namespace EsMx::Kws {

Dice6Decision guardDice6(const Signature& signature, Intent baseline, const Dice6Model& guard){
	if(baseline != Intent::Unknown && baseline != Intent::Dice6){
		return {baseline, 0};
	}
	std::array<int32_t, GuardHidden> hidden;
	for(size_t unit = 0; unit < GuardHidden; ++unit){
		int64_t value = guard.inputBias[unit];
		for(int frame = 0; frame < Frames; ++frame){
			for(int band = 0; band < Bands; ++band){
				value += int32_t(signature.frames[frame][band]) * guard.input[unit][frame * Bands + band];
			}
		}
		hidden[unit] = static_cast<int32_t>(std::max<int64_t>(value, 0) / GuardDivisor);
	}
	std::array<int64_t, GuardClasses> logits;
	for(size_t label = 0; label < GuardClasses; ++label){
		int64_t value = guard.outputBias[label];
		for(size_t unit = 0; unit < GuardHidden; ++unit){
			value += int32_t(guard.output[label][unit]) * hidden[unit];
		}
		logits[label] = value;
	}
	const int64_t margin = logits[1] - std::max({logits[0], logits[2], logits[3]});
	const bool accepted = guard.minMargin > 0 && margin >= guard.minMargin;
	return {accepted ? Intent::Dice6 : Intent::Unknown, margin};
}

Dice6Decision confirmDice6(const Signature& signature, Intent baseline, const Dice6Anchors& anchors){
	if(baseline != Intent::Unknown && baseline != Intent::Dice6){
		return {baseline, 0};
	}
	const auto cost = std::min(signatureDistance(signature, anchors.examples[0]),
	                           signatureDistance(signature, anchors.examples[1]));
	const auto margin = int64_t(anchors.maxCost) - cost;
	return {anchors.maxCost > 0 && margin >= 0 ? Intent::Dice6 : Intent::Unknown, margin};
}
}
