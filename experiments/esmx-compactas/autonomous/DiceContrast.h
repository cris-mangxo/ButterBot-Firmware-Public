#pragma once

#include "AcousticDice.h"
#include "WordDice.h"
#include <algorithm>

namespace EsMx::Kws {

inline Intent completeContrastedDice(const Signature& input, Intent baseline, Intent verified,
                                     const AcousticDice& acoustic, const WordFeatures& words,
                                     const WordBank& lexicon){
	if(baseline != Intent::Unknown || verified != Intent::Unknown){
		return verified;
	}
	const auto scores = acousticDiceScores(input, acoustic);
	const auto alternative = std::max({scores[0], scores[2], scores[3]});
	if(scores[1] >= acoustic.minimumScore && scores[1] - alternative >= acoustic.minimumMargin &&
	   confirmContinuousDice(verified, Intent::Dice6, words, lexicon, DiceConfirmation::OrderedRescue) == Intent::Dice6){
		return Intent::Dice6;
	}
	// Resolve only confusion with another die. In particular, do not override
	// lexical rejection when the CNN already prefers D6.
	constexpr int32_t OutsideGap = 7000, OtherDieGap = -5500, NegatedGap = 6000;
	if(scores[2] > scores[1] && scores[1] - scores[2] >= OtherDieGap &&
	   scores[1] - scores[0] >= OutsideGap &&
	   scores[1] - scores[3] >= NegatedGap && acceptsContrastedDice(words, lexicon)){
		return Intent::Dice6;
	}
	return Intent::Unknown;
}
}
