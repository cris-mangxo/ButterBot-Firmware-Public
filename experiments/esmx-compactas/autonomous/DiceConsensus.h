#pragma once

#include "AcousticDice.h"
#include "WordDice.h"
#include <algorithm>

namespace EsMx::Kws {

inline bool prefersNegation(const DiceScores& scores, const AcousticDice& bank){
	const auto alternative = std::max({scores[DiceOutside], scores[DiceOther], scores[DiceRequest]});
	return scores[DiceNegation] - alternative >= bank.minimumMargin;
}

inline bool hasCompetingNegation(const DiceScores& scores, const AcousticDice& bank){
	const auto requestLead = scores[DiceRequest] - scores[DiceNegation];
	return requestLead <= -bank.minimumMargin ||
	       (scores[DiceNegation] > std::max(scores[DiceOutside], scores[DiceOther]) &&
	        requestLead < bank.minimumMargin);
}

inline Intent completeConsensusDice(const Signature& input, Intent baseline, Intent verified,
                                    const AcousticDice& reference, const AcousticDice& candidate,
                                    const WordFeatures& words, const WordBank& primary,
                                    const WordBank& extra, const WordBank& adapted){
	if(baseline != Intent::Unknown || (verified != Intent::Unknown && verified != Intent::Dice6)){
		return verified;
	}
	const auto referenceScores = acousticDiceScores(input, reference);
	// Correct a supplementary lexical activation only when both whole-turn
	// banks prefer negation. A complete match in the primary lexicon wins.
	if(verified == Intent::Dice6){
		return prefersNegation(referenceScores, reference) &&
		       prefersNegation(acousticDiceScores(input, candidate), candidate) &&
		       !acceptsWordDice(words, primary)
		       ? Intent::Unknown : verified;
	}
	// Preserve decisions already confirmed by the reference.
	if(proposesDice6(referenceScores, reference) &&
	   confirmContinuousDice(verified, Intent::Dice6, words, adapted,
	                         DiceConfirmation::OrderedRescue) == Intent::Dice6){
		return Intent::Dice6;
	}
	// A competitive negation in the reference blocks the secondary route even
	// when the candidate proposes D6.
	if(hasCompetingNegation(referenceScores, reference)){
		return Intent::Unknown;
	}
	if(proposesDice6(acousticDiceScores(input, candidate), candidate) &&
	   confirmsSixAcrossBanks(words, primary, extra, adapted)){
		return Intent::Dice6;
	}
	return Intent::Unknown;
}
}
