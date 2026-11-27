#pragma once

#include "AcousticDice.h"
#include "WordDice.h"
#include <algorithm>

namespace EsMx::Kws {

enum class ConsensusMode { SharedSix, OrderedDie };

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
                                    const WordBank& extra, const WordBank& adapted,
                                    ConsensusMode mode = ConsensusMode::SharedSix){
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
	// Preserve the shared-six veto. The ordered route can examine weak
	// competition below the reference bank's absolute score floor.
	const bool competing = hasCompetingNegation(referenceScores, reference);
	if(competing && (mode == ConsensusMode::SharedSix || prefersNegation(referenceScores, reference) ||
	                 referenceScores[DiceNegation] >= reference.minimumScore)){
		return Intent::Unknown;
	}
	if(!proposesDice6(acousticDiceScores(input, candidate), candidate)){
		return Intent::Unknown;
	}
	if((!competing && confirmsSixAcrossBanks(words, primary, extra, adapted)) ||
	   (mode == ConsensusMode::OrderedDie && confirmsOrderedDieSix(words, primary, extra, adapted))){
		return Intent::Dice6;
	}
	return Intent::Unknown;
}
}
