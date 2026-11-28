#pragma once

#include "AcousticDice.h"
#include "WordDice.h"
#include <algorithm>
#include <cstdint>

namespace EsMx::Kws {

enum class ConsensusMode { SharedSix, OrderedDie };

enum class DiceRejection : std::uint8_t { None, NegationConflict, NegationVeto };

struct DiceCompletion {
	Intent intent;
	DiceRejection rejection = DiceRejection::None;
	bool orderedRescue = false;
};

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

inline DiceCompletion resolveConsensusDice(const Signature& input, Intent baseline, Intent verified,
                                            const AcousticDice& reference, const AcousticDice& candidate,
                                            const WordFeatures& words, const WordBank& primary,
                                            const WordBank& extra, const WordBank& adapted,
                                            ConsensusMode mode, const AcousticDice* backup){
	if(baseline != Intent::Unknown || (verified != Intent::Unknown && verified != Intent::Dice6)){
		return {verified};
	}
	const auto referenceScores = acousticDiceScores(input, reference);
	// Correct a supplementary lexical activation only when both whole-turn
	// banks prefer negation. A complete match in the primary lexicon wins.
	if(verified == Intent::Dice6){
		const bool vetoed = prefersNegation(referenceScores, reference) &&
		                    prefersNegation(acousticDiceScores(input, candidate), candidate) &&
		                    !acceptsWordDice(words, primary);
		return vetoed ? DiceCompletion{Intent::Unknown, DiceRejection::NegationVeto}
		              : DiceCompletion{verified};
	}
	// Preserve decisions already confirmed by the reference.
	if(proposesDice6(referenceScores, reference) &&
	   confirmContinuousDice(verified, Intent::Dice6, words, adapted,
	                         DiceConfirmation::OrderedRescue) == Intent::Dice6){
		return {.intent = Intent::Dice6, .orderedRescue = true};
	}
	// Preserve the shared-six veto. The ordered route can examine weak
	// competition below the reference bank's absolute score floor.
	const bool competing = hasCompetingNegation(referenceScores, reference);
	if(competing && (mode == ConsensusMode::SharedSix || prefersNegation(referenceScores, reference) ||
	                 referenceScores[DiceNegation] >= reference.minimumScore)){
		return {Intent::Unknown, DiceRejection::NegationVeto};
	}
	const auto candidateScores = acousticDiceScores(input, candidate);
	const bool proposed = proposesDice6(candidateScores, candidate);
	if(proposed && !competing && confirmsSixAcrossBanks(words, primary, extra, adapted)){
		return {Intent::Dice6};
	}
	// Only try the fallback on an abstention without acoustic competition from negation.
	// Check the lexicons first to avoid the extra CNN on unrelated turns.
	if(mode == ConsensusMode::OrderedDie &&
	   (proposed || (backup && !competing && !hasCompetingNegation(candidateScores, candidate))) &&
	   confirmsOrderedDieSix(words, primary, extra, adapted)){
		if(proposed || proposesDice6(acousticDiceScores(input, *backup), *backup)){
			return {.intent = Intent::Dice6, .orderedRescue = true};
		}
	}
	const auto rejection = competing || hasCompetingNegation(candidateScores, candidate)
	                     ? DiceRejection::NegationConflict : DiceRejection::None;
	return {Intent::Unknown, rejection};
}

inline Intent completeConsensusDice(const Signature& input, Intent baseline, Intent verified,
                                    const AcousticDice& reference, const AcousticDice& candidate,
                                    const WordFeatures& words, const WordBank& primary,
                                    const WordBank& extra, const WordBank& adapted,
                                    ConsensusMode mode = ConsensusMode::SharedSix,
                                    const AcousticDice* backup = nullptr){
	return resolveConsensusDice(input, baseline, verified, reference, candidate,
	                            words, primary, extra, adapted, mode, backup).intent;
}
}
