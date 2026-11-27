#ifdef BB_ESMX_PILOT
#include "Recognition.h"

#include "DiceConsensus.h"
#include "phrase_bank.h"
#include "pilot_model.h"
#include "word_bank.h"
#include "word_extra.h"
#include "word_adapted.h"
#include "acoustic_dice.h"
#include "consensus_acoustic_dice.h"
#include "backup_acoustic_dice.h"

namespace EsMx::Pilot {

Kws::Intent recognize(std::span<const int16_t> pcm){
	using namespace Kws;
	static bb::asr::LogMel frontend;
	static WordFeatures words;
	static const bool ready = bb::asr::logmel_init(&frontend);
	constexpr const WordBank* supplements[]{&PilotWordExtra, &PilotWordAdapted};

	Signature signature;
	if(!ready || !extract(frontend, pcm, signature)){
		return Intent::Unknown;
	}

	const auto baseline = classify(signature, PilotModel, PilotLimits).intent;
	const auto verified = verifyPhrase(signature, baseline, PilotPhrases);
	words.count = 0;
	const auto wordIntent = completeWordDice(frontend, pcm, baseline, verified, PilotWordDice, words, supplements);
	return completeConsensusDice(signature, baseline, wordIntent, PilotAcousticDice, PilotConsensusDice, words,
	                             PilotWordDice, PilotWordExtra, PilotWordAdapted, ConsensusMode::OrderedDie,
	                             &PilotBackupDice);
}

}
#endif
