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

#include <algorithm>

namespace EsMx::Pilot {
namespace {

RecognitionDecision recognizeTurn(std::span<const int16_t> pcm){
	using namespace Kws;
	static bb::asr::LogMel frontend;
	static WordFeatures words;
	static const bool ready = bb::asr::logmel_init(&frontend);
	constexpr const WordBank* supplements[]{&PilotWordExtra, &PilotWordAdapted};

	Signature signature;
	if(!ready || !extract(frontend, pcm, signature)){
		return {Intent::Unknown, TurnDisposition::InvalidAudio};
	}

	const auto baseline = classify(signature, PilotModel, PilotLimits).intent;
	const auto verified = verifyPhrase(signature, baseline, PilotPhrases);
	words.count = 0;
	const auto wordIntent = completeWordDice(frontend, pcm, baseline, verified, PilotWordDice, words, supplements);
	const auto dice = resolveConsensusDice(signature, baseline, wordIntent, PilotAcousticDice, PilotConsensusDice,
	                                      words, PilotWordDice, PilotWordExtra, PilotWordAdapted,
	                                      ConsensusMode::OrderedDie, &PilotBackupDice);
	if(dice.intent != Intent::Unknown){
		return {dice.intent, TurnDisposition::Accepted};
	}
	if(verified == Intent::Unknown && baseline != Intent::Unknown){
		switch(baseline){
			case Intent::Fact: return {Intent::Unknown, TurnDisposition::FactVeto};
			case Intent::Dice6: return {Intent::Unknown, TurnDisposition::UnconfirmedDice};
			default: return {Intent::Unknown, TurnDisposition::PhraseRejected};
		}
	}
	switch(dice.rejection){
		case DiceRejection::NegationVeto: return {Intent::Unknown, TurnDisposition::DiceNegationVeto};
		case DiceRejection::NegationConflict: return {Intent::Unknown, TurnDisposition::DiceNegationConflict};
		case DiceRejection::None: return {Intent::Unknown, TurnDisposition::NoMatch};
	}
	return {Intent::Unknown, TurnDisposition::InvalidAudio};
}

bool hasSpeechEnergy(std::span<const int16_t> frame){
	constexpr uint64_t RmsFloor = 180;
	uint64_t energy = 0;
	for(const int32_t sample : frame){
		energy += static_cast<uint64_t>(sample * sample);
	}
	return energy >= frame.size() * RmsFloor * RmsFloor;
}

bool hasAmbiguousDiceFollowUp(std::span<const int16_t> pcm){
	constexpr size_t FrameSamples = bb::asr::kSampleRate / 50;
	constexpr size_t FollowUpGap = 12 * FrameSamples;
	constexpr unsigned MaxPrefixChecks = 2;
	size_t lastVoiceEnd = 0;
	unsigned checks = 0;
	for(size_t offset = 0; offset < pcm.size(); offset += FrameSamples){
		const auto frame = pcm.subspan(offset, std::min(FrameSamples, pcm.size() - offset));
		if(!hasSpeechEnergy(frame)){
			continue;
		}
		if(lastVoiceEnd && offset - lastVoiceEnd >= FollowUpGap){
			// A completed D6 request followed by separate speech is ambiguous.
			// Replay only complete prefixes and cap extra inference on the MCU.
			if(checks == MaxPrefixChecks || recognizeTurn(pcm.first(lastVoiceEnd)).intent == Kws::Intent::Dice6){
				return true;
			}
			++checks;
		}
		lastVoiceEnd = offset + frame.size();
	}
	return false;
}

}

RecognitionDecision recognizeDetailed(std::span<const int16_t> pcm){
	const auto decision = recognizeTurn(pcm);
	if(decision.intent == Kws::Intent::Dice6 && hasAmbiguousDiceFollowUp(pcm)){
		return {Kws::Intent::Unknown, TurnDisposition::DiceFollowUp};
	}
	return decision;
}

Kws::Intent recognize(std::span<const int16_t> pcm){
	return recognizeDetailed(pcm).intent;
}

}
#endif
