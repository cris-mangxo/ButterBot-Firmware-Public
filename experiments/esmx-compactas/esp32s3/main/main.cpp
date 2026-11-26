#include <cstdio>

#ifdef KWS_WITH_MODEL
#include "Compact.h"
#include "pilot_model.h"
#ifdef KWS_WITH_DICE6_ANCHORS
#include "Dice6Guard.h"
#include "dice6_anchors.h"
#elif defined(KWS_WITH_PHRASE_BANK)
#include "PhraseGate.h"
#include "phrase_bank.h"
#ifdef KWS_WITH_TRANSCRIPT_GRAMMAR
#include "DiceGrammar.h"
#elif defined(KWS_WITH_WORD_DICE)
#include "WordDice.h"
#include "word_bank.h"
#ifdef KWS_WITH_WORD_DICE_EXTRA
#include "word_extra.h"
#ifdef KWS_WITH_WORD_DICE_ADAPTED
#include "word_adapted.h"
#endif
static constexpr const WordBank* WordSupplements[]{
	&PilotWordExtra,
#ifdef KWS_WITH_WORD_DICE_ADAPTED
	&PilotWordAdapted,
#endif
};
#else
static constexpr std::span<const WordBank* const> WordSupplements;
#endif
#ifdef KWS_WITH_WORD_TEST_PCM
#include "word_pcm.h"
#endif
static_assert(sizeof(WordBank) == 9140);
EsMx::Kws::WordFeatures wordWorkspace;
#endif
#endif

bb::asr::LogMel frontend;
Signature signature;
int16_t probePcm[8000];

static int probe(std::span<const int16_t> pcm, unsigned& cost){
	int label = 0;
	if(extract(frontend, pcm, signature)){
		const auto result = classify(signature, PilotModel, PilotLimits);
#ifdef KWS_WITH_DICE6_ANCHORS
		label = static_cast<int>(confirmDice6(signature, result.intent, PilotDice6Anchors).intent);
#elif defined(KWS_WITH_PHRASE_BANK)
		const auto verified = verifyPhrase(signature, result.intent, PilotPhrases);
#ifdef KWS_WITH_TRANSCRIPT_GRAMMAR
		label = static_cast<int>(completeDice6(result.intent, verified, "Podrías lanzarme un dado de seis."));
#elif defined(KWS_WITH_WORD_DICE)
		label = static_cast<int>(completeWordDice(frontend, pcm, result.intent, verified,
		                                         PilotWordDice, wordWorkspace, WordSupplements));
#else
		label = static_cast<int>(verified);
#endif
#else
		label = static_cast<int>(result.intent);
#endif
		cost = result.cost;
	}
	return label;
}
#endif

extern "C" void app_main(){
	int label = 0;
	unsigned cost = 0;
#ifdef KWS_WITH_MODEL
	if(!bb::asr::logmel_init(&frontend)){
		std::printf("KWS frontend error\n");
		return;
	}
#ifdef KWS_WITH_WORD_TEST_PCM
	unsigned failures = 0;
	for(const auto& test : WordPcmCases){
		label = probe(test.samples, cost);
		std::printf("WORD_D6_CASE %s %d %d\n", test.name, test.expected, label);
		failures += label != test.expected;
	}
	std::printf("WORD_D6_DONE %u\n", failures);
	return;
#else
	for(unsigned i = 0; i < std::size(probePcm); ++i){
		probePcm[i] = (int(i % 56) - 28) * 140;
	}
	label = probe(probePcm, cost);
#endif
#endif
	std::printf("KWS probe %d %u\n", label, cost);
}
