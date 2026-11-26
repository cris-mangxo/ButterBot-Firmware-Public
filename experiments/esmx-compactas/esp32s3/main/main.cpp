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
#endif

bb::asr::LogMel frontend;
Signature signature;
int16_t probePcm[8000];
#endif

extern "C" void app_main(){
	int label = 0;
	unsigned cost = 0;
#ifdef KWS_WITH_MODEL
	for(unsigned i = 0; i < std::size(probePcm); ++i){
		probePcm[i] = (int(i % 56) - 28) * 140;
	}
	if(bb::asr::logmel_init(&frontend) && extract(frontend, probePcm, signature)){
		const auto result = classify(signature, PilotModel, PilotLimits);
#ifdef KWS_WITH_DICE6_ANCHORS
		label = static_cast<int>(confirmDice6(signature, result.intent, PilotDice6Anchors).intent);
#elif defined(KWS_WITH_PHRASE_BANK)
		label = static_cast<int>(verifyPhrase(signature, result.intent, PilotPhrases));
#else
		label = static_cast<int>(result.intent);
#endif
		cost = result.cost;
	}
#endif
	std::printf("KWS probe %d %u\n", label, cost);
}
