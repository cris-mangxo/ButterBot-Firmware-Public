#include <cstdio>

#ifdef KWS_WITH_MODEL
#include "Compact.h"
#include "pilot_model.h"

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
		label = static_cast<int>(result.intent);
		cost = result.cost;
	}
#endif
	std::printf("KWS probe %d %u\n", label, cost);
}
