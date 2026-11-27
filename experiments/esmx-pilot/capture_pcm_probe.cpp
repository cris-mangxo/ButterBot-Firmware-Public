#include "capture_pcm_probe.h"

#include "Audio/EsMxPilot/Recognition.h"
#include <vector>

namespace {
bool enabled = false;
std::vector<int16_t> samples;
}

namespace EsMx::Pilot {
Kws::Intent recognizeKernel(std::span<const int16_t> pcm);

Kws::Intent recognize(std::span<const int16_t> pcm){
    if(enabled){
        samples.assign(pcm.begin(), pcm.end());
    }
    return recognizeKernel(pcm);
}
}

void enablePcmProbe(){
    enabled = true;
}

std::span<const int16_t> recordedPcm(){
    return samples;
}
