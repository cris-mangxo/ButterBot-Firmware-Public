#pragma once

#include <cstdint>
#include <span>

namespace EsMx::Pilot {

// Called only by the single recognition worker after an eligible abstention.
bool recognizesAcousticClock(std::span<const int16_t> pcm);

#ifdef ESP_PLATFORM
bool prepareAcousticClock();
void startAcousticClockWarmup();
#endif

}
