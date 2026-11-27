#pragma once

#include "Audio/EsMxKws/Compact.h"

namespace EsMx::Pilot {

// Call from the pilot's single recognition task with a complete, 16 kHz mono turn.
Kws::Intent recognize(std::span<const int16_t> pcm);

}
