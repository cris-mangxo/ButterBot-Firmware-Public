#pragma once

#include "Audio/EsMxKws/Compact.h"

#include <string_view>

namespace EsMx::Kws {

constexpr size_t MaxTranscriptBytes = 192;

// Complete, single affirmative request for one six-sided die.
bool isDice6Request(std::string_view transcript);

// A previously accepted intent or an acoustic veto is terminal.
Intent completeDice6(Intent baseline, Intent verified, std::string_view transcript);

}
