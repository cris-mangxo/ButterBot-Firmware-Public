#pragma once

#include <cstdint>
#include <span>

namespace bb::asr { struct LogMel; }
namespace EsMx::Kws { struct WordFeatures; }

namespace EsMx::Pilot {

// Only call on a complete turn eligible for fallback, on the same recognition task.
bool matchesJokeFallback(bb::asr::LogMel& frontend, Kws::WordFeatures& words,
                         std::span<const int16_t> pcm);

}
