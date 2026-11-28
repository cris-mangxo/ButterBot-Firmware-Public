#pragma once

#include <cstdint>
#include <span>

namespace bb::asr { struct LogMel; }
namespace EsMx::Kws {
enum class Intent : uint8_t;
struct WordFeatures;
}

namespace EsMx::Pilot {

// Only call on a complete turn eligible for fallback, on the same recognition task.
Kws::Intent recognizePhonemeFallback(bb::asr::LogMel& frontend, Kws::WordFeatures& words,
                                    std::span<const int16_t> pcm);

}
