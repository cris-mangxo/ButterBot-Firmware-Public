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

#ifdef BB_ESMX_BUTTER_ACOUSTIC
struct ButterPhonemeEvidence {
    double posterior = 0;
    double margin = 0;
    float negation = 0;
    bool valid = false;
};

ButterPhonemeEvidence measureButterPhonemes(bb::asr::LogMel& frontend, Kws::WordFeatures& words,
                                          std::span<const int16_t> pcm);
bool confirmsButterPhonemes(const ButterPhonemeEvidence& head);
#endif

}
