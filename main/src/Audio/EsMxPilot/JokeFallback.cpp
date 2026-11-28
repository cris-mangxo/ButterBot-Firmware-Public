#if defined(BB_ESMX_PILOT) && defined(BB_ESMX_JOKE_FALLBACK)
#include "JokeFallback.h"

#include "JokeGuard.h"
#include "WordDice.h"
#include "joke_rule.h"
#include "student_bank.h"
#ifdef BB_ESMX_CLOCK_FALLBACK
#include "clock_rule.h"
#endif
#include <algorithm>
#include <cstring>
#ifdef BB_ESMX_BUTTER_ACOUSTIC
#include <cmath>
#endif
#include <memory>
#include <new>
#include <string_view>

#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
#endif

namespace EsMx::Pilot {
namespace {
using namespace PhonemeNative;

static_assert(std::string_view(JokeBankSha256) == std::string_view(FrozenSha256),
              "The es-MX Chiste rule requires its frozen phoneme bank");
static_assert(Bands == Kws::Bands && MaxFrames == Kws::MaxWordFrames);
static_assert(sizeof(std::array<int8_t, Bands>) == Bands);
static_assert(static_cast<unsigned>(Kws::Intent::Joke) == 3);
#ifdef BB_ESMX_CLOCK_FALLBACK
static_assert(std::string_view(ClockBankSha256) == std::string_view(FrozenSha256),
              "The es-MX Hora rule requires its frozen phoneme bank");
static_assert(static_cast<unsigned>(Kws::Intent::Time) == 5);
#endif

constexpr size_t TrimWindow = bb::asr::kSampleRate / 50;
constexpr uint64_t FloorEnergy = 90 * 90 * TrimWindow;
constexpr uint64_t RelativeGate = 512;
constexpr size_t MinSamples = 400;
constexpr auto samplesForFrames(size_t frames){
    return bb::asr::kWin + (frames - 1) * Kws::WordStride;
}
constexpr size_t JokeMinFrames = 16;
#ifdef BB_ESMX_CLOCK_FALLBACK
constexpr size_t ClockMinFrames = 12;
#endif

uint64_t blockEnergy(std::span<const int16_t> pcm, size_t offset){
    uint64_t sum = 0;
    for(const int32_t sample : pcm.subspan(offset, std::min(TrimWindow, pcm.size() - offset))){
        sum += static_cast<uint64_t>(sample * sample);
    }
    return sum;
}

std::span<const int16_t> trimTurn(std::span<const int16_t> pcm){
    if(pcm.size() < MinSamples || pcm.size() > Kws::MaxSamples){
        return {};
    }

    uint64_t peak = 0;
    for(size_t offset = 0; offset < pcm.size(); offset += TrimWindow){
        peak = std::max(peak, blockEnergy(pcm, offset));
    }
    if(peak < FloorEnergy){
        return {};
    }

    // Scale the RMS tests by the full block length. Missing samples in a
    // final partial block count as zero. Keep one block at both speech edges.
    size_t first = pcm.size();
    size_t end = 0;
    for(size_t offset = 0; offset < pcm.size(); offset += TrimWindow){
        const auto sum = blockEnergy(pcm, offset);
        if(sum >= FloorEnergy && sum * RelativeGate >= peak){
            first = std::min(first, offset);
            end = offset + 2 * TrimWindow;
        }
    }

    const size_t begin = first >= TrimWindow ? first - TrimWindow : 0;
    return pcm.subspan(begin, std::min(end, pcm.size()) - begin);
}

struct Scratch {
    std::array<int8_t, MaxFrames * Bands> features;
    Workspace workspace;
    Output output;
};

Scratch* allocateScratch(){
#ifdef ESP_PLATFORM
    void* memory = heap_caps_aligned_alloc(alignof(Scratch), sizeof(Scratch),
                                         MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    return memory ? new (memory) Scratch : nullptr;
#else
    return new (std::nothrow) Scratch;
#endif
}

void freeScratch(Scratch* scratch){
#ifdef ESP_PLATFORM
    scratch->~Scratch();
    heap_caps_free(scratch);
#else
    delete scratch;
#endif
}

#ifdef BB_ESMX_CLOCK_FALLBACK
std::span<const int16_t> stripDigitalEdges(std::span<const int16_t> pcm){
    const auto voiced = [](int16_t sample){ return sample != 0; };
    const auto first = std::find_if(pcm.begin(), pcm.end(), voiced);
    if(first == pcm.end()){
        return {};
    }
    const auto end = std::find_if(pcm.rbegin(), pcm.rend(), voiced).base();
    return {first, end};
}
#endif

size_t inferView(bb::asr::LogMel& frontend, Kws::WordFeatures& words, std::span<const int16_t> pcm,
                 size_t minimum, Scratch& scratch){
    if(pcm.size() < samplesForFrames(minimum) || !Kws::extractWordFeatures(frontend, pcm, words) ||
       words.count < minimum || words.count > MaxFrames){
        return 0;
    }
    const size_t count = words.count;
    auto features = std::span(scratch.features).first(count * Bands);
    std::memcpy(features.data(), words.frames.data(), features.size_bytes());
    return infer(features, Frozen, scratch.workspace, scratch.output) ? count : 0;
}

}

Kws::Intent recognizePhonemeFallback(bb::asr::LogMel& frontend, Kws::WordFeatures& words,
                                    std::span<const int16_t> pcm){
    using Kws::Intent;
    if(pcm.size() > Kws::MaxSamples){
        return Intent::Unknown;
    }
    const auto jokePcm = trimTurn(pcm);
    const bool jokeEligible = jokePcm.size() >= samplesForFrames(JokeMinFrames);
#ifdef BB_ESMX_CLOCK_FALLBACK
    const auto clockPcm = trimTurn(stripDigitalEdges(pcm));
    const bool clockEligible = clockPcm.size() >= samplesForFrames(ClockMinFrames);
    if(!jokeEligible && !clockEligible){
#else
    if(!jokeEligible){
#endif
        return Intent::Unknown;
    }
    std::unique_ptr<Scratch, decltype(&freeScratch)> scratch(allocateScratch(), &freeScratch);
    if(!scratch){
        return Intent::Unknown;
    }
    const auto jokeFrames = jokeEligible ? inferView(frontend, words, jokePcm, JokeMinFrames, *scratch) : 0;
    // The encoder's first work buffer is free once inference has produced its output.
    if(jokeFrames && confirmsJoke(scratch->output, jokeFrames, FrozenJoke, scratch->workspace.first)){
        return Intent::Joke;
    }
#ifdef BB_ESMX_CLOCK_FALLBACK
    if(clockEligible){
        const bool sameView = jokeFrames && jokePcm.data() == clockPcm.data() && jokePcm.size() == clockPcm.size();
        const auto clockFrames = sameView ? jokeFrames : inferView(frontend, words, clockPcm, ClockMinFrames, *scratch);
        if(clockFrames){
            const auto score = scoreClock(scratch->output, clockFrames, FrozenClock, scratch->workspace.first);
            if(score && *score > FrozenClock.threshold){
                return Intent::Time;
            }
        }
    }
#endif
    return Intent::Unknown;
}

#ifdef BB_ESMX_BUTTER_ACOUSTIC
ButterPhonemeEvidence measureButterPhonemes(bb::asr::LogMel& frontend, Kws::WordFeatures& words,
                                          std::span<const int16_t> pcm){
    static_assert(std::string_view(FrozenSha256) ==
                  "1785628e682569e77cdc4e934da8714cf5021b518a2bf745abc76aa24cfd0619");
    constexpr size_t MinFrames = 8;
    constexpr size_t ButterSlot = 1;
    const auto trimmed = trimTurn(pcm);
    if(trimmed.size() < samplesForFrames(MinFrames)){
        return {};
    }
    std::unique_ptr<Scratch, decltype(&freeScratch)> scratch(allocateScratch(), &freeScratch);
    if(!scratch || !inferView(frontend, words, trimmed, MinFrames, *scratch)){
        return {};
    }
    const auto& output = scratch->output;
    if(!std::isfinite(output.negation) ||
       !std::all_of(output.intents.begin(), output.intents.end(), [](float value){ return std::isfinite(value); })){
        return {};
    }
    const double peak = *std::max_element(output.intents.begin(), output.intents.end());
    double sum = 0;
    double butter = 0;
    double other = 0;
    for(size_t index = 0; index < output.intents.size(); ++index){
        const double weight = std::exp(static_cast<double>(output.intents[index]) - peak);
        sum += weight;
        if(index == ButterSlot){
            butter = weight;
        }else{
            other = std::max(other, weight);
        }
    }
    return {butter / sum, (butter - other) / sum, output.negation, true};
}

bool confirmsButterPhonemes(const ButterPhonemeEvidence& head){
    return head.valid && head.posterior >= 0.80 && head.margin >= 0.30 && head.negation < -0.5f;
}
#endif

}
#endif
