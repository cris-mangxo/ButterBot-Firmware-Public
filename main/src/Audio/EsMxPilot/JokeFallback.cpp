#if defined(BB_ESMX_PILOT) && defined(BB_ESMX_JOKE_FALLBACK)
#include "JokeFallback.h"

#include "JokeGuard.h"
#include "WordDice.h"
#include "joke_rule.h"
#include "student_bank.h"
#include <algorithm>
#include <cstring>
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

constexpr size_t TrimWindow = bb::asr::kSampleRate / 50;
constexpr uint64_t FloorEnergy = 90 * 90 * TrimWindow;
constexpr uint64_t RelativeGate = 512;
constexpr size_t MinSamples = 400;
constexpr size_t MinFrames = 16;
constexpr size_t MinFeatureSamples = bb::asr::kWin + (MinFrames - 1) * Kws::WordStride;

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

}

bool matchesJokeFallback(bb::asr::LogMel& frontend, Kws::WordFeatures& words,
                         std::span<const int16_t> pcm){
    const auto trimmed = trimTurn(pcm);
    if(trimmed.size() < MinFeatureSamples){
        return false;
    }

    std::unique_ptr<Scratch, decltype(&freeScratch)> scratch(allocateScratch(), &freeScratch);
    if(!scratch || !Kws::extractWordFeatures(frontend, trimmed, words)){
        return false;
    }

    const size_t count = words.count;
    if(count < MinFrames || count > MaxFrames){
        return false;
    }
    auto features = std::span(scratch->features).first(count * Bands);
    std::memcpy(features.data(), words.frames.data(), features.size_bytes());
    return infer(features, Frozen, scratch->workspace, scratch->output) &&
           confirmsJoke(scratch->output, count, FrozenJoke);
}

}
#endif
