#ifdef BB_ESMX_CLOCK_ACOUSTIC
#include "ClockAcoustic.h"
#include "AcousticSearch.h"
#if (defined(BB_ESMX_D6_SCOPE_HOST) && !defined(ESP_PLATFORM)) || defined(BB_ESMX_D6_SCOPE_BENCH)
#include "DiceScope.h"
#endif

#include "Audio/EsMxKws/Compact.h"
#include "ClockGrammar.h"
#include <cstdlib>
#include <memory>
#include <pocketsphinx.h>
#include <string>
#include <string_view>

#ifdef ESP_PLATFORM
#include "ClockModelVfs.h"
#include "esp_heap_caps.h"
#include <atomic>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

namespace {
constexpr uint32_t ClockMemoryCaps = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
}

extern "C" void* bb_clock_malloc(size_t bytes) {
    return heap_caps_malloc(bytes, ClockMemoryCaps);
}
extern "C" void* bb_clock_calloc(size_t count, size_t bytes) {
    return heap_caps_calloc(count, bytes, ClockMemoryCaps);
}
extern "C" void* bb_clock_realloc(void* data, size_t bytes) {
    return heap_caps_realloc(data, bytes, ClockMemoryCaps);
}
#endif

namespace EsMx::Pilot {
#ifdef BB_ESMX_BUTTER_ACOUSTIC
bool installButterAcousticSearches(ps_decoder_t* decoder, const char* keywords);
#endif
namespace {

using Config = std::unique_ptr<ps_config_t, decltype(&ps_config_free)>;
using Decoder = std::unique_ptr<ps_decoder_t, decltype(&ps_free)>;

#ifdef BB_ESMX_D6_SCOPE_BENCH
bool diceScopeInstalled = false;
#endif

Decoder createDecoder() {
#ifdef ESP_PLATFORM
    // PocketSphinx treats allocation failure as fatal. Keep this optional path
    // dormant if its measured working set cannot be warmed in external RAM.
    constexpr size_t MinimumExternalHeap = 3 * 1024 * 1024;
    if (heap_caps_get_free_size(ClockMemoryCaps) < MinimumExternalHeap ||
        heap_caps_get_largest_free_block(ClockMemoryCaps) < 512 * 1024) {
        return {nullptr, &ps_free};
    }
    const char* root = mountClockModel();
#else
    const char* root = std::getenv("BB_ESMX_CLOCK_ACOUSTIC_DIR");
#endif
    if (!root || root[0] == '\0') {
        return {nullptr, &ps_free};
    }

#ifdef BB_ESMX_BUTTER_ACOUSTIC
    const std::string dictionary = std::string(root) + "/butter-clock.dict";
#else
    const std::string dictionary = std::string(root) + "/clock.dict";
#endif
    const Config config(ps_config_init(nullptr), &ps_config_free);
    if (!config) {
        return {nullptr, &ps_free};
    }
    ps_config_set_str(config.get(), "hmm", root);
    ps_config_set_str(config.get(), "dict", dictionary.c_str());
    ps_config_set_str(config.get(), "lm", nullptr);
    ps_config_set_str(config.get(), "loglevel", "ERROR");
    ps_config_set_bool(config.get(), "mmap", 0);
    ps_config_set_float(config.get(), "samprate", bb::asr::kSampleRate);

    Decoder decoder(ps_init(config.get()), &ps_free);
    if (!decoder || ps_add_jsgf_string(decoder.get(), "clock", ClockGrammar) < 0 ||
        ps_activate_search(decoder.get(), "clock") < 0) {
        return {nullptr, &ps_free};
    }
#ifdef BB_ESMX_BUTTER_ACOUSTIC
    const std::string keywords = std::string(root) + "/butter.kws";
#ifdef BB_ESMX_BUTTER_APP_EXPERIMENT
    // The last search installed is the opposing FSG. Its readiness is checked
    // before each Butter decode, so a partial installation leaves Hora usable.
    installButterAcousticSearches(decoder.get(), keywords.c_str());
#else
    if (!installButterAcousticSearches(decoder.get(), keywords.c_str())) {
        return {nullptr, &ps_free};
    }
#endif
#endif
#if (defined(BB_ESMX_D6_SCOPE_HOST) && !defined(ESP_PLATFORM)) || defined(BB_ESMX_D6_SCOPE_BENCH)
    const std::string diceKeywords = std::string(root) + "/dice-scope.kws";
    // An unavailable optional D6 pack leaves the existing decoder usable.
#ifdef BB_ESMX_D6_SCOPE_BENCH
    diceScopeInstalled = installDiceScopeSearches(decoder.get(), diceKeywords.c_str());
#else
#ifdef BB_ESMX_D6_RESCUE_WITNESS_HOST
    if (installDiceScopeSearches(decoder.get(), diceKeywords.c_str())) {
        const std::string witnessPath = std::string(root) + "/dice-rescue-witness.kws";
        installDiceRescueWitness(decoder.get(), witnessPath.c_str());
    }
#else
    installDiceScopeSearches(decoder.get(), diceKeywords.c_str());
#endif
#endif
#endif
    return decoder;
}

const Decoder& decoderInstance() {
    static const auto decoder = createDecoder();
    return decoder;
}

#ifdef ESP_PLATFORM
std::atomic<ps_decoder_t*> readyDecoder{nullptr};
#endif

}

#ifdef ESP_PLATFORM
#ifdef BB_ESMX_D6_SCOPE_BENCH
bool isDiceScopeInstalledInBench() {
    return diceScopeInstalled;
}
#endif

bool prepareAcousticClock() {
    auto* decoder = decoderInstance().get();
    readyDecoder.store(decoder, std::memory_order_release);
    return decoder != nullptr;
}

void startAcousticClockWarmup() {
    constexpr uint32_t StackBytes = 8 * 1024;
    xTaskCreate([](void*) {
        prepareAcousticClock();
        vTaskDelete(nullptr);
    }, "esmx_clock_load", StackBytes, nullptr, tskIDLE_PRIORITY + 1, nullptr);
}
#endif

ps_decoder_t* sharedAcousticDecoder(){
#ifdef ESP_PLATFORM
    return readyDecoder.load(std::memory_order_acquire);
#else
    return decoderInstance().get();
#endif
}

bool decodeAcoustic(ps_decoder_t* decoder, const char* search, std::span<const int16_t> pcm){
    if(!decoder || ps_activate_search(decoder, search) < 0 || ps_start_utt(decoder) < 0){
        return false;
    }
    const auto processed = ps_process_raw(decoder, pcm.data(), pcm.size(), 0, 1);
    const auto ended = ps_end_utt(decoder);
    return processed >= 0 && ended >= 0;
}

bool recognizesAcousticClock(std::span<const int16_t> pcm){
    if(pcm.empty() || pcm.size() > Kws::MaxSamples){
        return false;
    }
    auto* decoder = sharedAcousticDecoder();
    if(!decodeAcoustic(decoder, "clock", pcm)){
        return false;
    }
    const auto* hypothesis = ps_get_hyp(decoder, nullptr);
    return hypothesis && std::string_view(hypothesis) == "qué hora es";
}

}
#endif
