#ifdef BB_ESMX_CLOCK_ACOUSTIC
#include "ClockAcoustic.h"

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
namespace {

using Config = std::unique_ptr<ps_config_t, decltype(&ps_config_free)>;
using Decoder = std::unique_ptr<ps_decoder_t, decltype(&ps_free)>;

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

    const std::string dictionary = std::string(root) + "/clock.dict";
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

bool recognizesAcousticClock(std::span<const int16_t> pcm) {
    if (pcm.empty() || pcm.size() > Kws::MaxSamples) {
        return false;
    }
#ifdef ESP_PLATFORM
    auto* decoder = readyDecoder.load(std::memory_order_acquire);
#else
    auto* decoder = decoderInstance().get();
#endif
    if (!decoder || ps_start_utt(decoder) < 0) {
        return false;
    }
    const auto processed = ps_process_raw(decoder, pcm.data(), pcm.size(), 0, 1);
    const auto ended = ps_end_utt(decoder);
    if (processed < 0 || ended < 0) {
        return false;
    }
    const auto* hypothesis = ps_get_hyp(decoder, nullptr);
    return hypothesis && std::string_view(hypothesis) == "qué hora es";
}

}
#endif
