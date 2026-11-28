#include "Audio/EsMxPilot/ClockAcoustic.h"
#include "Audio/EsMxPilot/ClockModelVfs.h"
#include "Audio/EsMxPilot/Recognition.h"
#include "ClockTargetCases.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "mbedtls/sha256.h"

#include <array>
#include <cstdio>
#include <cstring>

namespace {
using namespace EsMx::Pilot;

bool verifyModelFile(const char* path, const char* expected) {
    auto* file = std::fopen(path, "rb");
    if (!file) {
        return false;
    }
    mbedtls_sha256_context state;
    mbedtls_sha256_init(&state);
    std::array<unsigned char, 384> block;
    int result = mbedtls_sha256_starts(&state, 0);
    while (result == 0) {
        const auto bytes = std::fread(block.data(), 1, block.size(), file);
        if (!bytes) {
            break;
        }
        result = mbedtls_sha256_update(&state, block.data(), bytes);
    }
    const bool readable = !std::ferror(file);
    std::fclose(file);
    std::array<unsigned char, 32> digest;
    const bool done = result == 0 && mbedtls_sha256_finish(&state, digest.data()) == 0;
    mbedtls_sha256_free(&state);
    if (!done || !readable) {
        return false;
    }
    char hex[65]{};
    for (size_t i = 0; i < digest.size(); ++i) {
        std::snprintf(hex + 2 * i, 3, "%02x", static_cast<unsigned>(digest[i]));
    }
    return std::strcmp(hex, expected) == 0;
}
}

extern "C" void app_main() {
    using EsMx::Pilot::TargetCases;
    if (!mountClockModel() ||
        !verifyModelFile("/bbclock/means", ClockTargetMeansSha) ||
        !verifyModelFile("/bbclock/variances", ClockTargetVariancesSha)) {
        std::puts("BB_CLOCK_MODEL_MISMATCH");
        return;
    }
    std::puts("BB_CLOCK_MODEL_OK");
    const auto internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    const auto psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    std::printf("BB_CLOCK_HEAP_BEFORE %zu %zu\n", internal, psram);
    const auto coldStart = esp_timer_get_time();
    if (!prepareAcousticClock()) {
        std::puts("BB_CLOCK_FAIL");
        return;
    }
    std::printf("BB_CLOCK_WARMUP %lld\n", static_cast<long long>(esp_timer_get_time() - coldStart));
    bool ok = true;
    for (const auto& item : TargetCases) {
        const auto elapsedStart = esp_timer_get_time();
        const auto recognized = static_cast<unsigned>(recognize(item.samples));
        const auto elapsed = esp_timer_get_time() - elapsedStart;
        const auto upstream = recognizeDetailed(item.samples).disposition;
        const bool expected = recognized == item.intent && upstream == item.upstream;
        ok &= expected;
#if defined(BB_ESMX_BUTTER_ACOUSTIC) || defined(BB_ESMX_D6_SCOPE_BENCH)
        constexpr int64_t InferenceWindowUs = 3'000'000;
        const bool withinBudget = elapsed <= InferenceWindowUs;
        ok &= withinBudget;
#ifdef BB_ESMX_D6_SCOPE_BENCH
        constexpr const char* BudgetEvent = "BB_PILOT_BUDGET";
#else
        constexpr const char* BudgetEvent = "BB_BUTTER_BUDGET";
#endif
        std::printf("%s %s %lld %u\n", BudgetEvent, item.name,
                    static_cast<long long>(elapsed), withinBudget);
#endif
        std::printf("BB_CLOCK_CASE %s %u %u %lld %u\n", item.name, recognized,
                    static_cast<unsigned>(upstream), static_cast<long long>(elapsed), expected);
    }
    std::printf("BB_CLOCK_HEAP_AFTER %zu %zu %zu %zu\n",
                heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
                heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
                heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
                heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM));
    std::printf("BB_CLOCK_STACK_FREE %u\n", static_cast<unsigned>(uxTaskGetStackHighWaterMark(nullptr)));
    std::puts(ok ? "BB_CLOCK_PASS" : "BB_CLOCK_FAIL");
}
