#include "Audio/EsMxPilot/ClockAcoustic.h"
#include "Audio/EsMxPilot/ClockModelVfs.h"
#include "Audio/EsMxPilot/Recognition.h"
#ifdef BB_ESMX_BUTTER_CEPSTRA
#include "Audio/EsMxPilot/ButterAcoustic.h"
#endif
#ifdef BB_ESMX_EXPECT_OPPOSING_POOL
#include "Audio/EsMxPilot/AcousticSearch.h"
extern "C" int bb_fsg_opposing_pool_active(const ps_decoder_t*);
#endif
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
#ifdef BB_ESMX_EXPECT_OPPOSING_POOL
    const bool poolActive = bb_fsg_opposing_pool_active(sharedAcousticDecoder());
    std::printf("BB_BUTTER_POOL active=%u\n", static_cast<unsigned>(poolActive));
    if (!poolActive) {
        std::puts("BB_CLOCK_FAIL");
        return;
    }
#endif
    bool ok = true;
#ifdef BB_ESMX_D6_SCOPE_BENCH
    const bool d6Installed = isDiceScopeInstalledInBench();
#ifdef BB_ESMX_D6_BENCH_OMIT_KWS
    constexpr bool ExpectD6 = false;
#else
    constexpr bool ExpectD6 = true;
#endif
    std::printf("BB_D6_INSTALLED %u\n", d6Installed);
    ok &= d6Installed == ExpectD6;
#endif
    for (const auto& item : TargetCases) {
#ifdef BB_ESMX_D6_SCOPE_BENCH
        std::printf("BB_D6_CASE_BEGIN %s\n", item.name);
#endif
#ifdef BB_ESMX_BUTTER_CEPSTRA
        resetButterCepstraStatus();
#endif
        const auto elapsedStart = esp_timer_get_time();
        const auto recognized = static_cast<unsigned>(recognize(item.samples));
        const auto elapsed = esp_timer_get_time() - elapsedStart;
#ifdef BB_ESMX_BUTTER_CEPSTRA
        const auto cepstra = butterCepstraStatus();
        const bool usedCepstra = cepstra == ButterCepstraStatus::Used;
        const bool requiredCepstra = item.intent == static_cast<unsigned>(EsMx::Kws::Intent::Butter);
        ok &= cepstra != ButterCepstraStatus::Failed && (!requiredCepstra || usedCepstra);
        std::printf("BB_BUTTER_CEPSTRA used=%u\n", static_cast<unsigned>(usedCepstra));
#endif
        const auto upstream = recognizeDetailed(item.samples).disposition;
        const bool expected = recognized == item.intent && upstream == item.upstream;
        ok &= expected;
#if defined(BB_ESMX_BUTTER_ACOUSTIC) || defined(BB_ESMX_D6_SCOPE_BENCH)
        constexpr int64_t InferenceWindowUs = 3'000'000;
        const bool withinBudget = elapsed <= InferenceWindowUs;
        ok &= withinBudget;
#ifdef BB_ESMX_BUTTER_ACOUSTIC
        std::printf("BB_BUTTER_BUDGET %s %lld %u\n", item.name,
                    static_cast<long long>(elapsed), withinBudget);
#endif
#ifdef BB_ESMX_D6_SCOPE_BENCH
        std::printf("BB_PILOT_BUDGET %s %lld %u\n", item.name,
                    static_cast<long long>(elapsed), withinBudget);
        std::printf("BB_D6_CASE_STACK %s %u\n", item.name,
                    static_cast<unsigned>(uxTaskGetStackHighWaterMark(nullptr)));
#endif
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
