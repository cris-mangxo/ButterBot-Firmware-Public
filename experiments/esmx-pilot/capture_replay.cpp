#include "Audio/EsMxKws/PilotCapture.h"
#include "esp_err.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <string_view>
#include <thread>

namespace {
using EsMx::Kws::Intent;
using EsMx::Pilot::PilotCapture;

constexpr size_t FrameMilliseconds = 20;
constexpr size_t FrameSamples = bb::asr::kSampleRate * FrameMilliseconds / 1000;
constexpr size_t PreRollSamples = 12 * FrameSamples;
constexpr int RmsThreshold = 180;
constexpr size_t NominalAfeHoldMilliseconds = 1000;

bool voiced(std::span<const int16_t> samples) {
    uint64_t energy = 0;
    for (const int64_t sample : samples) {
        energy += sample * sample;
    }
    return energy >= samples.size() * RmsThreshold * RmsThreshold;
}

afe_fetch_result_t frame(std::span<int16_t> samples, bool speech, std::span<int16_t> prefix = {}) {
    return {samples.data(), static_cast<int>(samples.size_bytes()), prefix.data(),
            static_cast<int>(prefix.size_bytes()), speech ? VAD_SPEECH : VAD_SILENCE, ESP_OK};
}

struct ReplayResult {
    std::optional<Intent> intent;
    size_t voicedFrames = 0;
    size_t preRoll = 0;
    bool gapInjected = false;
};

ReplayResult replay(std::span<int16_t> samples, bool interrupt, size_t holdMilliseconds) {
    PilotCapture capture;
    ReplayResult result;
    if (!capture.start()) {
        return result;
    }

    // Analyze the input once so the reported metrics are independent of when
    // the worker thread makes the first result available.
    std::array<bool, (EsMx::Kws::MaxSamples + FrameSamples - 1) / FrameSamples> active{};
    size_t firstVoice = samples.size();
    size_t lastVoiceEnd = 0;
    for (size_t start = 0; start < samples.size(); start += FrameSamples) {
        const auto chunk = samples.subspan(start, std::min(FrameSamples, samples.size() - start));
        if (voiced(chunk)) {
            active[start / FrameSamples] = true;
            ++result.voicedFrames;
            firstVoice = std::min(firstVoice, start);
            lastVoiceEnd = start + chunk.size();
        }
    }

    const size_t holdSamples = bb::asr::kSampleRate * holdMilliseconds / 1000;
    auto speechWithHold = [holdSamples, remaining = size_t{0}](bool speech, size_t count) mutable {
        if (speech) {
            remaining = holdSamples;
            return true;
        }
        const bool held = remaining > 0;
        remaining -= std::min(remaining, count);
        return held;
    };

    // Interrupt the positive control after its last voiced frame. Keep the
    // turn open so the injected gap, rather than a silence timer, closes it.
    const size_t inputEnd = interrupt ? lastVoiceEnd : samples.size();
    for (size_t start = 0; start < inputEnd && !result.intent; start += FrameSamples) {
        auto chunk = samples.subspan(start, std::min(FrameSamples, inputEnd - start));
        const auto speech = interrupt || speechWithHold(active[start / FrameSamples], chunk.size());
        std::span<int16_t> prefix;
        if (!interrupt && start == firstVoice) {
            result.preRoll = std::min(start, PreRollSamples);
            prefix = samples.subspan(start - result.preRoll, result.preRoll);
        }
        auto input = frame(chunk, speech, prefix);
        CaptureStub::ticks = static_cast<TickType_t>(start / FrameSamples * FrameMilliseconds);
        result.intent = capture.accept(&input);
    }

    if (interrupt && result.voicedFrames > 0 && !result.intent) {
        result.gapInjected = true;
        result.intent = capture.accept(nullptr);
    } else {
        std::array<int16_t, FrameSamples> silence{};
        auto tail = frame(silence, false);
        // Give the real capture time to close a spoken turn. For a clip that
        // contains only silence, advance the simulated clock to its deadline.
        if (result.voicedFrames == 0) {
            CaptureStub::ticks += 8000;
            result.intent = capture.accept(&tail);
        }
        for (size_t elapsed = 0; elapsed < holdMilliseconds + 400 && !result.intent;
             elapsed += FrameMilliseconds) {
            CaptureStub::ticks += FrameMilliseconds;
            tail.vad_state = speechWithHold(false, silence.size()) ? VAD_SPEECH : VAD_SILENCE;
            result.intent = capture.accept(&tail);
        }
    }

    // A host worker may still be classifying. Allow it to finish while halting
    // the simulated audio clock. This does not measure ESP32 inference time.
    const auto waitUntil = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!result.intent && std::chrono::steady_clock::now() < waitUntil) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        result.intent = capture.accept(nullptr);
    }
    if (result.intent && capture.accept(nullptr)) {
        return {};
    }
    return result;
}
}

int main(int argc, char** argv) {
    const auto option = argc == 2 ? std::string_view(argv[1]) : std::string_view{};
    const bool interrupt = option == "--interrupted";
    const bool nominalHold = option == "--nominal-hangover";
    if (argc != 1 && !interrupt && !nominalHold) {
        return 2;
    }

    static std::array<int16_t, EsMx::Kws::MaxSamples + 1> pcm;
    std::cin.read(reinterpret_cast<char*>(pcm.data()), sizeof(pcm));
    const auto bytes = static_cast<size_t>(std::cin.gcount());
    if (bytes == 0 || bytes % sizeof(int16_t) != 0 ||
        bytes > EsMx::Kws::MaxSamples * sizeof(int16_t) || std::cin.bad()) {
        return 2;
    }

    const auto result = replay(std::span(pcm.data(), bytes / sizeof(int16_t)), interrupt,
                               nominalHold ? NominalAfeHoldMilliseconds : 0);
    if (!result.intent || (interrupt && !result.gapInjected)) {
        return 1;
    }
    std::cout << "{\"observed\":" << static_cast<int>(*result.intent)
              << ",\"voiced_frames\":" << result.voicedFrames
              << ",\"preroll_samples\":" << result.preRoll
              << ",\"interrupted\":" << (result.gapInjected ? "true" : "false") << "}\n";
    return std::cout ? 0 : 1;
}
