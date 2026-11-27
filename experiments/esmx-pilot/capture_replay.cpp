#include "Audio/EsMxKws/PilotCapture.h"
#include "esp_err.h"
#include "dispatch_probe.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
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
constexpr TickType_t ListenDeadlineMilliseconds = 8000;
constexpr size_t EndpointMilliseconds = PilotCapture::EndpointSilenceSamples * 1000 / bb::asr::kSampleRate;

std::string_view rejectionName(PilotCapture::Rejection cause){
    using R = PilotCapture::Rejection;
    switch(cause){
        case R::None: return {};
        case R::RawOverflow: return "raw_overflow";
        case R::ModelSpan: return "model_span";
        case R::Empty: return "empty";
        case R::InvalidFrame: return "invalid_frame";
        case R::Gap: return "gap";
        case R::ListenDeadline: return "listen_deadline";
        case R::InferenceDeadline: return "inference_deadline";
    }
    return {};
}

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
    std::string_view reason;
    size_t voicedFrames = 0;
    size_t preRoll = 0;
    size_t inputSent = 0;
};

std::optional<Intent> waitForInference(PilotCapture& capture) {
    // Exercise the completed response with the simulated clock frozen. A
    // timed-out host job is a failed experiment, never an acoustic abstention.
    const auto waitUntil = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < waitUntil) {
        if (const auto result = capture.accept(nullptr)) {
            return result;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    std::cerr << "capture replay: host inference timed out\n" << std::flush;
    // PilotCapture's destructor joins the task. This standalone host process
    // stops immediately if that task failed to finish within the observation.
    std::_Exit(EXIT_FAILURE);
}

ReplayResult replay(std::span<int16_t> samples, bool interrupt, size_t holdMilliseconds) {
    PilotCapture capture;
    ReplayResult result;
    if (!capture.start()) {
        return result;
    }

    // Analyze the input once so the reported metrics are independent of when
    // the worker thread makes its result available.
    std::array<bool, (PilotCapture::RawCapacitySamples + FrameSamples - 1) / FrameSamples> active{};
    size_t firstVoice = samples.size();
    for (size_t start = 0; start < samples.size(); start += FrameSamples) {
        const auto chunk = samples.subspan(start, std::min(FrameSamples, samples.size() - start));
        if (voiced(chunk)) {
            active[start / FrameSamples] = true;
            ++result.voicedFrames;
            firstVoice = std::min(firstVoice, start);
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

    const unsigned initialNotifications = CaptureStub::notifications.load();
    auto feed = [&](std::span<int16_t> chunk, bool speech, std::span<int16_t> prefix = {}) {
        const auto input = frame(chunk, speech, prefix);
        result.intent = capture.accept(&input);
        if (CaptureStub::notifications.load() != initialNotifications) {
            result.intent = waitForInference(capture);
            result.reason = "classified";
            return true;
        }
        result.reason = rejectionName(capture.rejection());
        return result.intent.has_value();
    };

    // Cut the affirmative control after only half its voiced frames, to
    // simulate a fetch gap in a real, incomplete spoken prefix.
    const auto halfVoice = result.voicedFrames / 2;
    size_t voiceSent = 0;
    for (size_t start = 0; start < samples.size(); start += FrameSamples) {
        auto chunk = samples.subspan(start, std::min(FrameSamples, samples.size() - start));
        const bool measuredSpeech = active[start / FrameSamples];
        const auto speech = speechWithHold(measuredSpeech, chunk.size());
        std::span<int16_t> prefix;
        if (start == firstVoice) {
            result.preRoll = std::min(start, PreRollSamples);
            prefix = samples.subspan(start - result.preRoll, result.preRoll);
        }
        CaptureStub::ticks = static_cast<TickType_t>(start / FrameSamples * FrameMilliseconds);
        result.inputSent = start + chunk.size();
        if (feed(chunk, speech, prefix)) {
            return result;
        }
        if (interrupt && measuredSpeech && ++voiceSent == halfVoice) {
            result.intent = capture.accept(nullptr);
            result.reason = rejectionName(capture.rejection());
            return result;
        }
    }

    std::array<int16_t, FrameSamples> silence{};
    if (result.voicedFrames == 0) {
        const auto tail = frame(silence, false);
        CaptureStub::ticks = ListenDeadlineMilliseconds;
        result.intent = capture.accept(&tail);
        result.reason = rejectionName(capture.rejection());
        return result;
    }
    for (size_t elapsed = 0; elapsed < holdMilliseconds + EndpointMilliseconds + 80; elapsed += FrameMilliseconds) {
        CaptureStub::ticks += FrameMilliseconds;
        if (feed(silence, speechWithHold(false, silence.size()))) {
            break;
        }
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

    static std::array<int16_t, PilotCapture::RawCapacitySamples + 1> pcm;
    std::cin.read(reinterpret_cast<char*>(pcm.data()), sizeof(pcm));
    const auto bytes = static_cast<size_t>(std::cin.gcount());
    if (bytes == 0 || bytes % sizeof(int16_t) != 0 ||
        bytes > PilotCapture::RawCapacitySamples * sizeof(int16_t) || std::cin.bad()) {
        return 2;
    }

    const auto inputSamples = bytes / sizeof(int16_t);
    const auto result = replay(std::span(pcm.data(), inputSamples), interrupt,
                               nominalHold ? NominalAfeHoldMilliseconds : 0);
    if (!result.intent || result.reason.empty() || (interrupt && result.reason != "gap")) {
        std::cerr << "capture replay: no attributable terminal result\n";
        return 1;
    }
    std::cout << "{\"observed\":" << static_cast<int>(*result.intent)
              << ",\"route\":\"" << probeDispatch(*result.intent) << "\""
              << ",\"reason\":\"" << result.reason << "\""
              << ",\"voiced_frames\":" << result.voicedFrames
              << ",\"preroll_samples\":" << result.preRoll
              << ",\"input_samples\":" << inputSamples
              << ",\"input_samples_sent\":" << result.inputSent << "}\n";
    return std::cout ? 0 : 1;
}
