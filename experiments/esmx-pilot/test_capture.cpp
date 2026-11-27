#include "Audio/EsMxKws/PilotCapture.h"
#include "Audio/EsMxPilot/Recognition.h"
#include "esp_err.h"
#include <array>
#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

using EsMx::Kws::Intent;
using EsMx::Pilot::PilotCapture;
using Rejection = PilotCapture::Rejection;

static_assert(PilotCapture::EndpointSilenceSamples == 19200);
static_assert(PilotCapture::RawCapacitySamples == 100000);

namespace {
std::vector<int16_t> observed;
std::atomic<unsigned> invocations{0};
std::atomic<bool> hold{false};

afe_fetch_result_t frame(std::span<int16_t> samples, vad_state_t vad, std::span<int16_t> prefix = {}) {
    return {samples.data(), static_cast<int>(samples.size_bytes()), prefix.data(),
            static_cast<int>(prefix.size_bytes()), vad, ESP_OK};
}

std::optional<Intent> awaitIntent(PilotCapture& capture) {
    for (unsigned iteration = 0; iteration < 1000; ++iteration) {
        if (const auto result = capture.accept(nullptr)) {
            return result;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    return std::nullopt;
}
}

Intent EsMx::Pilot::recognize(std::span<const int16_t> samples) {
    observed.assign(samples.begin(), samples.end());
    ++invocations;
    while (hold.load()) {
        std::this_thread::yield();
    }
    return Intent::Dice6;
}

int main() {
    PilotCapture capture;
    int16_t first[]{1200, 1300};
    std::array<int16_t, bb::asr::kWin + 33> prefix{};
    prefix.back() = -1;
    int16_t repeated[]{2010, -2020};
    int16_t next[]{1400, 1500};
    int16_t quietSuffix[]{0, 1, 0};
    std::vector<int16_t> pause(1100 * bb::asr::kSampleRate / 1000);
    pause[pause.size() / 2] = -2;
    std::array<int16_t, PilotCapture::EndpointSilenceSamples> silence{};
    auto start = frame(first, VAD_SPEECH, prefix);
    auto interior = frame(pause, VAD_SILENCE);
    auto speech = frame(next, VAD_SPEECH, repeated);
    auto quiet = frame(quietSuffix, VAD_SILENCE);
    auto endpoint = frame(silence, VAD_SILENCE);

    // Keep the weak edges, precisely one copy of pre-roll, and all samples
    // in the 1.1 s pause. An endpoint is confirmed only after 1.2 s.
    assert(capture.start());
    assert(!capture.accept(&interior));
    assert(!capture.accept(&start));
    assert(!capture.accept(&interior));
    assert(invocations.load() == 0);
    assert(!capture.accept(&speech));
    assert(!capture.accept(&quiet));
    auto almost = frame(std::span(silence).first(silence.size() - std::size(quietSuffix) - 1), VAD_SILENCE);
    assert(!capture.accept(&almost));
    assert(invocations.load() == 0);
    auto crossing = frame(std::span(silence).first(1), VAD_SILENCE);
    assert(!capture.accept(&crossing));
    assert(awaitIntent(capture) == Intent::Dice6);
    assert(capture.rejection() == Rejection::None);
    std::vector<int16_t> expected(bb::asr::kWin, 0);
    expected.push_back(-1);
    expected.insert(expected.end(), std::begin(first), std::end(first));
    expected.insert(expected.end(), pause.begin(), pause.end());
    expected.insert(expected.end(), std::begin(next), std::end(next));
    expected.insert(expected.end(), {0, 1});
    expected.resize(expected.size() + bb::asr::kWin, 0);
    assert(observed == expected);
    assert(!capture.accept(&endpoint));
    capture.stop();

    assert(capture.start());
    assert(!capture.accept(&start));
    auto malformed = speech;
    malformed.data_size = 3;
    assert(capture.accept(&malformed) == Intent::Unknown);
    assert(capture.rejection() == Rejection::InvalidFrame);
    assert(!capture.accept(&endpoint));
    capture.stop();

    // A final silence frame that would prove the endpoint still has to fit.
    // Overflow rejects the complete turn without invoking recognition.
    assert(capture.start());
    std::vector<int16_t> tooMuchRaw(PilotCapture::RawCapacitySamples - silence.size() + 1);
    tooMuchRaw.front() = 2000;
    auto overRaw = frame(tooMuchRaw, VAD_SPEECH);
    assert(!capture.accept(&overRaw));
    assert(capture.accept(&endpoint) == Intent::Unknown);
    assert(capture.rejection() == Rejection::RawOverflow);
    capture.stop();

    // A quiet suffix still counts toward model PCM, even when removing it
    // would make an otherwise long turn fit.
    assert(capture.start());
    std::vector<int16_t> tooMuchModel(EsMx::Kws::MaxSamples + 1);
    tooMuchModel.front() = 2000;
    tooMuchModel.back() = -1;
    auto overModel = frame(tooMuchModel, VAD_SPEECH);
    assert(!capture.accept(&overModel));
    assert(capture.accept(&endpoint) == Intent::Unknown);
    assert(capture.rejection() == Rejection::ModelSpan);
    assert(invocations.load() == 1);
    capture.stop();

    // The whole 4 s model window fits after a 1 s zero AFE hold and the
    // 1.2 s endpoint confirmation. No nonzero sample has to be removed.
    assert(capture.start());
    std::vector<int16_t> fullSpeech(EsMx::Kws::MaxSamples, 1);
    auto full = frame(fullSpeech, VAD_SPEECH);
    auto afeHold = frame(std::span(silence).first(bb::asr::kSampleRate), VAD_SPEECH);
    assert(!capture.accept(&full));
    assert(!capture.accept(&afeHold));
    assert(!capture.accept(&endpoint));
    assert(awaitIntent(capture) == Intent::Dice6);
    assert(capture.rejection() == Rejection::None);
    assert(observed == fullSpeech);
    capture.stop();

    assert(capture.start());
    assert(!capture.accept(&afeHold));
    assert(capture.accept(&endpoint) == Intent::Unknown);
    assert(capture.rejection() == Rejection::Empty);
    capture.stop();

    assert(capture.start());
    assert(capture.rejection() == Rejection::None);
    assert(!capture.accept(&start));
    assert(capture.accept(nullptr) == Intent::Unknown);
    assert(capture.rejection() == Rejection::Gap);
    capture.stop();

    assert(capture.start());
    CaptureStub::ticks = 8000;
    assert(capture.accept(nullptr) == Intent::Unknown);
    assert(capture.rejection() == Rejection::ListenDeadline);
    assert(!capture.accept(&start));
    capture.stop();

    assert(capture.start());
    capture.stop();
    assert(invocations.load() == 2);

    // A late classification can finish, but its result cannot be dispatched or reused.
    hold = true;
    assert(capture.start());
    assert(!capture.accept(&start));
    assert(!capture.accept(&endpoint));
    for (unsigned iteration = 0; invocations.load() == 2 && iteration < 1000; ++iteration) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    assert(invocations.load() == 3);
    CaptureStub::ticks = 11000;
    assert(capture.accept(nullptr) == Intent::Unknown);
    assert(capture.rejection() == Rejection::InferenceDeadline);
    assert(!capture.accept(&speech));
    auto cancel = std::async(std::launch::async, [&]{ capture.stop(); });
    assert(cancel.wait_for(std::chrono::milliseconds(10)) == std::future_status::timeout);
    hold = false;
    cancel.get();

    // An unsuccessful start frees its resources and does not poison the next turn.
    CaptureStub::refuseTask = true;
    assert(!capture.start());
    CaptureStub::refuseTask = false;
    assert(capture.start());
    capture.stop();
    assert(invocations.load() == 3);

    std::cout << "PILOT_CAPTURE 11/11\n";
}
