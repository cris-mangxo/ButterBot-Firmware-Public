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

namespace {
std::vector<int16_t> observed;
std::atomic<unsigned> invocations{0};
std::atomic<bool> hold{false};

afe_fetch_result_t frame(std::span<int16_t> samples, vad_state_t vad, std::span<int16_t> prefix = {}) {
    return {samples.data(), static_cast<int>(samples.size_bytes()), prefix.data(),
            static_cast<int>(prefix.size_bytes()), vad, ESP_OK};
}
}

Intent EsMx::Pilot::recognize(std::span<const int16_t> samples) {
    observed.assign(samples.begin(), samples.end());
    ++invocations;
    while(hold.load()) {
        std::this_thread::yield();
    }
    return Intent::Dice6;
}

int main() {
    PilotCapture capture;
    int16_t first[]{3, 4};
    int16_t cache[]{1, 2};
    int16_t repeated[]{2, 3, 4, 5};
    int16_t pause[]{5, 6};
    int16_t next[]{7, 8};
    auto start = frame(first, VAD_SPEECH, cache);
    auto gap = frame(pause, VAD_SILENCE);
    auto speech = frame(next, VAD_SPEECH, repeated);
    std::array<int16_t, 5120> silence{};
    auto end = frame(silence, VAD_SILENCE);

    // The decoder receives one complete turn with its prefix, including the interior pause.
    assert(capture.start());
    assert(!capture.accept(&gap));
    assert(!capture.accept(&start));
    assert(!capture.accept(&gap));
    assert(!capture.accept(&speech));
    auto beforeEndpoint = frame(std::span(silence).first(5119), VAD_SILENCE);
    assert(!capture.accept(&beforeEndpoint));
    assert(invocations.load() == 0);
    assert(!capture.accept(&gap));
    std::optional<Intent> answer;
    for(unsigned iteration = 0; !answer && iteration < 1000; ++iteration) {
        answer = capture.accept(nullptr);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    assert(answer == Intent::Dice6);
    assert((observed == std::vector<int16_t>{1, 2, 3, 4, 5, 6, 7, 8}));
    assert(!capture.accept(&end));
    capture.stop();

    assert(capture.start());
    assert(!capture.accept(&start));
    auto malformed = speech;
    malformed.data_size = 3;
    assert(capture.accept(&malformed) == Intent::Unknown);
    assert(!capture.accept(&end));
    capture.stop();

    assert(capture.start());
    std::vector<int16_t> maxLength(EsMx::Kws::MaxSamples);
    auto full = frame(maxLength, VAD_SPEECH);
    assert(!capture.accept(&full));
    assert(capture.accept(&speech) == Intent::Unknown);
    assert(!capture.accept(&end));
    capture.stop();

    assert(capture.start());
    assert(!capture.accept(&start));
    assert(capture.accept(nullptr) == Intent::Unknown);
    capture.stop();

    assert(capture.start());
    CaptureStub::ticks = 8000;
    assert(capture.accept(nullptr) == Intent::Unknown);
    assert(!capture.accept(&start));
    capture.stop();

    assert(capture.start());
    capture.stop();
    assert(invocations.load() == 1);

    // A late classification can finish, but its result cannot be dispatched or reused.
    hold = true;
    assert(capture.start());
    assert(!capture.accept(&start));
    assert(!capture.accept(&end));
    for(unsigned iteration = 0; invocations.load() == 1 && iteration < 1000; ++iteration) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    assert(invocations.load() == 2);
    CaptureStub::ticks = 11000;
    assert(capture.accept(nullptr) == Intent::Unknown);
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
    assert(invocations.load() == 2);

    std::cout << "PILOT_CAPTURE 8/8\n";
}
