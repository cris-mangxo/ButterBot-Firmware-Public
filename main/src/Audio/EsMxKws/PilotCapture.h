#pragma once

#ifdef BB_ESMX_PILOT
#include "Compact.h"
#include "esp_afe_sr_iface.h"
#include <atomic>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/task.h>
#include <optional>

namespace EsMx::Pilot {

// The AFE task assembles one turn. Only the inference task reads the completed buffer.
class PilotCapture {
public:
	// 4 s of model PCM, 1 s of nominal AFE VAD hold, 1.2 s of endpoint
	// confirmation, and 50 ms for the final frame crossing the endpoint.
	static constexpr size_t EndpointSilenceSamples = 1200 * bb::asr::kSampleRate / 1000;
	static constexpr size_t RawCapacitySamples =
		Kws::MaxSamples + bb::asr::kSampleRate + EndpointSilenceSamples + bb::asr::kSampleRate / 20;

	enum class Rejection { None, RawOverflow, ModelSpan, Empty, InvalidFrame, Gap, ListenDeadline, InferenceDeadline };

	~PilotCapture();

	// Call from the state task before starting or after parking the AFE tasks.
	bool start();
	void stop();

	// Call from the AFE task only. nullopt means the turn is still in progress or reported.
	std::optional<Kws::Intent> accept(const afe_fetch_result_t* frame);

	// Read from the AFE task after accept returns a result. None includes a
	// completed inference that abstained. The next start resets this value.
	Rejection rejection() const { return lastRejection; }

private:
	enum class Phase { Waiting, Collecting, Classifying, Reported };

	bool append(std::span<const int16_t> samples);
	std::optional<Kws::Intent> submit(TickType_t now);
	void classify();
	std::optional<Kws::Intent> reject(Rejection reason);

	int16_t* pcm = nullptr;
	TaskHandle_t worker = nullptr;
	SemaphoreHandle_t workerDone = nullptr;
	std::atomic<bool> submitted{false};
	std::atomic<bool> cancelled{false};
	std::atomic<bool> completed{false};
	Kws::Intent result = Kws::Intent::Unknown;

	Phase phase = Phase::Reported;
	Rejection lastRejection = Rejection::None;
	TickType_t started = 0;
	TickType_t submittedAt = 0;
	size_t captured = 0;
	size_t silence = 0;
	size_t firstNonzero = RawCapacitySamples;
	size_t lastNonzeroEnd = 0;
	size_t modelBegin = 0;
	size_t modelLength = 0;
};

}
#endif
