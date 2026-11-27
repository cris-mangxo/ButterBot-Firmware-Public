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
	~PilotCapture();

	// Call from the state task before starting or after parking the AFE tasks.
	bool start();
	void stop();

	// Call from the AFE task only. nullopt means the turn is still in progress or reported.
	std::optional<Kws::Intent> accept(const afe_fetch_result_t* frame);

private:
	enum class Phase { Waiting, Collecting, Classifying, Reported };

	bool append(const int16_t* data, int bytes);
	void classify();
	std::optional<Kws::Intent> reject();

	int16_t* pcm = nullptr;
	TaskHandle_t worker = nullptr;
	SemaphoreHandle_t workerDone = nullptr;
	std::atomic<bool> submitted{false};
	std::atomic<bool> cancelled{false};
	std::atomic<bool> completed{false};
	Kws::Intent result = Kws::Intent::Unknown;

	Phase phase = Phase::Reported;
	TickType_t started = 0;
	TickType_t submittedAt = 0;
	size_t captured = 0;
	size_t silence = 0;
	size_t speechEnd = 0;
};

}
#endif
