#include "PilotCapture.h"

#ifdef BB_ESMX_PILOT
#include "Audio/EsMxPilot/Recognition.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include <cstring>

namespace EsMx::Pilot {
namespace {

constexpr TickType_t ListenDeadline = pdMS_TO_TICKS(8000);
constexpr TickType_t InferenceDeadline = pdMS_TO_TICKS(3000);
constexpr size_t EndSilenceSamples = bb::asr::kSampleRate * 320 / 1000;
constexpr uint32_t WorkerStackBytes = 8 * 1024;

bool validSamples(const int16_t* data, int size){
	return data && size > 0 && size % static_cast<int>(sizeof(int16_t)) == 0;
}

}

PilotCapture::~PilotCapture(){
	stop();
}

bool PilotCapture::start(){
	if(worker){
		return false;
	}

	pcm = static_cast<int16_t*>(heap_caps_malloc(Kws::MaxSamples * sizeof(*pcm),
	                                              MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
	workerDone = xSemaphoreCreateBinary();
	if(!pcm || !workerDone){
		stop();
		return false;
	}

	cancelled.store(false);
	completed.store(false);
	submitted.store(false);
	captured = 0;
	silence = 0;
	speechEnd = 0;
	phase = Phase::Waiting;
	started = xTaskGetTickCount();

	auto run = [](void* arg){
		auto& capture = *static_cast<PilotCapture*>(arg);
		capture.classify();
		xSemaphoreGive(capture.workerDone);
		vTaskDelete(nullptr);
	};

	if(xTaskCreate(run, "esmx_kws", WorkerStackBytes, this, 3, &worker) != pdPASS){
		stop();
		return false;
	}
	return true;
}

void PilotCapture::stop(){
	if(worker){
		cancelled.store(true);
		// If scheduled for inference it will already wake, or may have already finished.
		if(!submitted.load()){
			xTaskNotifyGive(worker);
		}
		xSemaphoreTake(workerDone, portMAX_DELAY);
		worker = nullptr;
	}
	if(workerDone){
		vSemaphoreDelete(workerDone);
		workerDone = nullptr;
	}
	heap_caps_free(pcm);
	pcm = nullptr;
	phase = Phase::Reported;
}

bool PilotCapture::append(const int16_t* data, int bytes){
	if(!validSamples(data, bytes)){
		return false;
	}
	const auto samples = static_cast<size_t>(bytes) / sizeof(*pcm);
	if(samples > Kws::MaxSamples - captured){
		return false;
	}
	std::memcpy(pcm + captured, data, static_cast<size_t>(bytes));
	captured += samples;
	return true;
}

void PilotCapture::classify(){
	ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
	if(submitted.load() && !cancelled.load()){
		result = recognize({pcm, speechEnd});
		completed.store(true);
	}
}

std::optional<Kws::Intent> PilotCapture::reject(){
	phase = Phase::Reported;
	return Kws::Intent::Unknown;
}

std::optional<Kws::Intent> PilotCapture::accept(const afe_fetch_result_t* frame){
	if(phase == Phase::Reported){
		return std::nullopt;
	}

	const TickType_t now = xTaskGetTickCount();
	if(phase == Phase::Classifying){
		if(now - submittedAt >= InferenceDeadline){
			return reject();
		}
		if(!completed.load()){
			return std::nullopt;
		}
		phase = Phase::Reported;
		return result;
	}

	if(now - started >= ListenDeadline){
		return reject();
	}

	if(!frame || frame->ret_value != ESP_OK || !validSamples(frame->data, frame->data_size)){
		// A fetch gap inside speech invalidates the turn, including a valid-sounding prefix.
		return phase == Phase::Collecting ? reject() : std::nullopt;
	}

	const bool speaking = (frame->vad_state == VAD_SPEECH);
	if(phase == Phase::Waiting){
		if(!speaking){
			return std::nullopt;
		}
		phase = Phase::Collecting;
		// Espressif supplies the samples before the current processed frame. On later speech
		// edges we have already retained the intervening frames, so their cache isn't appended.
		if(frame->vad_cache_size < 0 ||
		   (frame->vad_cache_size > 0 && !append(frame->vad_cache, frame->vad_cache_size))){
			return reject();
		}
	}

	if(!speaking){
		silence += static_cast<size_t>(frame->data_size) / sizeof(*pcm);
		if(silence >= EndSilenceSamples){
			phase = Phase::Classifying;
			submittedAt = now;
			submitted.store(true);
			xTaskNotifyGive(worker);
			return std::nullopt;
		}
	}

	// Keep short interior pauses. Any overflow invalidates the entire turn.
	if(!append(frame->data, frame->data_size)){
		return reject();
	}
	if(speaking){
		silence = 0;
		speechEnd = captured;
	}
	return std::nullopt;
}

}
#endif
