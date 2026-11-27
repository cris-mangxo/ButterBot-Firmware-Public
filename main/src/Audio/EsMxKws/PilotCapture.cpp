#include "PilotCapture.h"

#ifdef BB_ESMX_PILOT
#include "Audio/EsMxPilot/Recognition.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include <algorithm>
#include <cstring>

namespace EsMx::Pilot {
namespace {

constexpr TickType_t ListenDeadline = pdMS_TO_TICKS(8000);
constexpr TickType_t InferenceDeadline = pdMS_TO_TICKS(3000);
constexpr size_t EdgeContextSamples = bb::asr::kWin;
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

	pcm = static_cast<int16_t*>(heap_caps_malloc(RawCapacitySamples * sizeof(*pcm),
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
	firstNonzero = RawCapacitySamples;
	lastNonzeroEnd = 0;
	modelBegin = 0;
	modelLength = 0;
	lastRejection = Rejection::None;
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

bool PilotCapture::append(std::span<const int16_t> samples){
	if(samples.size() > RawCapacitySamples - captured){
		return false;
	}

	// Only digital zeros can be discarded. Track both edges without scanning
	// the complete PSRAM buffer on the AFE task when the endpoint arrives.
	const auto nonzero = [](int16_t sample){ return sample != 0; };
	if(firstNonzero == RawCapacitySamples){
		const auto first = std::find_if(samples.begin(), samples.end(), nonzero);
		if(first != samples.end()){
			firstNonzero = captured + (first - samples.begin());
		}
	}
	const auto last = std::find_if(samples.rbegin(), samples.rend(), nonzero);
	if(last != samples.rend()){
		lastNonzeroEnd = captured + (last.base() - samples.begin());
	}

	std::memcpy(pcm + captured, samples.data(), samples.size_bytes());
	captured += samples.size();
	return true;
}

std::optional<Kws::Intent> PilotCapture::submit(TickType_t now){
	if(firstNonzero == RawCapacitySamples){
		return reject(Rejection::Empty);
	}

	const size_t nonzeroSpan = lastNonzeroEnd - firstNonzero;
	if(nonzeroSpan > Kws::MaxSamples){
		return reject(Rejection::ModelSpan);
	}

	// Retain an analysis window of zero context on each side where the model
	// budget permits. Interior samples, quiet edges and all pauses stay intact.
	const size_t spare = Kws::MaxSamples - nonzeroSpan;
	const size_t lead = std::min({firstNonzero, EdgeContextSamples, spare / 2});
	const size_t tail = std::min({captured - lastNonzeroEnd, EdgeContextSamples, spare - lead});
	modelBegin = firstNonzero - lead;
	modelLength = nonzeroSpan + lead + tail;

	phase = Phase::Classifying;
	submittedAt = now;
	submitted.store(true);
	xTaskNotifyGive(worker);
	return std::nullopt;
}

void PilotCapture::classify(){
	ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
	if(submitted.load() && !cancelled.load()){
		result = recognize({pcm + modelBegin, modelLength});
		completed.store(true);
	}
}

std::optional<Kws::Intent> PilotCapture::reject(Rejection reason){
	lastRejection = reason;
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
			return reject(Rejection::InferenceDeadline);
		}
		if(!completed.load()){
			return std::nullopt;
		}
		phase = Phase::Reported;
		return result;
	}

	if(now - started >= ListenDeadline){
		return reject(Rejection::ListenDeadline);
	}

	if(!frame || frame->ret_value != ESP_OK){
		// A fetch gap inside the turn invalidates even a valid-sounding prefix.
		return phase == Phase::Collecting ? reject(Rejection::Gap) : std::nullopt;
	}
	if(!validSamples(frame->data, frame->data_size) ||
	   (frame->vad_state != VAD_SPEECH && frame->vad_state != VAD_SILENCE)){
		return phase == Phase::Collecting ? reject(Rejection::InvalidFrame) : std::nullopt;
	}

	const bool speaking = frame->vad_state == VAD_SPEECH;
	if(phase == Phase::Waiting){
		if(!speaking){
			return std::nullopt;
		}
		phase = Phase::Collecting;
		// The preceding AFE cache is appended once. Subsequent speech edges
		// already have their intervening audio in the same buffer.
		const int cacheBytes = frame->vad_cache_size;
		if(cacheBytes < 0 || (cacheBytes > 0 && !validSamples(frame->vad_cache, cacheBytes))){
			return reject(Rejection::InvalidFrame);
		}
		if(cacheBytes > 0 && !append({frame->vad_cache, static_cast<size_t>(cacheBytes) / sizeof(*pcm)})){
			return reject(Rejection::RawOverflow);
		}
	}

	const auto samples = static_cast<size_t>(frame->data_size) / sizeof(*pcm);
	if(!append({frame->data, samples})){
		return reject(Rejection::RawOverflow);
	}
	silence = speaking ? 0 : silence + samples;
	return silence >= EndpointSilenceSamples ? submit(now) : std::nullopt;
}

}
#endif
