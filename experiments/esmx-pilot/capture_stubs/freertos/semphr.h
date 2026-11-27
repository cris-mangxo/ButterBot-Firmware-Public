#pragma once

#include "FreeRTOS.h"
#include <condition_variable>
#include <mutex>

struct Semaphore {
    std::mutex mutex;
    std::condition_variable ready;
    bool available = false;
};

using SemaphoreHandle_t = Semaphore*;

inline auto xSemaphoreCreateBinary() { return new Semaphore; }

inline void xSemaphoreGive(SemaphoreHandle_t sem) {
    std::lock_guard lock(sem->mutex);
    sem->available = true;
    sem->ready.notify_one();
}

inline void xSemaphoreTake(SemaphoreHandle_t sem, TickType_t) {
    std::unique_lock lock(sem->mutex);
    sem->ready.wait(lock, [&]{ return sem->available; });
    sem->available = false;
}

inline void vSemaphoreDelete(SemaphoreHandle_t sem) { delete sem; }
