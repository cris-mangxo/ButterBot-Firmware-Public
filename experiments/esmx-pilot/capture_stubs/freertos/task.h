#pragma once

#include "FreeRTOS.h"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace CaptureStub {
struct Task {
    std::mutex mutex;
    std::condition_variable ready;
    bool notified = false;
    std::jthread thread;
};

inline thread_local Task* currentTask;
inline std::vector<std::unique_ptr<Task>> tasks;
inline std::atomic<TickType_t> ticks{0};
inline bool refuseTask = false;
}

using TaskHandle_t = CaptureStub::Task*;
inline TickType_t xTaskGetTickCount() { return CaptureStub::ticks.load(); }

inline int xTaskCreate(void (*entry)(void*), const char*, uint32_t, void* arg, int, TaskHandle_t* result) {
    if(CaptureStub::refuseTask) {
        return 0;
    }

    CaptureStub::tasks.push_back(std::make_unique<CaptureStub::Task>());
    auto* task = CaptureStub::tasks.back().get();
    *result = task;
    task->thread = std::jthread([task, entry, arg]{
        CaptureStub::currentTask = task;
        entry(arg);
    });
    return pdPASS;
}

inline void xTaskNotifyGive(TaskHandle_t task) {
    std::lock_guard lock(task->mutex);
    task->notified = true;
    task->ready.notify_one();
}

inline void ulTaskNotifyTake(int, TickType_t) {
    auto* task = CaptureStub::currentTask;
    std::unique_lock lock(task->mutex);
    task->ready.wait(lock, [&]{ return task->notified; });
    task->notified = false;
}

// The test joins the retained thread objects at process shutdown.
inline void vTaskDelete(std::nullptr_t) {}
