#pragma once

#include "FlowStubs.h"

class ServiceLocator {
public:
    inline static std::unique_ptr<SpeechGen> SpeechGenInstance;
    inline static std::unique_ptr<SpeechAudioGen> SpeechAudioGenInstance;
    inline static std::unique_ptr<Settings> SettingsInstance;
};
