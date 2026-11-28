#pragma once

#include "Audio/SpeechGen.h"
#include <BBData.h>
#include <Services/Audio/AudioSource.h>
#include <array>
#include <cassert>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace ActivityHost {
enum class Step { Set, Store, Sync, Change, Play, Wait, Listen };
inline std::vector<Step> steps;
inline SettingsChangeData change{};
inline SettingsSyncData sync{};
inline std::size_t pcmSamples = 0;
}

class AudioFrontend {
public:
    enum class Mode { Off };

    struct Phrase {
        const char* text;
        const char* phonemes;
        const char* fuzzy;
    };

    struct PhraseResult {
        bool recognized;
        int index;
        std::string transcript;
        float confidence;
    };

    void setMode(Mode) {}
    void unloadModel() {}
};

class SpeechAudioGen {};

class Audio {
public:
    void play(SpeechAudioGen*, std::unique_ptr<AudioSource> newSource){
        ActivityHost::steps.push_back(ActivityHost::Step::Play);
        source = std::move(newSource);
        source->open();
    }

    void waitEnd(TickType_t){
        ActivityHost::steps.push_back(ActivityHost::Step::Wait);
        std::array<int16_t, 128> buffer{};
        while(const auto bytes = source->getData(reinterpret_cast<uint8_t*>(buffer.data()),
                                                 buffer.size() * sizeof(int16_t))){
            assert(bytes % 2 == 0);
            ActivityHost::pcmSamples += bytes / sizeof(int16_t);
        }
        source->close();
        source.reset();
    }

    void stop(){
        if(source){
            source->close();
            source.reset();
        }
    }

private:
    std::unique_ptr<AudioSource> source;
};

class Com {
public:
    void sendData(BB::State state, BB::Action::Idle id, SettingsSyncData data){
        assert(state == BB::State::Idle && id == BB::Action::Idle::SettingsSync);
        ActivityHost::steps.push_back(ActivityHost::Step::Sync);
        ActivityHost::sync = data;
    }
    void sendData(BB::State state, BB::Action::Scenario id, SettingsChangeData data){
        assert(state == BB::State::Scenario && id == BB::Action::Scenario::SettingsActivity);
        ActivityHost::steps.push_back(ActivityHost::Step::Change);
        ActivityHost::change = data;
    }
};

struct SettingsStruct {
    FrontSensorMode frontSensor = FrontSensorMode::Off;
    BottomSensorMode bottomSensor = BottomSensorMode::Reduced;
    ActivityMode activityMode = ActivityMode::Active;
};

class Settings {
public:
    SettingsStruct get() const { return value; }

    void set(const SettingsStruct& next){
        ActivityHost::steps.push_back(ActivityHost::Step::Set);
        value = next;
    }

    void store() const {
        ActivityHost::steps.push_back(ActivityHost::Step::Store);
    }

private:
    SettingsStruct value;
};

class Application {
public:
    template<class Service>
    Service* getService() const {
        static Service service;
        return &service;
    }
};

class ApplicationStatics {
public:
    static const Application* getApplication(){
        static const Application app;
        return &app;
    }
};
