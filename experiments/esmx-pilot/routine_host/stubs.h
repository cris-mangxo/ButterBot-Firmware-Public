#pragma once

#include <BBData.h>
#include <freertos/FreeRTOS.h>
#include <Services/Audio/AudioSource.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <ctime>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace RoutineHost {

using Scenario = BB::Action::Scenario;
using Payload = std::variant<PassTheButterData, JokeData, CurrentTimeData, DiceRollData>;

struct Packet {
    BB::State state;
    Scenario scenario;
    Payload payload;
};

enum class Source { Aac, File, Speech };

struct Playback {
    Source source;
    std::vector<uint8_t> bytes;
    std::string argument;
};

struct Wait {};
using Event = std::variant<Packet, Playback, Wait>;

inline std::vector<Event> events;
inline std::array<uint32_t, 2> randomValues;
inline size_t randomReads;
inline unsigned legacyListening;
inline unsigned legacyModelCalls;

} // namespace RoutineHost

class SpeechGen {
public:
    enum class InputType { Text };
};

class SpeechAudioGen {};
class AACAudioGenerator {};

class SpeechAudioSource {
public:
    SpeechAudioSource(SpeechGen::InputType, std::string text) : text(std::move(text)) {}
    std::string text;
};

class FileAudioSource {
public:
    explicit FileAudioSource(const char* path) : path(path) {}
    std::string path;
};

class Audio {
public:
    void play(AACAudioGenerator* generator, std::unique_ptr<AudioSource> source) {
        assert(generator && source);
        RoutineHost::Playback playback{RoutineHost::Source::Aac, {}, {}};
        std::array<uint8_t, 257> buffer{};
        source->open();
        while(const auto count = source->getData(buffer.data(), buffer.size())){
            assert(count <= buffer.size());
            playback.bytes.insert(playback.bytes.end(), buffer.begin(), buffer.begin() + count);
        }
        assert(!static_cast<bool>(*source));
        source->close();
        RoutineHost::events.emplace_back(std::move(playback));
    }

    void play(AACAudioGenerator* generator, std::unique_ptr<FileAudioSource> source) {
        assert(generator && source);
        RoutineHost::events.emplace_back(RoutineHost::Playback{RoutineHost::Source::File, {}, source->path});
    }

    void play(SpeechAudioGen* generator, std::unique_ptr<SpeechAudioSource> source) {
        assert(generator && source);
        RoutineHost::events.emplace_back(RoutineHost::Playback{RoutineHost::Source::Speech, {}, source->text});
    }

    void waitEnd(uint32_t ticks) const {
        assert(ticks == portMAX_DELAY);
        RoutineHost::events.emplace_back(RoutineHost::Wait{});
    }

    void stop() const {
        assert(false && "unexpected Audio::stop");
    }
};

class Com {
public:
    template<class Data>
    void sendData(BB::State state, BB::Action::Scenario scenario, const Data& data) const {
        RoutineHost::events.emplace_back(RoutineHost::Packet{state, scenario, data});
    }
};

class Time {
public:
    std::tm value{};
    bool configured = false;

    bool isConfigured() const { return configured; }
    std::tm getTime() const { return value; }
};

class AudioFrontend {
public:
    struct Phrase {
        const char* string;
        const char* phonemes;
        const char* fuzzyCore = nullptr;
        float fuzzyThreshold = 0.2f;
        const void* action = nullptr;
    };
    struct PhraseResult {
        bool recognized;
        int index;
        std::string transcript;
        float confidence;
    };
    enum class Mode { Off };

    void setMode(Mode) { ++RoutineHost::legacyModelCalls; }
    void unloadModel() { ++RoutineHost::legacyModelCalls; }
};

namespace RoutineHost {

inline Audio audio;
inline Com com;
inline Time clock;
inline AACAudioGenerator aac;
inline AudioFrontend frontend;
inline bool aacAvailable = true;

} // namespace RoutineHost

class Application {
public:
    template<class Service>
    Service* getService() const {
        using namespace RoutineHost;
        if constexpr(std::is_same_v<Service, Audio>){
            return &audio;
        }else if constexpr(std::is_same_v<Service, Com>){
            return &com;
        }else if constexpr(std::is_same_v<Service, Time>){
            return &RoutineHost::clock;
        }else if constexpr(std::is_same_v<Service, AACAudioGenerator>){
            return aacAvailable ? &aac : nullptr;
        }else{
            static_assert(std::is_same_v<Service, AudioFrontend>, "unconfigured test service");
            return &frontend;
        }
    }
};

class ApplicationStatics {
public:
    static const Application* getApplication() {
        static Application app;
        return &app;
    }
};

struct ServiceLocator {
    static inline auto SpeechAudioGenInstance = std::make_unique<SpeechAudioGen>();
    static inline auto SpeechGenInstance = std::make_unique<SpeechGen>();
    static inline auto ScenarioRoutineServiceInstance = std::make_unique<int>(0);
};

inline uint32_t esp_random() {
    return RoutineHost::randomValues.at(RoutineHost::randomReads++);
}
