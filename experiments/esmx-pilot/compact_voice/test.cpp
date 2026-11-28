#include "Audio/CompactSpeech.h"
#include "Audio/SpeechAudioSource.h"
#include "compact_voice_bank.h"
#include <Phrases.h>
#include <Util/ServiceLocator.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

void testActivity();

namespace {

using Type = SpeechGen::InputType;

std::vector<int16_t> play(std::string text, Type type = Type::Text, std::size_t chunk = 127){
    SpeechAudioSource source(type, std::move(text));
    source.open();
    std::vector<int16_t> result;
    std::array<int16_t, 512> buffer{};
    for(;;){
        const auto size = source.getData(reinterpret_cast<uint8_t*>(buffer.data()),
                                         chunk * sizeof(int16_t));
        assert(size <= chunk * sizeof(int16_t) && size % sizeof(int16_t) == 0);
        if(!size){
            break;
        }
        result.insert(result.end(), buffer.begin(), buffer.begin() + size / sizeof(int16_t));
    }
    assert(!source);
    source.close();
    return result;
}

void check(Phrase phrase){
    for(int16_t id = 0; ; ++id){
        const auto stockText = Phrases::map(phrase, id);
        if(stockText.empty()){
            assert(id > 0);
            break;
        }
        const auto audio = play(stockText);
        assert(std::any_of(audio.begin(), audio.end(), [](auto sample){ return sample != 0; }));
    }
}

}

int main(){
    ServiceLocator::SpeechGenInstance = std::make_unique<SpeechGen>();
    for(const auto& fixed : EsMx::CompactSpeech::Bank::Fixed){
        assert(play(std::string(fixed.factoryText)).size() == fixed.clip.size());
    }
    // Use the OEM phrase catalogue and the actual SpeechAudioSource. No AAC service exists here.
    for(const auto phrase : {Phrase::Startup, Phrase::CameraFailure, Phrase::MotorBoardFailure,
                             Phrase::ShuttingDown, Phrase::BatteryCritical, Phrase::TurningOff,
                             Phrase::SettingsAskActivity, Phrase::SettingsActivityOff,
                             Phrase::SettingsActivityChill, Phrase::SettingsActivityActive,
                             Phrase::SettingsActivityHyper, Phrase::EightBall}){
        check(phrase);
    }

    const auto oneD6 = play("1d6 roll");
    const auto maxD100 = play("100d100 roll");
    assert(!oneD6.empty() && !maxD100.empty() && oneD6 != maxD100);
    assert(play("100d100 roll", Type::Text, 511) == maxD100);
    assert(!play("10000").empty() && play("10000") != play("1000"));
    assert(!play("0").empty() && play("0") != play("100"));
    EsMx::CompactSpeech::Stream numeric;
    for(unsigned value = 0; value <= 10'000; ++value){
        assert(numeric.start(std::to_string(value)));
        int16_t leading = 0;
        bool last = false;
        assert(numeric.generate(std::span{&leading, 1}, last) == 1 && !last);
    }
    assert(play("10001").empty() && play("0d6 roll").empty() && play("1d6 roll!").empty());
    assert(play("Mensaje nuevo de un teléfono.").empty());
    assert(play("SgKS", Type::Phonemes).empty());
    assert(play("<speak>Hola</speak>", Type::SSML).empty());

    // The caller may stop one clip while it is playing and queue another source.
    SpeechAudioSource first(Type::Text, "100d100 roll");
    first.open();
    int16_t sample = 0;
    assert(first.getData(reinterpret_cast<uint8_t*>(&sample), sizeof(sample)) == sizeof(sample));
    first.close();
    assert(first.getData(reinterpret_cast<uint8_t*>(&sample), sizeof(sample)) == 0);
    assert(play("1d6 roll") == oneD6);
    testActivity();
    std::cout << "COMPACT_VOICE_HOST pass fixed, early boot, activities, Eight Ball, numbers, cancel/restart, unsupported\n";
}
