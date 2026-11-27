#include "Voice.h"

#ifdef BB_ESMX_PILOT

#include <initializer_list>
#include <iterator>
#include <memory>
#include <span>
#include <utility>

#include <Core/Application.h>
#include <Services/Audio/AACAudioGenerator.h>
#include <Services/Audio/Audio.h>
#include <Statics/ApplicationStatics.h>

#include "Audio/MemoryAudioSource.h"
#include "voice_data.h"

namespace EsMx::Pilot {
namespace {

using Clip = std::span<const uint8_t>;
using namespace VoiceData;

static_assert(std::size(Numbers) == 33);
static_assert(std::size(CompoundDecades) == 3);
static_assert(std::size(DiceFaces) == 6);

bool play(Audio& audio, std::initializer_list<Clip> clips){
    auto* generator = ApplicationStatics::getApplication()->getService<AACAudioGenerator>();
    if(generator == nullptr){
        return false;
    }

    size_t bytes = 0;
    for(const Clip clip : clips){
        bytes += clip.size();
    }
    if(bytes == 0){
        return false;
    }

    // Every clip is mono 16 kHz AAC-LC with ADTS. Consecutive frames share a decoder.
    PSRAMByteBuffer stream;
    stream.reserve(bytes);
    for(const Clip clip : clips){
        if(!clip.empty()){
            stream.insert(stream.end(), clip.begin(), clip.end());
        }
    }
    audio.play(generator, std::make_unique<MemoryAudioSource>(std::move(stream)));
    audio.waitEnd(portMAX_DELAY);
    return true;
}

bool indexed(Audio& audio, std::span<const Clip> clips, int16_t index){
    if(index < 0 || static_cast<size_t>(index) >= clips.size()){
        return false;
    }
    return play(audio, { clips[index] });
}

}

int16_t butterVoiceId(int16_t stockId){
    return stockId < 0 ? stockId : stockId % std::size(Butter);
}

int16_t jokeVoiceId(int16_t stockId){
    return stockId < 0 ? stockId : stockId % std::size(Joke);
}

bool playButter(Audio& audio, int16_t id){
    return indexed(audio, Butter, id);
}

bool playJoke(Audio& audio, int16_t id){
    return indexed(audio, Joke, id);
}

bool playClockUnset(Audio& audio, int16_t id){
    return indexed(audio, ClockUnset, id);
}

bool playClockTime(Audio& audio, unsigned hour24, unsigned minute){
    if(hour24 >= 24 || minute >= 60){
        return false;
    }
    const unsigned hour12 = (hour24 + 11) % 12 + 1;
    const Clip period = hour24 == 0 || hour24 >= 19 ? ClockNight
        : hour24 < 12 ? ClockMorning
        : hour24 == 12 ? ClockNoon : ClockAfternoon;
    const Clip hourIntro = hour12 == 1 ? ClockOne : ClockPlural;
    const Clip hourNumber = hour12 == 1 ? Clip{} : Numbers[hour12];

    if(minute == 0){
        return play(audio, { hourIntro, hourNumber, period, ClockOnTime });
    }
    if(minute < 30){
        return play(audio, { hourIntro, hourNumber, ClockWith, Numbers[minute], period });
    }
    const unsigned decade = minute / 10 - 3;
    if(minute % 10 == 0){
        return play(audio, { hourIntro, hourNumber, ClockWith, Numbers[30 + decade], period });
    }
    return play(audio, { hourIntro, hourNumber, ClockWith, CompoundDecades[decade], Numbers[minute % 10], period });
}

bool playDiceIntro(Audio& audio){
    return play(audio, { DiceIntro });
}

bool playDiceResult(Audio& audio, unsigned face){
    if(face == 0 || face > std::size(DiceFaces)){
        return false;
    }
    return play(audio, { DiceFaces[face - 1] });
}

bool playListening(Audio& audio){
    return play(audio, { Listening });
}

bool playUnrecognized(Audio& audio){
    return play(audio, { Unrecognized });
}

bool playCancelled(Audio& audio){
    return play(audio, { Cancelled });
}

}

#endif
