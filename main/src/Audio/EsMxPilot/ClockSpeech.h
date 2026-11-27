#pragma once

#include <array>
#include <cstdint>
#include <iterator>
#include <span>

#include "voice_data.h"

namespace EsMx::Pilot {

using VoiceClip = std::span<const uint8_t>;
using ClockSpeech = std::array<VoiceClip, 6>;

inline constexpr ClockSpeech clockSpeech(unsigned hour24, unsigned minute){
    using namespace VoiceData;
    static_assert(std::size(Numbers) == 33);
    static_assert(std::size(CompoundDecades) == 3);
    if(hour24 >= 24 || minute >= 60){
        return {};
    }

    const unsigned hour12 = (hour24 + 11) % 12 + 1;
    const VoiceClip period = hour24 == 0 || hour24 >= 19 ? ClockNight
        : hour24 < 12 ? ClockMorning
        : hour24 == 12 ? ClockNoon : ClockAfternoon;
    const VoiceClip intro = hour12 == 1 ? ClockOne : ClockPlural;
    const VoiceClip hour = hour12 == 1 ? VoiceClip{} : Numbers[hour12];
    if(minute == 0){
        return { intro, hour, period, ClockOnTime };
    }
    if(minute == 1){
        return { intro, hour, ClockWith, ClockOneMinute, period };
    }
    if(minute < 30){
        return { intro, hour, ClockWith, Numbers[minute], period };
    }
    const unsigned decade = minute / 10 - 3;
    const unsigned unit = minute % 10;
    if(unit == 0){
        return { intro, hour, ClockWith, Numbers[30 + decade], period };
    }
    return { intro, hour, ClockWith, CompoundDecades[decade], Numbers[unit], period };
}

}
