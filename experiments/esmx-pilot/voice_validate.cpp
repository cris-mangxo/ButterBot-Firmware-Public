#include <array>
#include <cstdio>
#include <initializer_list>

#include "aacdec.h"
#include "Audio/EsMxPilot/ClockSpeech.h"

namespace {

bool matches(const EsMx::Pilot::ClockSpeech& composed, std::initializer_list<EsMx::Pilot::VoiceClip> expected){
    auto clip = expected.begin();
    for(const auto part : composed){
        if(part.empty()){
            continue;
        }
        if(clip == expected.end() || clip->data() != part.data() || clip->size() != part.size()){
            return false;
        }
        ++clip;
    }
    return clip == expected.end();
}

bool clockIsValid(){
    using namespace EsMx::Pilot;
    using namespace VoiceData;
    for(unsigned hour = 0; hour < 24; ++hour){
        for(unsigned minute = 0; minute < 60; ++minute){
            if(clockSpeech(hour, minute).front().empty()){
                return false;
            }
        }
    }
    return matches(clockSpeech(0, 0), {ClockPlural, Numbers[12], ClockNight, ClockOnTime})
        && matches(clockSpeech(1, 1), {ClockOne, ClockWith, ClockOneMinute, ClockMorning})
        && matches(clockSpeech(12, 0), {ClockPlural, Numbers[12], ClockNoon, ClockOnTime})
        && matches(clockSpeech(13, 29), {ClockOne, ClockWith, Numbers[29], ClockAfternoon})
        && matches(clockSpeech(18, 30), {ClockPlural, Numbers[6], ClockWith, Numbers[30], ClockAfternoon})
        && matches(clockSpeech(19, 31), {ClockPlural, Numbers[7], ClockWith, CompoundDecades[0], Numbers[1], ClockNight})
        && matches(clockSpeech(23, 59), {ClockPlural, Numbers[11], ClockWith, CompoundDecades[2], Numbers[9], ClockNight})
        && matches(clockSpeech(24, 0), {})
        && matches(clockSpeech(0, 60), {});
}

}

int main(){
    if(!clockIsValid()){
        std::fputs("Incorrect Spanish clock composition\n", stderr);
        return 1;
    }
    using EsMx::Pilot::VoiceData::Samples;
    HAACDecoder decoder = AACInitDecoder();
    if(decoder == nullptr){
        std::fputs("Could not initialize the factory AAC decoder\n", stderr);
        return 1;
    }

    // The decoder API takes a mutable pointer to advance it without modifying the stream.
    auto* input = const_cast<unsigned char*>(Samples);
    int remaining = sizeof(Samples);
    unsigned frames = 0;
    std::array<short, 2 * AAC_MAX_NSAMPS> output{};
    while(remaining > 0){
        const int before = remaining;
        const int result = AACDecode(decoder, &input, &remaining, output.data());
        if(result != ERR_AAC_NONE || remaining >= before){
            std::fprintf(stderr, "Factory AAC decoder failed on frame %u (%d)\n", frames, result);
            AACFreeDecoder(decoder);
            return 1;
        }
        AACFrameInfo format{};
        AACGetLastFrameInfo(decoder, &format);
        if(format.nChans != 1 || format.sampRateOut != 16000 || format.bitsPerSample != 16
                || format.outputSamps != AAC_MAX_NSAMPS || format.profile != AAC_PROFILE_LC){
            std::fprintf(stderr, "Unexpected decoder output on frame %u\n", frames);
            AACFreeDecoder(decoder);
            return 1;
        }
        ++frames;
    }
    AACFreeDecoder(decoder);
    std::printf("{\"aac_bytes\":%zu,\"aac_frames\":%u,\"sample_rate\":16000,\"channels\":1,\"clock_compositions\":1440}\n",
                sizeof(Samples), frames);
    return frames == 0;
}
