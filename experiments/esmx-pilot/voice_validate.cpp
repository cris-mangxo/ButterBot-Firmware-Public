#include <array>
#include <cstdio>

#include "aacdec.h"
#include "voice_data.h"

int main(){
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
    std::printf("{\"aac_bytes\":%zu,\"aac_frames\":%u,\"sample_rate\":16000,\"channels\":1}\n",
                sizeof(Samples), frames);
    return frames == 0;
}
