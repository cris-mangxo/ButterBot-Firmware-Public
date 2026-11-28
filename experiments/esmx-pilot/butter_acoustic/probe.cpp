#include "Audio/EsMxPilot/ButterAcoustic.h"
#include "Audio/EsMxPilot/ClockAcoustic.h"
#include "Audio/EsMxKws/Compact.h"
#include "ButterModelAssets.h"
#include <array>
#include <iostream>

int main(){
    using namespace EsMx;
    std::array<int16_t, Kws::MaxSamples + 1> pcm{};
    std::cin.read(reinterpret_cast<char*>(pcm.data()), sizeof(pcm));
    const auto bytes = static_cast<size_t>(std::cin.gcount());
    if(bytes == 0 || bytes > Kws::MaxSamples * sizeof(int16_t) || bytes % sizeof(int16_t) || std::cin.bad()){
        return 2;
    }

    const auto samples = std::span(pcm).first(bytes / sizeof(int16_t));
    const bool clockBefore = Pilot::recognizesAcousticClock(samples);
    const auto butter = Pilot::measureAcousticButter(samples);
    if(clockBefore != Pilot::recognizesAcousticClock(samples)){
        std::cerr << "Shared decoder changed the result of the Hora search\n";
        return 1;
    }
    std::cout << std::boolalpha
              << "{\"complete\":" << butter.complete
              << ",\"clock\":" << clockBefore
              << ",\"affirmative\":" << butter.affirmative
              << ",\"opposing\":" << butter.opposing
              << ",\"frames\":" << butter.frames
              << ",\"last_speech\":" << butter.lastSpeech
              << ",\"affirm_score\":" << butter.affirmScore
              << ",\"oppose_score\":" << butter.opposeScore
              << ",\"object\":[" << butter.affirmObject.start << ',' << butter.affirmObject.end
              << "],\"accepted\":" << Pilot::acceptsAcousticButter(butter)
              << ",\"article\":[" << butter.affirmArticle.start << ',' << butter.affirmArticle.end << ']'
              << ",\"ran_fsg\":" << butter.ranFsg
              << ",\"profile_enabled\":" << Pilot::ButterModelAssets::ValidationEnabled
              << ",\"hits\":[";
    const auto hits = std::span(butter.hits).first(butter.hitCount);
    for(size_t i = 0; i < hits.size(); ++i){
        const auto& hit = hits[i];
        if(i){
            std::cout << ',';
        }
        std::cout << '[' << static_cast<int>(hit.type) << ',' << hit.segment.start << ',' << hit.segment.end
                  << ',' << hit.segment.score << ']';
    }
    std::cout << "]}\n";
    return std::cout ? 0 : 2;
}
