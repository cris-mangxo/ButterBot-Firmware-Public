#include "JokeGuard.h"
#include "joke_rule.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>

int main(){
    using namespace EsMx::PhonemeNative;
    constexpr std::array frameCounts{36u, 80u, 128u, 199u};
    static std::array<Output, frameCounts.size()> cases{};
    std::uint32_t state = 106543;
    for(auto& record : cases){
        for(auto& element : record.phones){
            state = state * 1103515245u + 12345u;
            element = (static_cast<int>((state >> 16) & 0xffu) - 128) * 0.03125f;
        }
        record.intents = {1.0f, -1.0f, 2.0f, 0.5f, -2.0f};
        record.negation = -0.25f;
    }
    constexpr unsigned iterations = 200;
    double sum = 0;
#ifdef BB_PREPARED_ALIGNMENT
    std::array<float, MaxFrames> peaks;
#endif
    const auto start = std::chrono::steady_clock::now();
    for(unsigned run = 0; run < iterations; ++run){
        for(std::size_t index = 0; index < cases.size(); ++index){
#ifdef BB_PREPARED_ALIGNMENT
            auto result = scoreJoke(cases[index], frameCounts[index], FrozenJoke, peaks);
#else
            auto result = scoreJoke(cases[index], frameCounts[index], FrozenJoke);
#endif
            if(!result){
                return 1;
            }
            sum += *result;
        }
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    std::cout << std::setprecision(17) << sum << ' '
              << std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()
              << ' ' << (iterations * cases.size()) << '\n';
}
