#include "WordDice.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
using namespace EsMx::Kws;
int main(){
    WordFeatures input{};
    WordBank bank{};
    bank.inputShift = bank.hiddenShift = 9;
    uint32_t rng = 0x1357;
    const auto fill = [&](auto& dst){
        for(auto& v : dst){
            rng = rng * UINT32_C(1664525) + UINT32_C(1013904223);
            v = int8_t(rng >> 24);
        }
    };
    fill(bank.input);
    fill(bank.hidden);
    fill(bank.output);
    for(auto& row : input.frames){
        fill(row);
    }
    uint32_t checksum = 0;
    auto before = std::chrono::steady_clock::now();
    for(unsigned pass = 0; pass < 500; ++pass){
        input.count = 1 + (pass * 17 % input.frames.size());
        for(size_t t = 0; t < input.count; ++t){
            const auto result = wordLogits(input, t, bank);
            for(const auto v : result){
                checksum = checksum * UINT32_C(16777619) ^ uint32_t(v);
            }
        }
    }
    auto after = std::chrono::steady_clock::now();
    std::printf("%08x %lld\n", checksum, static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(after-before).count()));
}
