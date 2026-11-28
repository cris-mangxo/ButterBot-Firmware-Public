#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace EsMx::CompactSpeech {

using Clip = std::span<const uint8_t>;

struct TextClip {
    std::string_view factoryText;
    Clip clip;
};

class Stream {
public:
    bool start(std::string_view text);
    std::size_t generate(std::span<int16_t> output, bool& last);
    void reset();

private:
    void addNumber(unsigned number);
    void add(Clip clip);

    std::array<Clip, 16> clips{};
    std::size_t count = 0;
    std::size_t current = 0;
    std::size_t offset = 0;
};

}
