#ifdef BB_ESMX_COMPACT_VOICE
#include "CompactSpeech.h"
#include "compact_voice_bank.h"
#include <algorithm>
#include <charconv>
#include <optional>

namespace EsMx::CompactSpeech {
namespace {

constexpr unsigned MaxNumber = 10'000;

std::optional<unsigned> parseNumber(std::string_view text){
    if(text.empty() || text.size() > 5){
        return std::nullopt;
    }
    unsigned number = 0;
    const auto end = text.data() + text.size();
    const auto parsed = std::from_chars(text.data(), end, number);
    if(parsed.ec != std::errc{} || parsed.ptr != end || number > MaxNumber){
        return std::nullopt;
    }
    return number;
}

int16_t decode(uint8_t code){
    const unsigned bits = static_cast<uint8_t>(~code);
    const int value = (((bits & 0x0f) << 3) + 132) << ((bits >> 4) & 7);
    return static_cast<int16_t>((bits & 0x80) ? 132 - value : value - 132);
}

}

void Stream::reset(){
    count = current = offset = 0;
}

void Stream::add(Clip clip){
    clips[count++] = clip;
}

void Stream::addNumber(unsigned number){
    using namespace Bank;
    if(number >= 1000){
        const unsigned thousands = number / 1000;
        if(thousands > 1){
            add(Small[thousands]);
        }
        add(Thousand);
        number %= 1000;
        if(number == 0){
            return;
        }
    }
    if(number == 100){
        add(HundredExact);
        return;
    }
    if(number >= 100){
        add(Hundreds[number / 100 - 1]);
        number %= 100;
        if(number == 0){
            return;
        }
    }
    if(number >= 30){
        add(Tens[number / 10 - 3]);
        number %= 10;
        if(number == 0){
            return;
        }
        add(And);
    }
    add(Small[number]);
}

bool Stream::start(std::string_view text){
    reset();
    if(const auto entry = std::find_if(std::begin(Bank::Fixed), std::end(Bank::Fixed),
                                      [text](const TextClip& clip){ return clip.factoryText == text; });
       entry != std::end(Bank::Fixed)){
        add(entry->clip);
        return true;
    }

    if(const auto scalar = parseNumber(text)){
        addNumber(*scalar);
        return true;
    }

    constexpr std::string_view RollSuffix = " roll";
    if(!text.ends_with(RollSuffix)){
        return false;
    }
    text.remove_suffix(RollSuffix.size());
    const auto separator = text.find('d');
    if(separator == std::string_view::npos){
        return false;
    }
    const auto amount = parseNumber(text.substr(0, separator));
    const auto sides = parseNumber(text.substr(separator + 1));
    if(!amount || !sides || *amount < 1 || *amount > 100 || *sides < 4 || *sides > 100){
        return false;
    }

    add(Bank::Roll);
    if(*amount == 1){
        add(Bank::OneBeforeNoun);
        add(Bank::Die);
    }else{
        addNumber(*amount);
        add(Bank::Dice);
    }
    addNumber(*sides);
    add(Bank::Sides);
    return true;
}

std::size_t Stream::generate(std::span<int16_t> output, bool& last){
    std::size_t written = 0;
    while(written < output.size() && current < count){
        const Clip clip = clips[current];
        output[written++] = decode(clip[offset++]);
        if(offset == clip.size()){
            ++current;
            offset = 0;
        }
    }
    last = current == count;
    return written;
}

}
#endif
