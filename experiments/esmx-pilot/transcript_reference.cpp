#include "Audio/EsMxCommand.h"
#include "DiceGrammar.h"
#include "dispatch_probe.h"

#include <array>
#include <iostream>
#include <string>
#include <string_view>

namespace {
using EsMx::Kws::Intent;
using namespace std::literals;

Intent classifyText(std::string_view text){
    if(EsMx::Kws::isDice6Request(text)){
        return Intent::Dice6;
    }
    if(const auto command = EsMx::matchCommand(text)){
        if(command->name == "pass_the_butter"){
            return Intent::Butter;
        }
        if(command->name == "joke"){
            return Intent::Joke;
        }
        if(command->name == "current_time"){
            return Intent::Time;
        }
    }
    // Fixed full-turn aliases for common requests about the current time.
    constexpr std::array TimeAliases{"qué horas son"sv, "qué horas son ahorita"sv,
                                     "me puedes decir la hora"sv, "dime qué hora es"sv,
                                     "me puede decir qué hora es"sv, "me puede dar la hora"sv};
    std::string lower(text);
    for(auto& ch : lower){
        if(ch >= 'A' && ch <= 'Z'){
            ch += 'a' - 'A';
        }
    }
    text = lower;
    if(text.starts_with("¿")){
        text.remove_prefix("¿"sv.size());
    }
    if(!text.empty() && (text.back() == '?' || text.back() == '.')){
        text.remove_suffix(1);
    }
    for(const auto alias : TimeAliases){
        if(text == alias){
            return Intent::Time;
        }
    }
    return Intent::Unknown;
}
}

int main(){
    std::array<char, EsMx::Kws::MaxTranscriptBytes + 1> input{};
    std::cin.read(input.data(), input.size());
    if(!std::cin.eof() || std::cin.bad()){
        return 2;
    }
    const auto intent = classifyText({input.data(), static_cast<size_t>(std::cin.gcount())});
    std::cout << "{\"intent\":" << static_cast<int>(intent)
              << ",\"route\":\"" << probeDispatch(intent) << "\"}\n";
}
