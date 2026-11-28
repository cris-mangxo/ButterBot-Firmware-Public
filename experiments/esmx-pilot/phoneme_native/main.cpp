#include "Kernel.h"
#include "student_bank.h"

#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    using namespace EsMx::PhonemeNative;
    const bool dump = argc == 2 && std::string_view(argv[1]) == "--logits";
    if(argc != 1 && !dump) {
        return 2;
    }

    static std::array<int8_t, MaxFrames * Bands + 1> frames;
    static Workspace workspace;
    static Output output;
    std::cin.read(reinterpret_cast<char*>(frames.data()), frames.size());
    const auto length = static_cast<std::size_t>(std::cin.gcount());
    if(std::cin.bad() || !infer(std::span(frames).first(length), Frozen, workspace, output)) {
        return 2;
    }

    if(dump) {
        // Host transport: little-endian f32 phoneme logits (frame-major), then
        // f32 turn logits, f32 negation, and one byte for the final intent ID.
        std::cout.write(reinterpret_cast<const char*>(output.phones.data()),
                        length / Bands * PhoneCount * sizeof(float));
        std::cout.write(reinterpret_cast<const char*>(output.intents.data()), sizeof(output.intents));
        std::cout.write(reinterpret_cast<const char*>(&output.negation), sizeof(output.negation));
        std::cout.put(static_cast<char>(output.intentId));
    } else {
        std::cout << static_cast<unsigned>(output.intentId) << '\n';
    }
    return std::cout ? 0 : 2;
}
