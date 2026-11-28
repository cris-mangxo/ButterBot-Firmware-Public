#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace EsMx::PhonemeNative {

inline constexpr std::size_t Bands = 20;
inline constexpr std::size_t Width = 28;
inline constexpr std::size_t PhoneCount = 32;
inline constexpr std::size_t IntentCount = 5;
inline constexpr std::size_t MaxFrames = 199;
inline constexpr std::size_t Taps = 5;
inline constexpr std::size_t PoolSize = Width * 5;

template <std::size_t Outputs, std::size_t Inputs>
struct Layer {
    std::array<int8_t, Outputs * Inputs> weights;
    std::array<float, Outputs> scales;
    std::array<float, Outputs> bias;
};

struct Block {
    Layer<Width, Taps> temporal;
    Layer<Width, Width> channel;
};

struct Bank {
    Layer<Width, Bands * Taps> input;
    std::array<Block, 3> blocks;
    Layer<PhoneCount, Width> phones;
    Layer<IntentCount, PoolSize> intent;
    Layer<1, PoolSize> negation;
};

struct Workspace {
    std::array<float, MaxFrames * Width> first;
    std::array<float, MaxFrames * Width> second;
};

struct Output {
    std::array<float, MaxFrames * PhoneCount> phones;
    std::array<float, IntentCount> intents;
    float negation;
    uint8_t intentId;
};

static_assert(sizeof(Bank) == 9180);
static_assert(sizeof(Workspace) == 44576);

// Frame-major signed int8 output from the existing word-frames extractor.
// Workspace and output are caller-owned. Do not put them on an ESP32 task stack.
bool infer(std::span<const int8_t> frames, const Bank& bank, Workspace& workspace, Output& output);

}
