#include "Kernel.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace EsMx::PhonemeNative {
namespace {

template <typename LayerType>
float weight(const LayerType& layer, std::size_t channel, std::size_t index) {
    return static_cast<float>(layer.weights[index]) * layer.scales[channel];
}

template <std::size_t Outputs, std::size_t Inputs>
std::array<float, Outputs> project(const Layer<Outputs, Inputs>& layer, const float* input) {
    std::array<float, Outputs> output;
    for(std::size_t out = 0; out < Outputs; ++out) {
        float total = 0;
        const auto begin = out * Inputs;
        for(std::size_t in = 0; in < Inputs; ++in) {
            total += input[in] * weight(layer, out, begin + in);
        }
        output[out] = total + layer.bias[out];
    }
    return output;
}

void encode(std::span<const int8_t> input, int frames, const Bank& bank, float* result) {
    for(int frame = 0; frame < frames; ++frame) {
        for(std::size_t out = 0; out < Width; ++out) {
            float total = 0;
            for(std::size_t band = 0; band < Bands; ++band) {
                for(int tap = 0; tap < static_cast<int>(Taps); ++tap) {
                    const int index = frame + tap - 2;
                    if(index >= 0 && index < frames) {
                        const auto pos = (out * Bands + band) * Taps + tap;
                        const auto sample = input[index * Bands + band] / 32.0f;
                        total += sample * weight(bank.input, out, pos);
                    }
                }
            }
            result[frame * Width + out] = std::max(0.0f, total + bank.input.bias[out]);
        }
    }
}

void apply(const Block& block, int dilation, int frames, const float* input, float* output) {
    for(int frame = 0; frame < frames; ++frame) {
        std::array<float, Width> depthwise;
        for(std::size_t channel = 0; channel < Width; ++channel) {
            float total = 0;
            for(int tap = 0; tap < static_cast<int>(Taps); ++tap) {
                const int index = frame + (tap - 2) * dilation;
                if(index >= 0 && index < frames) {
                    const auto pos = channel * Taps + tap;
                    total += input[index * Width + channel] * weight(block.temporal, channel, pos);
                }
            }
            depthwise[channel] = total + block.temporal.bias[channel];
        }
        const auto mixed = project(block.channel, depthwise.data());
        for(std::size_t channel = 0; channel < Width; ++channel) {
            const auto index = frame * Width + channel;
            output[index] = std::max(0.0f, input[index] + mixed[channel]);
        }
    }
}

uint8_t gate(const std::array<float, IntentCount>& logits, float negation) {
    constexpr std::array<uint8_t, IntentCount> IntentIds{0, 1, 3, 5, 6};
    if(1.0f / (1.0f + std::exp(-negation)) >= 0.50f) {
        return 0;
    }
    const auto best = std::max_element(logits.begin(), logits.end());
    const auto bestIndex = static_cast<std::size_t>(best - logits.begin());
    float second = -std::numeric_limits<float>::infinity();
    float sum = 0;
    for(std::size_t index = 0; index < logits.size(); ++index) {
        sum += std::exp(logits[index] - *best);
        if(index != bestIndex) {
            second = std::max(second, logits[index]);
        }
    }
    const float top = 1.0f / sum;
    const float other = std::exp(second - *best) / sum;
    return top >= 0.70f && top - other >= 0.20f ? IntentIds[bestIndex] : 0;
}

}

bool infer(std::span<const int8_t> frames, const Bank& bank, Workspace& workspace, Output& output) {
    if(frames.empty() || frames.size() > MaxFrames * Bands || frames.size() % Bands != 0) {
        return false;
    }
    const auto count = static_cast<int>(frames.size() / Bands);
    auto* current = workspace.first.data();
    auto* next = workspace.second.data();
    encode(frames, count, bank, current);
    for(std::size_t index = 0; index < bank.blocks.size(); ++index) {
        apply(bank.blocks[index], 1 << index, count, current, next);
        std::swap(current, next);
    }

    std::array<float, PoolSize> pool{};
    for(int frame = 0; frame < count; ++frame) {
        const auto* hidden = current + frame * Width;
        const auto quarter = 1 + (4 * frame / count);
        for(std::size_t channel = 0; channel < Width; ++channel) {
            pool[channel] += hidden[channel];
            const auto index = quarter * Width + channel;
            pool[index] = std::max(pool[index], hidden[channel]);
        }
        const auto phone = project(bank.phones, hidden);
        std::copy(phone.begin(), phone.end(), output.phones.begin() + frame * PhoneCount);
    }
    for(std::size_t channel = 0; channel < Width; ++channel) {
        pool[channel] /= static_cast<float>(count);
    }
    output.intents = project(bank.intent, pool.data());
    output.negation = project(bank.negation, pool.data()).front();
    output.intentId = gate(output.intents, output.negation);
    return true;
}

}
