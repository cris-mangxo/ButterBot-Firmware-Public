#pragma once

#include "Alignment.h"

namespace EsMx::PhonemeNative {

struct ClockRule {
	std::span<const std::span<const std::uint8_t>> now, event, other, noun, localNow;
	std::span<const std::uint8_t> eventPrefix, nowPrefix;
	std::array<double, 9> coefficients;
	double intercept;
	double threshold;
};

// Scratch is borrowed after inference, and must not overlap Output.
std::optional<double> scoreClock(const Output& output, std::size_t frames,
                                const ClockRule& rule, std::span<float> scratch);

}
