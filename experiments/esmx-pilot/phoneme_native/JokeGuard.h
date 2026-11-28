#pragma once

#include "Alignment.h"

namespace EsMx::PhonemeNative {

using TokenPath = std::span<const std::uint8_t>;

struct JokeRule {
	std::span<const TokenPath> noun, request, denial, positive, negative;
	std::array<double, 9> coefficients;
	double intercept;
	double threshold;
};

std::optional<double> scoreJoke(const Output& output, std::size_t frames, const JokeRule& rule);
bool confirmsJoke(const Output& output, std::size_t frames, const JokeRule& rule);

}
