#pragma once

#include "Kernel.h"
#include <cstdint>
#include <optional>

namespace EsMx::PhonemeNative {

constexpr std::size_t MaxTemplatePhones = 32;

struct Alignment {
	double deficitPerPhone;
	std::uint16_t beginFrame;
	std::uint16_t endFrame;
};

// Frame-major logits. Token zero is reserved for the CTC blank.
// The result uses a half-open interval and the earliest end on equal cost.
std::optional<Alignment> align(std::span<const float> logits, std::span<const std::uint8_t> phones);

}
