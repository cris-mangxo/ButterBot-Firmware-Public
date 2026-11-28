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

class AlignmentFrames {
public:
	// The logits and non-overlapping peak workspace must outlive this view.
	static std::optional<AlignmentFrames> prepare(std::span<const float> logits,
	                                              std::span<float> peakWorkspace);
	std::optional<Alignment> match(std::span<const std::uint8_t> phones, std::size_t frameCount) const;

private:
	AlignmentFrames(std::span<const float> samples, std::span<const float> peaks):
		logits(samples), rowPeaks(peaks){}

	std::span<const float> logits;
	std::span<const float> rowPeaks;
};

// Frame-major logits. Token zero is reserved for the CTC blank.
// The result uses a half-open interval and the earliest end on equal cost.
std::optional<Alignment> align(std::span<const float> logits, std::span<const std::uint8_t> phones);

}
