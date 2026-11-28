#pragma once

#include "Audio/EsMxKws/Compact.h"

namespace EsMx::Pilot {

enum class TurnDisposition : uint8_t {
	Accepted,
	NoMatch,
	InvalidAudio,
	FactVeto,
	UnconfirmedDice,
	DiceNegationVeto,
	DiceNegationConflict,
	DiceFollowUp,
};

struct RecognitionDecision {
	Kws::Intent intent;
	TurnDisposition disposition;

	// An eligible abstention still needs independent affirmative and negation checks.
	constexpr bool allowsFallback() const {
		return intent == Kws::Intent::Unknown && disposition == TurnDisposition::NoMatch;
	}
};

// Call from the pilot's single recognition task with a complete, 16 kHz mono turn.
RecognitionDecision recognizeDetailed(std::span<const int16_t> pcm);
Kws::Intent recognize(std::span<const int16_t> pcm);

}
