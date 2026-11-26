#pragma once

#include "Audio/EsMxKws/Compact.h"

namespace EsMx::Kws {

constexpr uint32_t MaxPhraseCost = Frames * Bands * 255;

struct BoundedPhrase {
	Signature signature;
	uint32_t radius; // Zero disables this individual template.
};

struct PhraseBank {
	std::array<BoundedPhrase, 12> dice;
	std::array<Signature, 2> facts;
	std::array<BoundedPhrase, 6> deniedFacts;
	uint32_t factGap;
};

bool validPhraseBank(const PhraseBank& bank);
Intent verifyPhrase(const Signature& signature, Intent baseline, const PhraseBank& bank);
}
