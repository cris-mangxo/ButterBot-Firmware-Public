#include "PhraseGate.h"

#include <algorithm>

namespace EsMx::Kws {

bool validPhraseBank(const PhraseBank& bank){
	const auto validRadius = [](const BoundedPhrase& item){ return item.radius <= MaxPhraseCost; };
	return bank.factGap > 0 && bank.factGap <= MaxPhraseCost &&
	       std::all_of(bank.dice.begin(), bank.dice.end(), validRadius) &&
	       std::all_of(bank.deniedFacts.begin(), bank.deniedFacts.end(), validRadius);
}

Intent verifyPhrase(const Signature& signature, Intent baseline, const PhraseBank& bank){
	if(baseline == Intent::Fact){
		const auto affirmative = std::min(signatureDistance(signature, bank.facts[0]),
		                                  signatureDistance(signature, bank.facts[1]));
		for(const auto& item : bank.deniedFacts){
			if(item.radius == 0){
				continue;
			}
			const auto cost = signatureDistance(signature, item.signature);
			if(cost <= item.radius && affirmative > cost && affirmative - cost >= bank.factGap){
				return Intent::Unknown;
			}
		}
		return baseline;
	}
	if(baseline != Intent::Unknown && baseline != Intent::Dice6){
		return baseline;
	}
	for(const auto& item : bank.dice){
		if(item.radius > 0 && signatureDistance(signature, item.signature) <= item.radius){
			return Intent::Dice6;
		}
	}
	return Intent::Unknown;
}
}
