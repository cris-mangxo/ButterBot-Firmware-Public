#include "PhraseGate.h"

#include <cassert>
#include <initializer_list>

using namespace EsMx::Kws;

namespace {
Signature constant(int8_t value){
	Signature signature;
	for(auto& frame : signature.frames){
		frame.fill(value);
	}
	return signature;
}
}

int main(){
	const auto d6 = constant(0);
	const auto affirmative = constant(10);
	const auto negation = constant(20);
	const auto outside = constant(30);

	PhraseBank bank{};
	bank.dice[0] = {d6, 1000};
	bank.facts = {affirmative, affirmative};
	bank.deniedFacts[0] = {negation, 1000};
	bank.factGap = 256;
	assert(validPhraseBank(bank));

	assert(verifyPhrase(d6, Intent::Unknown, bank) == Intent::Dice6);
	assert(verifyPhrase(outside, Intent::Dice6, bank) == Intent::Unknown);
	assert(verifyPhrase(negation, Intent::Fact, bank) == Intent::Unknown);
	assert(verifyPhrase(affirmative, Intent::Fact, bank) == Intent::Fact);
	assert(verifyPhrase(negation, Intent::Unknown, bank) == Intent::Unknown);
	for(auto intent : {Intent::Butter, Intent::YouButter, Intent::Joke, Intent::Time}){
		assert(verifyPhrase(negation, intent, bank) == intent);
		assert(verifyPhrase(d6, intent, bank) == intent);
	}

	bank.facts = {negation, affirmative};
	assert(verifyPhrase(negation, Intent::Fact, bank) == Intent::Fact);
	bank.dice[0].radius = 0;
	assert(verifyPhrase(d6, Intent::Dice6, bank) == Intent::Unknown);
	bank.deniedFacts[0].radius = MaxPhraseCost + 1;
	assert(!validPhraseBank(bank));
	bank.deniedFacts[0].radius = 0;
	bank.factGap = 0;
	assert(!validPhraseBank(bank));
}
