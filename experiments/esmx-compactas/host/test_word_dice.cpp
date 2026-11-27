#include "WordDice.h"
#include "DiceContrast.h"

#include <cstdlib>

using namespace EsMx::Kws;
namespace {

void require(bool condition){
	if(!condition){
		std::abort();
	}
}

WordBank fakeLexicon(){
	WordBank model{};
	model.inputShift = 1;
	model.hiddenShift = 1;
	model.wordMargin = 1;
	model.vetoMargin = 1;
	for(int word = 1; word < WordLabels; ++word){
		model.input[(word * Bands + word) * WordWindow + WordWindow / 2] = 2;
		model.hidden[word * WordHidden + word] = 2;
		model.output[word * WordHidden + word] = 2;
	}
	return model;
}

void word(WordFeatures& input, size_t start, int label, size_t duration = 3){
	for(size_t frame = start; frame < start + duration; ++frame){
		input.frames[frame].fill(0);
		input.frames[frame][label] = 32;
	}
}
}

int main(){
	const auto model = fakeLexicon();
	require(validWordBank(model));
	auto invalid = model;
	invalid.inputShift = 0;
	require(!validWordBank(invalid));

	WordFeatures signal{};
	signal.count = 25;
	word(signal, 1, 1);
	word(signal, 8, 2);
	word(signal, 15, 3);
	require(acceptsWordDice(signal, model));
	require(confirmContinuousDice(Intent::Unknown, Intent::Dice6, signal, model) == Intent::Dice6);

	auto contrasted = signal;
	word(contrasted, 15, 5);
	require(!acceptsWordDice(contrasted, model));
	require(confirmContinuousDice(Intent::Unknown, Intent::Dice6, contrasted, model) == Intent::Unknown);
	contrasted = signal;
	word(contrasted, 22, 4);
	require(!acceptsWordDice(contrasted, model));
	require(confirmContinuousDice(Intent::Unknown, Intent::Dice6, contrasted, model) == Intent::Unknown);
	require(confirmContinuousDice(Intent::Dice6, Intent::Dice6, contrasted, model) == Intent::Dice6);
	require(confirmContinuousDice(Intent::Fact, Intent::Dice6, contrasted, model) == Intent::Fact);
	contrasted = signal;
	word(contrasted, 22, 6);
	require(!acceptsWordDice(contrasted, model));
	const auto rescue = [&](const WordFeatures& features){
		return confirmContinuousDice(Intent::Unknown, Intent::Dice6, features, model, DiceConfirmation::OrderedRescue);
	};
	require(confirmContinuousDice(Intent::Unknown, Intent::Dice6, contrasted, model) == Intent::Unknown);
	require(rescue(contrasted) == Intent::Dice6);

	auto contextOnly = contrasted;
	word(contextOnly, 1, 0);
	require(rescue(contextOnly) == Intent::Unknown);
	word(contextOnly, 18, 1);
	require(rescue(contextOnly) == Intent::Unknown);
	auto tooClose = contrasted;
	word(tooClose, 1, 0);
	word(tooClose, 10, 1);
	require(rescue(tooClose) == Intent::Unknown);
	auto tooFar = contrasted;
	tooFar.count = 95;
	word(tooFar, 15, 0);
	word(tooFar, 85, 3);
	require(rescue(tooFar) == Intent::Unknown);

	// Two-frame tokens each emit one stable detection, so the measured
	// gap equals the difference between their start frames.
	for(size_t gap : {6, 7, 70, 71}){
		WordFeatures boundary{};
		boundary.count = 84;
		word(boundary, 1, 1, 2);
		word(boundary, 1 + gap, 3, 2);
		word(boundary, 79, 6);
		require((rescue(boundary) == Intent::Dice6) == (gap == 7 || gap == 70));
	}

	// Continued evidence for the same too-early «seis» is not a new
	// onset and must not eventually move into the admissible interval.
	WordFeatures prolonged{};
	prolonged.count = 27;
	word(prolonged, 1, 1, 2);
	word(prolonged, 5, 3, 13);
	word(prolonged, 22, 6, 2);
	require(rescue(prolonged) == Intent::Unknown);

	for(int veto : {4, 5}){
		auto vetoed = contrasted;
		word(vetoed, 4, veto);
		require(rescue(vetoed) == Intent::Unknown);
	}
	require(confirmContinuousDice(Intent::Unknown, Intent::Unknown, contrasted, model,
	                              DiceConfirmation::OrderedRescue) == Intent::Unknown);

	contrasted = signal;
	word(contrasted, 1, 2);
	require(!acceptsWordDice(contrasted, model));

	// Scale the fixture logits to the fixed contrast margins.
	auto strong = model;
	for(auto& weight : strong.output){
		weight *= 32;
	}
	require(!acceptsContrastedDice(signal, model));
	WordFeatures validContrast{};
	validContrast.count = 30;
	word(validContrast, 1, 1, 2);
	word(validContrast, 9, 2, 2);
	word(validContrast, 17, 3, 2);
	require(acceptsContrastedDice(validContrast, strong));
	for(int veto : {4, 5, 6}){
		auto vetoed = validContrast;
		word(vetoed, 23, veto, 2);
		require(!acceptsContrastedDice(vetoed, strong));
	}
	auto denied = validContrast;
	word(denied, 23, 4, 2);
	auto missingDie = validContrast;
	word(missingDie, 9, 0, 2);
	require(!acceptsContrastedDice(missingDie, strong));
	auto reversed = validContrast;
	word(reversed, 9, 3, 2);
	word(reversed, 17, 2, 2);
	require(!acceptsContrastedDice(reversed, strong));
	WordFeatures prematureSix{};
	prematureSix.count = 30;
	word(prematureSix, 1, 1, 2);
	word(prematureSix, 7, 3, 12);
	for(size_t frame : {7, 8}){
		prematureSix.frames[frame][3] = 72;
		prematureSix.frames[frame][2] = 32;
	}
	// A single run starting too early can't enter the valid window later.
	require(!acceptsContrastedDice(prematureSix, strong));
	auto dieBeforeVerb = validContrast;
	word(dieBeforeVerb, 1, 2, 2);
	word(dieBeforeVerb, 9, 1, 2);
	require(!acceptsContrastedDice(dieBeforeVerb, strong));

	AcousticDice acoustic{};
	acoustic.first.shift = acoustic.second.shift = acoustic.third.shift = 1;
	acoustic.biases = {0, 8000, 9000, 0};
	acoustic.minimumScore = 7900;
	acoustic.minimumMargin = 5000;
	const Signature signature{};
	require(completeAcousticDice(signature, Intent::Unknown, Intent::Unknown, acoustic) == Intent::Unknown);
	require(completeContrastedDice(signature, Intent::Unknown, Intent::Unknown, acoustic, validContrast, strong) == Intent::Dice6);
	require(completeContrastedDice(signature, Intent::Unknown, Intent::Unknown, acoustic, denied, strong) == Intent::Unknown);
	require(completeContrastedDice(signature, Intent::Fact, Intent::Unknown, acoustic, validContrast, strong) == Intent::Unknown);
	require(completeContrastedDice(signature, Intent::Time, Intent::Time, acoustic, validContrast, strong) == Intent::Time);
	// Keep the established path when the model already ranks D6 first
	// but cannot confirm it against the other classes.
	acoustic.biases[2] = 7000;
	require(completeContrastedDice(signature, Intent::Unknown, Intent::Unknown, acoustic, validContrast, strong) == Intent::Unknown);

	bb::asr::LogMel frontend{};
	const std::array<const WordBank*, 2> supplements{&model, &model};
	for(size_t count = 0; count <= supplements.size(); ++count){
		const auto configured = std::span(supplements).first(count);
		require(completeWordDice(frontend, {}, Intent::Fact, Intent::Unknown, model, signal, configured) == Intent::Unknown);
		require(completeWordDice(frontend, {}, Intent::Unknown, Intent::Dice6, model, signal, configured) == Intent::Dice6);
	}
	for(auto intent : {Intent::Butter, Intent::YouButter, Intent::Joke, Intent::Fact, Intent::Time}){
		require(completeWordDice(frontend, {}, intent, intent, model, signal) == intent);
	}
	require(completeWordDice(frontend, {}, Intent::Unknown, Intent::Unknown, model, signal) == Intent::Unknown);
	require(signal.count == 0);
	require(confirmContinuousDice(Intent::Unknown, Intent::Dice6, signal, model) == Intent::Unknown);
}
