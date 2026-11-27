#include "WordDice.h"
#include "DiceContrast.h"
#include "DiceConsensus.h"

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

AcousticDice fakeAcoustic(DiceScores scores){
	AcousticDice model{};
	model.first.shift = model.second.shift = model.third.shift = 1;
	model.minimumScore = 4;
	model.minimumMargin = 6;
	model.biases = scores;
	return model;
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

	// The supplementary route requires two shared «seis» frames and checks
	// negation before and after them over the entire turn.
	WordFeatures agreed{};
	agreed.count = 25;
	word(agreed, 9, 3, 2);
	require(confirmsSixAcrossBanks(agreed, model, model, model));
	auto silentBank = model;
	silentBank.output.fill(0);
	require(!confirmsSixAcrossBanks(agreed, model, silentBank, model));
	for(size_t start : {1, 20}){
		auto denied = agreed;
		word(denied, start, 4, 3);
		require(!confirmsSixAcrossBanks(denied, model, model, model));
		denied.frames[start + 2].fill(0);
		require(confirmsSixAcrossBanks(denied, model, model, model));
	}
	const Signature empty{};
	const auto outside = fakeAcoustic({9, 4, 1, -7});
	const auto positive = fakeAcoustic({1, 11, 1, 0});
	const auto closeNegation = fakeAcoustic({1, 8, 1, 5});
	const auto negative = fakeAcoustic({1, 0, 1, 11});
	const auto weakNegative = fakeAcoustic({1, 0, 1, 2});
	const auto combined = [&](Intent baseline, Intent verified, const AcousticDice& reference,
	                          const AcousticDice& candidate, const WordFeatures& words){
		return completeConsensusDice(empty, baseline, verified, reference, candidate, words, model, model, model);
	};
	require(combined(Intent::Unknown, Intent::Unknown, outside, positive, agreed) == Intent::Dice6);
	require(combined(Intent::Unknown, Intent::Unknown, closeNegation, positive, agreed) == Intent::Unknown);
	require(!hasCompetingNegation({1, 11, 1, 5}, closeNegation));
	require(!hasCompetingNegation({1, 1, 1, 1}, closeNegation));
	require(hasCompetingNegation({9, 0, 1, 8}, closeNegation));
	require(combined(Intent::Unknown, Intent::Unknown, positive, negative, agreed) == Intent::Dice6);
	require(combined(Intent::Fact, Intent::Unknown, positive, positive, agreed) == Intent::Unknown);
	require(combined(Intent::Unknown, Intent::Dice6, negative, negative, agreed) == Intent::Unknown);
	require(combined(Intent::Unknown, Intent::Dice6, negative, weakNegative, agreed) == Intent::Dice6);
	require(combined(Intent::Unknown, Intent::Dice6, negative, positive, agreed) == Intent::Dice6);
	require(combined(Intent::Unknown, Intent::Dice6, positive, negative, agreed) == Intent::Dice6);
	require(combined(Intent::Unknown, Intent::Dice6, negative, negative, signal) == Intent::Dice6);

	auto delayedSix = model;
	delayedSix.input[(3 * Bands + 3) * WordWindow + WordWindow / 2] = 0;
	delayedSix.input[(3 * Bands + 7) * WordWindow + WordWindow / 2] = 2;
	WordFeatures orderedDie{};
	orderedDie.count = 30;
	word(orderedDie, 4, 2, 2);
	word(orderedDie, 14, 7, 1);
	word(orderedDie, 15, 3, 2);
	require(confirmsOrderedDieSix(orderedDie, model, model, delayedSix));
	auto withoutNoun = orderedDie;
	word(withoutNoun, 4, 0, 2);
	require(!confirmsOrderedDieSix(withoutNoun, model, model, delayedSix));
	auto tooFarApart = orderedDie;
	tooFarApart.frames[14].fill(0);
	word(tooFarApart, 21, 7, 1);
	require(!confirmsOrderedDieSix(tooFarApart, model, model, delayedSix));
	auto extraDenialBank = model;
	extraDenialBank.input[(4 * Bands + 4) * WordWindow + WordWindow / 2] = 0;
	extraDenialBank.input[(4 * Bands + 8) * WordWindow + WordWindow / 2] = 2;
	auto trailingNo = orderedDie;
	word(trailingNo, 23, 8, 1);
	require(confirmsOrderedDieSix(trailingNo, model, extraDenialBank, delayedSix));
	word(trailingNo, 24, 8, 1);
	require(!confirmsOrderedDieSix(trailingNo, model, extraDenialBank, delayedSix));
	auto replacement = orderedDie;
	word(replacement, 22, 2, 2);
	word(replacement, 26, 5, 3);
	require(!confirmsOrderedDieSix(replacement, model, model, delayedSix));
	for(size_t start : {1, 24}){
		auto denied = orderedDie;
		word(denied, start, 4, 1);
		require(!confirmsOrderedDieSix(denied, model, model, delayedSix));
	}
	for(int dieLead : {3, 4, 59, 60}){
		WordFeatures boundary{};
		boundary.count = 90;
		word(boundary, 2, 2, 2);
		word(boundary, 3 + dieLead, 3, 2);
		require(confirmsOrderedDieSix(boundary, model, model, model) == (dieLead == 4 || dieLead == 59));
	}
	WordFeatures reversedNominal{};
	reversedNominal.count = 30;
	word(reversedNominal, 4, 3, 3);
	word(reversedNominal, 14, 2, 4);
	require(!confirmsOrderedDieSix(reversedNominal, model, model, model));
	const auto weakCompetition = fakeAcoustic({1, 8, 1, 3});
	const auto nominal = [&](const AcousticDice& reference){
		return completeConsensusDice(empty, Intent::Unknown, Intent::Unknown, reference, positive, orderedDie,
		                             model, model, delayedSix, ConsensusMode::OrderedDie);
	};
	require(nominal(weakCompetition) == Intent::Dice6);
	require(nominal(closeNegation) == Intent::Unknown);
	require(completeConsensusDice(empty, Intent::Unknown, Intent::Unknown, weakCompetition, positive, orderedDie,
	                              model, model, delayedSix) == Intent::Unknown);
	require(completeConsensusDice(empty, Intent::Fact, Intent::Fact, outside, positive, orderedDie,
	                              model, model, delayedSix, ConsensusMode::OrderedDie) == Intent::Fact);

	const auto withBackup = [&](Intent baseline, Intent verified, const AcousticDice& reference,
	                            const AcousticDice& first, const WordFeatures& words){
		return completeConsensusDice(empty, baseline, verified, reference, first, words,
		                             model, model, delayedSix, ConsensusMode::OrderedDie, &positive);
	};
	require(withBackup(Intent::Unknown, Intent::Unknown, outside, outside, orderedDie) == Intent::Dice6);
	require(completeConsensusDice(empty, Intent::Unknown, Intent::Unknown, outside, outside, orderedDie,
	                              model, model, delayedSix, ConsensusMode::OrderedDie, &closeNegation)
	        == Intent::Unknown);
	require(completeConsensusDice(empty, Intent::Unknown, Intent::Unknown, outside, positive, agreed,
	                              model, model, model, ConsensusMode::OrderedDie, &outside) == Intent::Dice6);
	require(withBackup(Intent::Unknown, Intent::Unknown, weakCompetition, outside, orderedDie) == Intent::Unknown);
	require(withBackup(Intent::Unknown, Intent::Unknown, closeNegation, outside, orderedDie) == Intent::Unknown);
	require(withBackup(Intent::Unknown, Intent::Unknown, outside, closeNegation, orderedDie) == Intent::Unknown);
	require(withBackup(Intent::Unknown, Intent::Unknown, outside, negative, orderedDie) == Intent::Unknown);
	require(withBackup(Intent::Unknown, Intent::Unknown, outside, outside, withoutNoun) == Intent::Unknown);
	require(completeConsensusDice(empty, Intent::Unknown, Intent::Unknown, outside, outside, trailingNo,
	                              model, extraDenialBank, delayedSix, ConsensusMode::OrderedDie, &positive)
	        == Intent::Unknown);
	require(withBackup(Intent::Fact, Intent::Unknown, outside, outside, orderedDie) == Intent::Unknown);
	require(withBackup(Intent::Fact, Intent::Fact, outside, outside, orderedDie) == Intent::Fact);
	require(completeConsensusDice(empty, Intent::Unknown, Intent::Unknown, outside, outside, orderedDie,
	                              model, model, delayedSix, ConsensusMode::SharedSix, &positive) == Intent::Unknown);

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
	auto invalidCount = validContrast;
	invalidCount.count = invalidCount.frames.size() + 1;
	const std::array<int32_t, WordLabels> noLogits{};
	require(!acceptsWordDice(invalidCount, strong));
	require(wordLogits(invalidCount, 1, strong) == noLogits);
	require(wordLogits(validContrast, validContrast.count, strong) == noLogits);
	require(wordLogits(validContrast, static_cast<size_t>(-1), strong) == noLogits);
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
