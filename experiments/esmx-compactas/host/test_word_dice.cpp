#include "WordDice.h"

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

void word(WordFeatures& input, size_t start, int label){
	for(size_t frame = start; frame < start + 3; ++frame){
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

	auto contrasted = signal;
	word(contrasted, 15, 5);
	require(!acceptsWordDice(contrasted, model));
	contrasted = signal;
	word(contrasted, 22, 4);
	require(!acceptsWordDice(contrasted, model));
	contrasted = signal;
	word(contrasted, 22, 6);
	require(!acceptsWordDice(contrasted, model));
	contrasted = signal;
	word(contrasted, 1, 2);
	require(!acceptsWordDice(contrasted, model));

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
}
