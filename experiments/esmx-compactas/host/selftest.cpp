#include "Audio/EsMxKws/Compact.h"
#include "Audio/EsMxKws/ScenarioForIntent.h"

#include <cassert>

using namespace EsMx::Kws;

int main(){
	Signature query{};
	std::array<Prototype, static_cast<size_t>(Intent::Count)> model{};
	for(size_t i = 0; i < model.size(); ++i){
		model[i].intent = static_cast<Intent>(i);
		for(auto& frame : model[i].signature.frames){
			frame.fill(static_cast<int8_t>(20 * i));
		}
	}
	query = model[6].signature;
	const auto correct = classify(query, model, {6200, 600});
	assert(correct.intent == Intent::Dice6 && correct.cost == 0);
	const auto route = scenarioForIntent(correct.intent);
	assert(route && route->first == BB::Action::Scenario::DiceRoll);
	assert(route->second.raw == DiceScenarioData(DiceScenarioData::DiceType::D6).raw);
	assert(!scenarioForIntent(Intent::Unknown));

	assert(classify(query, std::span(model).first(6), {6200, 600}).intent == Intent::Unknown);
	assert(classify(query, model, {6200, 10000}).intent == Intent::Unknown);
	auto malformed = model;
	malformed[0].intent = Intent::Count;
	assert(classify(query, malformed, {6200, 600}).intent == Intent::Unknown);

	auto tied = model;
	tied[5].signature = query;
	assert(classify(query, tied, {6200, 0}).intent == Intent::Unknown);

	for(auto& frame : query.frames){
		frame.fill(-128);
	}
	assert(classify(query, model, {0, 600}).intent == Intent::Unknown);
	assert(classify(model[0].signature, model, {6200, 600}).intent == Intent::Unknown);

	bb::asr::LogMel frontend{};
	std::array<int16_t, 16000> silence{};
	assert(!extract(frontend, silence, query));
	assert(bb::asr::logmel_init(&frontend));
	assert(!extract(frontend, silence, query));
	assert(!extract(frontend, std::span(silence).first(479), query));
	std::array<int16_t, MaxSamples + 1> tooLong{};
	assert(!extract(frontend, tooLong, query));
}
