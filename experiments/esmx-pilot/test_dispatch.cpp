#include "Audio/EsMxPilot/Dispatch.h"
#include "Routines/DiceRollRoutine.h"
#include "States/ScenarioRoutineMappings.h"

#include <array>
#include <cassert>
#include <cstdio>
#include <tuple>

namespace {
using Intent = EsMx::Kws::Intent;
using Scenario = BB::Action::Scenario;

unsigned calls = 0;
std::pair<Scenario, ScenarioData> lastScenario{};

template<Scenario scenario>
class ScenarioProbe : public Routine {
public:
	using Routine::Routine;
};
}

// Host seam: the ESP-IDF build provides the original table and D6 tick.
RoutineFactory routineForScenario(const std::pair<Scenario, ScenarioData>& scenario){
	++calls;
	lastScenario = scenario;
	switch(scenario.first){
		case Scenario::PassTheButter: return &makeRoutine<ScenarioProbe<Scenario::PassTheButter>>;
		case Scenario::Joke: return &makeRoutine<ScenarioProbe<Scenario::Joke>>;
		case Scenario::CurrentTime: return &makeRoutine<ScenarioProbe<Scenario::CurrentTime>>;
		default: return nullptr;
	}
}

Routine::TickingState DiceRollEsMxD6Routine::tick(float){
	return TickingState::Done;
}

int main(){
	const std::array cases{
		std::tuple{Intent::Butter, Scenario::PassTheButter, &makeRoutine<ScenarioProbe<Scenario::PassTheButter>>},
		std::tuple{Intent::Joke, Scenario::Joke, &makeRoutine<ScenarioProbe<Scenario::Joke>>},
		std::tuple{Intent::Time, Scenario::CurrentTime, &makeRoutine<ScenarioProbe<Scenario::CurrentTime>>},
	};
	for(const auto& [intent, expected, factory] : cases){
		const auto before = calls;
		assert(EsMx::Pilot::routineForIntent(intent) == factory);
		assert(calls == before + 1);
		assert(lastScenario.first == expected);
		assert(lastScenario.second.raw == 0);
	}

	const auto before = calls;
	assert(EsMx::Pilot::routineForIntent(Intent::Dice6) == &makeRoutine<DiceRollEsMxD6Routine>);
	for(const Intent rejected : {Intent::Unknown, Intent::YouButter, Intent::Fact, Intent::Count,
	                             static_cast<Intent>(255)}){
		assert(EsMx::Pilot::routineForIntent(rejected) == nullptr);
	}
	assert(calls == before);

	std::puts("DISPATCH_HOST_OK stock_scenario_keys=3 d6_factory=1 rejected=5");
}
