#pragma once

#include "Compact.h"
#include "BBData.h"
#include <optional>
#include <utility>
#include <Scenarios.h>

namespace EsMx::Kws {
inline std::optional<std::pair<BB::Action::Scenario, ScenarioData>> scenarioForIntent(Intent intent){
	using Action = BB::Action::Scenario;
	switch(intent){
		case Intent::Butter: return {{Action::PassTheButter, {}}};
		case Intent::YouButter: return {{Action::YouPassButter, {}}};
		case Intent::Joke: return {{Action::Joke, {}}};
		case Intent::Fact: return {{Action::Fact, {}}};
		case Intent::Time: return {{Action::CurrentTime, {}}};
		case Intent::Dice6: return {{Action::DiceRoll, DiceScenarioData{DiceScenarioData::DiceType::D6}}};
		default: return std::nullopt;
	}
}
}
