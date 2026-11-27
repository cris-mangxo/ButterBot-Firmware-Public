#include "Dispatch.h"

#ifdef BB_ESMX_PILOT

#include "Audio/EsMxKws/ScenarioForIntent.h"
#include "Routines/DiceRollRoutine.h"
#include "States/ScenarioRoutineMappings.h"

namespace EsMx::Pilot {
RoutineFactory routineForIntent(Kws::Intent intent){
	switch(intent){
		case Kws::Intent::Butter:
		case Kws::Intent::Joke:
		case Kws::Intent::Time: {
			const auto scenario = Kws::scenarioForIntent(intent);
			return scenario ? routineForScenario(*scenario) : nullptr;
		}
		case Kws::Intent::Dice6:
			return &makeRoutine<DiceRollEsMxD6Routine>;
		default:
			return nullptr;
	}
}
}

#endif
