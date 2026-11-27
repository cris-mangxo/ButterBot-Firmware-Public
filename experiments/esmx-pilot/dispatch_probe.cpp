#include "dispatch_probe.h"

#include "Audio/EsMxPilot/Dispatch.h"
#include "Routines/DiceRollRoutine.h"
#include "States/ScenarioRoutineMappings.h"
#include <optional>

namespace {
using Scenario = BB::Action::Scenario;
std::optional<std::pair<Scenario, ScenarioData>> dispatched;

template<Scenario scenario>
class ScenarioProbe : public Routine {
public:
    using Routine::Routine;
};
}

RoutineFactory routineForScenario(const std::pair<Scenario, ScenarioData>& scenario){
    dispatched = scenario;
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

std::string_view probeDispatch(EsMx::Kws::Intent intent){
    dispatched.reset();
    const auto factory = EsMx::Pilot::routineForIntent(intent);
    if(dispatched && dispatched->second.raw == 0){
        if(dispatched->first == Scenario::PassTheButter && factory == &makeRoutine<ScenarioProbe<Scenario::PassTheButter>>){
            return "butter";
        }
        if(dispatched->first == Scenario::Joke && factory == &makeRoutine<ScenarioProbe<Scenario::Joke>>){
            return "joke";
        }
        if(dispatched->first == Scenario::CurrentTime && factory == &makeRoutine<ScenarioProbe<Scenario::CurrentTime>>){
            return "time";
        }
    }
    if(!dispatched && factory == &makeRoutine<DiceRollEsMxD6Routine>){
        return "dice6_factory";
    }
    if(!dispatched && !factory){
        return "none";
    }
    return "unexpected";
}
