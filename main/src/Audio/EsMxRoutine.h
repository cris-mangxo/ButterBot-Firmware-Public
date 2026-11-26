#pragma once

#include <string_view>
#include "States/ScenarioRoutineMappings.h"

namespace EsMx {
// Resolves a factory through the original dispatcher. The caller owns the
// listening state and may schedule the factory through ScenarioRoutineService.
RoutineFactory routineForTranscript(std::string_view transcript);
}
