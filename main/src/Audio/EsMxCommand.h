#pragma once

#include <optional>
#include <string_view>
#include <utility>
#include "Scenarios.h"

namespace EsMx {
struct Command {
	std::string_view name;
	std::pair<BB::Action::Scenario, ScenarioData> scenario;
};

// One complete transcript from a listening turn. Unknown or ambiguous text yields no command.
std::optional<Command> matchCommand(std::string_view transcript);
}
