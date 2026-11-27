#pragma once

#include "Audio/EsMxKws/Compact.h"
#include <string_view>

// Records the real dispatcher's chosen scenario and verifies the factory pointer.
// Routine implementations are replaced by host doubles and never actuated.
std::string_view probeDispatch(EsMx::Kws::Intent intent);
