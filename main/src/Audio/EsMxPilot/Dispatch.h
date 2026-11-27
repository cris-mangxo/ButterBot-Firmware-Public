#pragma once

#ifdef BB_ESMX_PILOT

#include "Audio/EsMxKws/Compact.h"
#include "States/StateRoutineFactory.h"

namespace EsMx::Pilot {
RoutineFactory routineForIntent(Kws::Intent intent);
}

#endif
