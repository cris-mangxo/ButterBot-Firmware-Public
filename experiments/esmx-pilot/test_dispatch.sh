#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf -- "$scratch"' EXIT
mkdir -p -- "$scratch/include/Audio" "$scratch/include/freertos"

# Thin declarations for ESP-only types unused by the dispatcher.
printf '#pragma once\n' > "$scratch/include/freertos/FreeRTOS.h"
printf '#pragma once\n' > "$scratch/include/freertos/task.h"
printf '#pragma once\nclass EventBag { public: struct EventData {}; };\n' > "$scratch/include/EventBag.h"
printf '#pragma once\nclass AudioFrontend { public: struct Phrase {}; struct PhraseResult {}; };\n' > "$scratch/include/Audio/AudioFrontend.h"

"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror -Wno-unused-parameter -DBB_ESMX_PILOT \
	-I "$scratch/include" -I "$repo/experiments/esmx-commands/host" \
	-I "$repo/main/src" -I "$repo/components/ButterBot-Common/src" -I "$repo/components/CMF/src" \
	"$repo/experiments/esmx-pilot/test_dispatch.cpp" "$repo/main/src/Audio/EsMxPilot/Dispatch.cpp" \
	-o "$scratch/test_dispatch"
"$scratch/test_dispatch"
