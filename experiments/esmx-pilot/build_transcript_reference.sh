#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
    printf 'Usage: %s OUTPUT_EXECUTABLE\n' "$0" >&2
    exit 2
fi

output="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Werror -Wno-unused-parameter -DBB_ESMX_PILOT \
    -I experiments/esmx-pilot/dispatch_stubs -I experiments/esmx-pilot/capture_stubs \
    -I experiments/esmx-commands/host -I experiments/esmx-compactas/transcript \
    -I main/src -I components/ButterBot-Common/src -I components/CMF/src \
    experiments/esmx-pilot/transcript_reference.cpp experiments/esmx-pilot/dispatch_probe.cpp \
    experiments/esmx-compactas/transcript/DiceGrammar.cpp main/src/Audio/EsMxCommand.cpp \
    main/src/Audio/EsMxPilot/Dispatch.cpp -o "$output"
