#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    printf 'Usage: %s GENERATED_ASSETS_DIR OUTPUT_EXECUTABLE\n' "$0" >&2
    exit 2
fi

assets="$(cd "$1" && pwd)"
output="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

kws=main/src/Audio/EsMxKws
experiments=experiments/esmx-compactas
${CXX:-c++} -std=c++20 -O2 -Wall -Wextra -Werror -DBB_ESMX_PILOT \
    -I main/src -I "$kws" -I "$experiments/guard" -I "$experiments/autonomous" -I "$assets" \
    experiments/esmx-pilot/recognize.cpp main/src/Audio/EsMxPilot/Recognition.cpp \
    "$kws/Compact.cpp" "$kws/logmel.cpp" "$experiments/guard/PhraseGate.cpp" \
    "$experiments/autonomous/WordDice.cpp" "$experiments/autonomous/AcousticDice.cpp" -o "$output"
