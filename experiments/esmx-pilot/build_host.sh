#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 || ( $# -eq 3 && $3 != --capture ) ]]; then
    printf 'Usage: %s GENERATED_ASSETS_DIR OUTPUT_EXECUTABLE [--capture]\n' "$0" >&2
    exit 2
fi

assets="$(cd "$1" && pwd)"
output="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

kws=main/src/Audio/EsMxKws
experiments=experiments/esmx-compactas
entry=(experiments/esmx-pilot/recognize.cpp)
flags=(-std=c++20 -O2 -Wall -Wextra -Werror -DBB_ESMX_PILOT)
if [[ ${3:-} == --capture ]]; then
    entry=(experiments/esmx-pilot/capture_replay.cpp "$kws/PilotCapture.cpp")
    flags+=(-pthread -I experiments/esmx-pilot/capture_stubs)
fi
${CXX:-c++} "${flags[@]}" \
    -I main/src -I "$kws" -I "$experiments/guard" -I "$experiments/autonomous" -I "$assets" \
    "${entry[@]}" main/src/Audio/EsMxPilot/Recognition.cpp \
    "$kws/Compact.cpp" "$kws/logmel.cpp" "$experiments/guard/PhraseGate.cpp" \
    "$experiments/autonomous/WordDice.cpp" "$experiments/autonomous/AcousticDice.cpp" -o "$output"
