#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 || ( $# -eq 3 && $3 != --capture && $3 != --dice-stages ) ]]; then
    printf 'Usage: %s GENERATED_ASSETS_DIR OUTPUT_EXECUTABLE [--capture|--dice-stages]\nSet BB_ESMX_JOKE_FALLBACK=1 to compile the experimental Chiste filter.\n' "$0" >&2
    exit 2
fi

assets="$(cd "$1" && pwd)"
output="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

kws=main/src/Audio/EsMxKws
experiments=experiments/esmx-compactas
entry=(experiments/esmx-pilot/recognize.cpp)
recognition=(main/src/Audio/EsMxPilot/Recognition.cpp)
flags=(-std=c++20 -O2 -Wall -Wextra -Werror -DBB_ESMX_PILOT)
joke=()
source experiments/esmx-pilot/clock_acoustic_host.sh
if [[ ${BB_ESMX_CLOCK_FALLBACK:-0} == 1 && ${BB_ESMX_JOKE_FALLBACK:-0} != 1 ]]; then
    printf 'BB_ESMX_CLOCK_FALLBACK=1 requires BB_ESMX_JOKE_FALLBACK=1.\n' >&2
    exit 2
fi
if [[ ${BB_ESMX_JOKE_FALLBACK:-0} == 1 ]]; then
    native=experiments/esmx-pilot/phoneme_native
    for header in student_bank.h joke_rule.h; do
        if [[ ! -f "$assets/$header" ]]; then
            printf 'Missing generated Chiste header: %s\n' "$header" >&2
            exit 2
        fi
    done
    flags+=(-DBB_ESMX_JOKE_FALLBACK -I "$native")
    joke_sources=(main/src/Audio/EsMxPilot/JokeFallback.cpp "$native/Kernel.cpp"
                  "$native/Alignment.cpp" "$native/JokeGuard.cpp")
    if [[ ${BB_ESMX_CLOCK_FALLBACK:-0} == 1 ]]; then
        if [[ ! -f "$assets/clock_rule.h" ]]; then
            printf 'Missing generated Hora header: clock_rule.h\n' >&2
            exit 2
        fi
        flags+=(-DBB_ESMX_CLOCK_FALLBACK)
        joke_sources+=("$native/ClockGuard.cpp")
    fi
fi
if [[ ${3:-} == --capture || ${BB_ESMX_JOKE_FALLBACK:-0} == 1 ]]; then
    scratch="$(mktemp -d)"
    trap 'rm -rf -- "$scratch"' EXIT
fi
if [[ ${BB_ESMX_JOKE_FALLBACK:-0} == 1 ]]; then
    for source in "${joke_sources[@]}"; do
        object="$scratch/$(basename "${source%.cpp}").o"
        "${CXX:-c++}" "${flags[@]}" -ffp-contract=off \
            -I main/src -I "$kws" -I "$experiments/autonomous" -I "$assets" \
            -c "$source" -o "$object"
        joke+=("$object")
    done
fi
if [[ ${3:-} == --capture ]]; then
    entry=(experiments/esmx-pilot/capture_replay.cpp "$kws/PilotCapture.cpp")
    entry+=(experiments/esmx-pilot/dispatch_probe.cpp main/src/Audio/EsMxPilot/Dispatch.cpp)
    flags+=(-pthread -Wno-unused-parameter -I experiments/esmx-pilot/capture_stubs -I experiments/esmx-pilot/dispatch_stubs
            -I experiments/esmx-commands/host -I components/ButterBot-Common/src -I components/CMF/src)
    "${CXX:-c++}" "${flags[@]}" -Drecognize=recognizeKernel \
        -I main/src -I "$kws" -I "$experiments/guard" -I "$experiments/autonomous" -I "$assets" \
        -c "${recognition[0]}" -o "$scratch/recognition.o"
    recognition=(experiments/esmx-pilot/capture_pcm_probe.cpp "$scratch/recognition.o")
elif [[ ${3:-} == --dice-stages ]]; then
    entry=(experiments/esmx-pilot/dice_stage_probe.cpp)
fi
if [[ ${BB_ESMX_JOKE_FALLBACK:-0} == 1 ]]; then
    recognition+=("${joke[@]}")
fi
${CXX:-c++} "${flags[@]}" \
    -I main/src -I "$kws" -I "$experiments/guard" -I "$experiments/autonomous" -I "$assets" \
    "${entry[@]}" "${recognition[@]}" \
    "$kws/Compact.cpp" "$kws/logmel.cpp" "$experiments/guard/PhraseGate.cpp" \
    "$experiments/autonomous/WordDice.cpp" "$experiments/autonomous/AcousticDice.cpp" \
    ${clock_sources[@]+"${clock_sources[@]}"} ${clock_libraries[@]+"${clock_libraries[@]}"} -o "$output"
