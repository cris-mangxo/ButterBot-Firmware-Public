#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || ( $# -eq 2 && $2 != --test-gate ) || ${BB_ESMX_BUTTER_ACOUSTIC:-0} != 1 ]]; then
    printf 'Usage: %s OUTPUT_EXECUTABLE [--test-gate] (requires BB_ESMX_BUTTER_ACOUSTIC=1)\n' "$0" >&2
    exit 2
fi
output="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$root"
flags=(-std=c++20 -Wall -Wextra -Werror -O2 -I main/src)
source experiments/esmx-pilot/clock_acoustic_host.sh
entry=probe.cpp
if [[ ${2:-} == --test-gate ]]; then
    entry=test_gate.cpp
fi
"${CXX:-c++}" "${flags[@]}" "experiments/esmx-pilot/butter_acoustic/$entry" \
    "${clock_sources[@]}" "${clock_libraries[@]}" -o "$output"
