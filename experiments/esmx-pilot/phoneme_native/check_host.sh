#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    printf 'Usage: %s TRAINING_PYTHON CHECKPOINT_DIR [SYNTHETIC_FEATURE_CACHE]\n' "$0" >&2
    exit 2
fi

here="$(cd "$(dirname "$0")" && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf -- "$scratch"' EXIT
metadata="$("$1" -B "$here/export_bank.py" "$2" "$scratch/student_bank.h")"
printf '%s\n' "$metadata" >&2
width="$("$1" -c 'import json, sys; print(json.load(sys.stdin)["width"])' <<< "$metadata")"
"${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Werror -ffp-contract=off \
    -D "BB_ESMX_PHONEME_WIDTH=$width" \
    -I"$scratch" -I"$here" "$here/Kernel.cpp" "$here/main.cpp" -o "$scratch/student"
extra=()
if [[ $# == 3 ]]; then
    extra=(--cache "$3")
fi
"$1" -B "$here/compare.py" "$scratch/student" "$2" "${extra[@]}"
