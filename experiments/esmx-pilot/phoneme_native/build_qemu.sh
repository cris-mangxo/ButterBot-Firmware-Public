#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 || ! -d "$3" ]]; then
    printf 'Usage: %s TRAINING_PYTHON CHECKPOINT_DIR EMPTY_OUTPUT_DIR [SYNTHETIC_FEATURE_CACHE]\n' "$0" >&2
    exit 2
fi

here="$(cd "$(dirname "$0")" && pwd)"
out="$(cd "$3" && pwd)"
assets="$out/assets"
mkdir "$assets"
metadata="$("$1" -B "$here/export_bank.py" "$2" "$assets/student_bank.h")"
printf '%s\n' "$metadata"
width="$("$1" -c 'import json, sys; print(json.load(sys.stdin)["width"])' <<< "$metadata")"
extra=()
if [[ $# == 4 ]]; then
    extra=(--cache "$4")
fi
"$1" -B "$here/qemu_fixtures.py" "$2" "$assets/student_tests.h" "${extra[@]}"
IDF_COMPONENT_MANAGER=0 idf.py -C "$here/qemu" -B "$out/build" \
    -D "SDKCONFIG=$out/sdkconfig" -D IDF_TARGET=esp32s3 -D "PHONEME_ASSETS=$assets" \
    -D "PHONEME_WIDTH=$width" build
