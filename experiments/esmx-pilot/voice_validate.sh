#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 || ! -f "$1/voice_data.h" ]]; then
    echo "Usage: $0 /path/to/generated-dir" >&2
    exit 2
fi

root="$(cd "$(dirname "$0")/../.." && pwd)"
helix="$root/components/CMF/lib/libhelix-aac/src"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

objects=()
for file in "$helix/"*.c; do
    object="$tmp/${file##*/}.o"
    "${CC:-cc}" -DESP_PLATFORM -O2 -I"$helix" -c "$file" -o "$object"
    objects+=("$object")
done

"${CXX:-c++}" -std=c++20 -DESP_PLATFORM -O2 -I"$helix" -I"$1" -I"$root/main/src" \
    "$root/experiments/esmx-pilot/voice_validate.cpp" "${objects[@]}" -o "$tmp/voice-test"
"$tmp/voice-test"
