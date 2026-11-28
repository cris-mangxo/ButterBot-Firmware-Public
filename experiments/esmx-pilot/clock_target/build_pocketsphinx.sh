#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 || ( $3 != host && $3 != esp32s3 ) ||
      ( $# -ge 4 && $4 != --oppose-pool ) ||
      ( $# == 5 && $5 != --skip-empty-history && ( $5 != --fsg-breakdown || $3 != esp32s3 ) ) ]]; then
    printf 'Usage: %s POCKETSPHINX_GIT LOCAL_OUTPUT_DIR host|esp32s3 [--oppose-pool [--fsg-breakdown|--skip-empty-history]]\n' "$0" >&2
    exit 2
fi

source_pin=a9ec4b881de75771d8e6b8ded2823266589cbd5d
patch="$(cd -- "$(dirname -- "$0")" && pwd)/pocketsphinx-5.0.4-xtensa.patch"
mkdir -- "$2"
output="$(cd -- "$2" && pwd)"
mkdir -- "$output/source"
vendor_git="$(cd -- "$1" && pwd)"
git -C "$vendor_git" archive "$source_pin" | tar -x -C "$output/source"
flags=(-G Ninja -DBUILD_TESTING=OFF)
if [[ $3 == esp32s3 ]]; then
    compiler="$(command -v xtensa-esp32s3-elf-gcc)"
    (cd -- "$output/source" && git apply --unidiff-zero --ignore-space-change --check "$patch" &&
        git apply --unidiff-zero --ignore-space-change "$patch")
    flags+=(-DCMAKE_BUILD_TYPE= -DCMAKE_SYSTEM_NAME=Generic -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY
            "-DCMAKE_C_COMPILER=$compiler" -DHAVE_POPEN=0
            "-DMATH_LIBRARY=$("$compiler" -print-file-name=libm.a)"
            "-DCMAKE_C_FLAGS=-Os -mlongcalls -ffunction-sections -fdata-sections -DBB_CLOCK_FSG_ONLY")
else
    flags+=(-DCMAKE_BUILD_TYPE=Release)
fi
# The laboratory patches use exact offsets from the pinned vendor source.
# Verify each patched file against its Git blob before applying the patch.
check_pinned_file() {
    local relative="$1" expected actual
    expected="$(git -C "$vendor_git" rev-parse "$source_pin:$relative")"
    actual="$(git -C "$vendor_git" hash-object "$output/source/$relative")"
    if [[ $expected != "$actual" ]]; then
        printf 'Pinned PocketSphinx source does not match: %s\n' "$relative" >&2
        exit 2
    fi
}

if [[ ${4:-} == --oppose-pool ]]; then
    check_pinned_file src/fsg_search.c
    check_pinned_file src/fsg_search_internal.h
    pool_patch="$(dirname "$patch")/pocketsphinx-5.0.4-fsg-active-pool.patch"
    (cd -- "$output/source" && git apply --unidiff-zero --check "$pool_patch" && git apply --unidiff-zero "$pool_patch")
fi
case ${5:-} in
    --fsg-breakdown) variant_patch=pocketsphinx-5.0.4-fsg-phases.patch ;;
    --skip-empty-history)
        check_pinned_file src/fsg_history.c
        variant_patch=pocketsphinx-5.0.4-fsg-skip-empty-history.patch
        ;;
    *) variant_patch= ;;
esac
if [[ -n $variant_patch ]]; then
    variant_patch="$(dirname "$patch")/$variant_patch"
    (cd -- "$output/source" && git apply --unidiff-zero --check "$variant_patch" && git apply --unidiff-zero "$variant_patch")
fi
cmake -S "$output/source" -B "$output/build" "${flags[@]}"
cmake --build "$output/build" --target libpocketsphinx.a
