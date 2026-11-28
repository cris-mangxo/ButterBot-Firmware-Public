#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 || ( $3 != host && $3 != esp32s3 ) ]]; then
    printf 'Usage: %s POCKETSPHINX_GIT LOCAL_OUTPUT_DIR host|esp32s3\n' "$0" >&2
    exit 2
fi

source_pin=a9ec4b881de75771d8e6b8ded2823266589cbd5d
patch="$(cd -- "$(dirname -- "$0")" && pwd)/pocketsphinx-5.0.4-xtensa.patch"
mkdir -- "$2"
output="$(cd -- "$2" && pwd)"
mkdir -- "$output/source"
git -C "$1" archive "$source_pin" | tar -x -C "$output/source"
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
cmake -S "$output/source" -B "$output/build" "${flags[@]}"
cmake --build "$output/build" --target libpocketsphinx.a
