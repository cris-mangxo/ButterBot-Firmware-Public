#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 || ! -f "$1/compact_voice_bank.h" ]]; then
    printf 'Usage: %s COMPACT_VOICE_ASSETS\n' "$0" >&2
    exit 2
fi

assets="$(cd -- "$1" && pwd)"
here="$(cd -- "$(dirname -- "$0")" && pwd)"
repo="$(cd -- "$here/../../.." && pwd)"
cd "$repo"
scratch="$(mktemp -d)"
trap 'rm -rf -- "$scratch"' EXIT

headers=(Audio/AudioFrontend.h Audio/SpeechAudioGen.h Core/Application.h Statics/ApplicationStatics.h
         Services/Audio/Audio.h Services/Com.h Services/Settings.h)
for header in "${headers[@]}"; do
    shim="$scratch/include/$header"
    mkdir -p -- "$(dirname -- "$shim")"
    printf '#pragma once\n#include "FlowStubs.h"\n' > "$shim"
done

"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror -Wno-sign-compare -O2 \
    -I "$here/stubs" -I components/ButterBot-Common/src \
    -c components/ButterBot-Common/src/Phrases.cpp -o "$scratch/phrases.o"

"${CXX:-c++}" -std=c++20 -Wall -Wextra -Werror -O2 -pthread -DBB_ESMX_COMPACT_VOICE \
    -I "$scratch/include" -I "$here/stubs" -I experiments/esmx-pilot/capture_stubs -I "$assets" \
    -I main/src -I components/CMF/src -I components/ButterBot-Common/src \
    main/src/Audio/CompactSpeech.cpp main/src/Audio/SpeechGenCompact.cpp \
    main/src/Audio/SpeechAudioSource.cpp "$scratch/phrases.o" \
    -Wno-unused-parameter -Wno-missing-field-initializers \
    main/src/Routines/SettingsActivityRoutine.cpp "$here/test_activity.cpp" "$here/test.cpp" -o "$scratch/test"
"$scratch/test"
