#!/usr/bin/env bash
set -euo pipefail

if [[ ( $# != 1 && $# != 3 ) || ! -f "$1/voice_data.h" ]]; then
    printf 'Usage: %s GENERATED_VOICE_DIR [GENERATED_MODEL_DIR EXPECTED_INTENT < pcm.s16le]\n' "$0" >&2
    exit 2
fi

voice="$(cd -- "$1" && pwd)"
if [[ $# == 3 ]]; then
    models="$(cd -- "$2" && pwd)"
    expected="$3"
fi
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
scratch="$(mktemp -d)"
trap 'rm -rf -- "$scratch"' EXIT
mkdir -- "$scratch/generated"
ln -s -- "$voice/voice_data.h" "$scratch/generated/voice_data.h"

# Use the genuine routine/voice translation units with host substitutes for hardware headers.
fixture="experiments/esmx-pilot/routine_host"
headers=(
    Audio/AudioFrontend.h Audio/SpeechAudioGen.h Audio/SpeechAudioSource.h Audio/SpeechGen.h
    Core/Application.h Services/Audio/AACAudioGenerator.h Services/Audio/Audio.h
    Services/Audio/FileAudioSource.h Services/Com.h Services/Time.h
    Statics/ApplicationStatics.h Util/ServiceLocator.h esp_random.h
)
for header in "${headers[@]}"; do
    shim="$scratch/include/$header"
    mkdir -p -- "$(dirname -- "$shim")"
    printf '#pragma once\n#include "stubs.h"\n' > "$shim"
done
mkdir -p -- "$scratch/include/Log"
cat > "$scratch/include/Log/Log.h" <<'LOG_HEADER'
#pragma once
namespace RoutineHost {
template<class... Args>
void ignoreLog(Args&&...) {}
}
#define DEFINE_LOG(...)
#define CMF_LOG(owner, level, ...) ::RoutineHost::ignoreLog(__VA_ARGS__)
LOG_HEADER
printf '#pragma once\n#include <cstdio>\n#define esp_rom_printf std::printf\n' > "$scratch/include/esp_rom_sys.h"

flags=(-std=c++20 -O2 -Wall -Wextra -Werror -Wno-unused-parameter -Wno-sign-compare
       -pthread -DBB_ESMX_PILOT -I "$scratch/include" -I "$fixture" -I "$scratch/generated"
       -I experiments/esmx-pilot/capture_stubs -I experiments/esmx-pilot/dispatch_stubs
       -I main/src -I components/ButterBot-Common/src -I components/CMF/src)
sources=("$fixture/main.cpp" main/src/Routines/PassTheButterRoutine.cpp main/src/Routines/JokeRoutine.cpp
         main/src/Routines/CurrentTimeRoutine.cpp main/src/Routines/DiceRollRoutine.cpp
         main/src/Audio/EsMxPilot/Dispatch.cpp main/src/Audio/EsMxPilot/Voice.cpp
         main/src/Audio/MemoryAudioSource.cpp components/ButterBot-Common/src/Phrases.cpp)
if [[ $# == 3 ]]; then
    kws=main/src/Audio/EsMxKws
    experiments=experiments/esmx-compactas
    model_headers=(pilot_model.h phrase_bank.h word_bank.h word_extra.h word_adapted.h
                   acoustic_dice.h consensus_acoustic_dice.h backup_acoustic_dice.h)
    for header in "${model_headers[@]}"; do
        if [[ ! -f "$models/$header" ]]; then
            printf 'Missing generated model header: %s\n' "$header" >&2
            exit 2
        fi
        ln -s -- "$models/$header" "$scratch/generated/$header"
    done
    flags+=(-DROUTINE_HOST_RECOGNITION -I "$kws" -I "$experiments/guard" -I "$experiments/autonomous")
    sources+=(main/src/Audio/EsMxPilot/Recognition.cpp "$kws/Compact.cpp" "$kws/logmel.cpp"
              "$experiments/guard/PhraseGate.cpp" "$experiments/autonomous/WordDice.cpp"
              "$experiments/autonomous/AcousticDice.cpp")
fi

"${CXX:-c++}" "${flags[@]}" "${sources[@]}" -o "$scratch/routine_host"

if [[ $# == 3 ]]; then
    "$scratch/routine_host" "$expected"
else
    "$scratch/routine_host"
fi
