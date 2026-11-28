#!/usr/bin/env bash

# Sourced by the two existing host build commands after validating their arguments.
clock_sources=()
clock_libraries=()
if [[ ${BB_ESMX_CLOCK_ACOUSTIC:-0} == 1 ]]; then
    if [[ ${BB_ESMX_JOKE_FALLBACK:-0} != 1 ]]; then
        printf 'BB_ESMX_CLOCK_ACOUSTIC=1 requires BB_ESMX_JOKE_FALLBACK=1.\n' >&2
        exit 2
    fi
    if [[ ! -f "${BB_ESMX_SPHINX_INCLUDE:-}/pocketsphinx.h" ||
          ! -f "${BB_ESMX_SPHINX_GENERATED_INCLUDE:-}/pocketsphinx/sphinx_config.h" ||
          ! -f "${BB_ESMX_SPHINX_LIBRARY:-}" ]]; then
        printf 'Provide the source and generated include directories and the local PocketSphinx static library.\n' >&2
        exit 2
    fi
    flags+=(-DBB_ESMX_CLOCK_ACOUSTIC -I "$BB_ESMX_SPHINX_INCLUDE" -I "$BB_ESMX_SPHINX_GENERATED_INCLUDE")
    clock_sources+=(main/src/Audio/EsMxPilot/ClockAcoustic.cpp)
    if [[ ${BB_ESMX_DICE_ACOUSTIC:-0} == 1 ]]; then
        if [[ ! -f "${BB_ESMX_CLOCK_ACOUSTIC_DIR:-}/DiceAcousticAssets.h" ||
              ! -f "${BB_ESMX_CLOCK_ACOUSTIC_DIR:-}/dice.kws" ||
              ! -f "${BB_ESMX_CLOCK_ACOUSTIC_DIR:-}/esmx-dice.dict" ]]; then
            printf 'Provide the local D6 acoustic assets.\n' >&2
            exit 2
        fi
        flags+=(-DBB_ESMX_DICE_ACOUSTIC -I "$BB_ESMX_CLOCK_ACOUSTIC_DIR")
        clock_sources+=(main/src/Audio/EsMxPilot/DiceAcoustic.cpp)
    fi
    clock_libraries+=("$BB_ESMX_SPHINX_LIBRARY" -lm)
fi
