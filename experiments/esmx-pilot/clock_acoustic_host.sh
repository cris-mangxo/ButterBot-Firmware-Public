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
    clock_libraries+=("$BB_ESMX_SPHINX_LIBRARY" -lm)
fi
