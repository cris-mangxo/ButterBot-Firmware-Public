get_filename_component(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(PILOT "${ROOT}/main/src/Audio/EsMxPilot")
set(KWS "${ROOT}/main/src/Audio/EsMxKws")
set(BANK "${ROOT}/experiments/esmx-compactas")
set(PHONEME "${ROOT}/experiments/esmx-pilot/phoneme_native")

idf_component_register(
    SRCS "main.cpp"
        "${PILOT}/ClockAcoustic.cpp" "${PILOT}/ClockModelVfs.cpp"
        "${PILOT}/Recognition.cpp" "${PILOT}/JokeFallback.cpp"
        "${KWS}/Compact.cpp" "${KWS}/logmel.cpp"
        "${BANK}/guard/PhraseGate.cpp" "${BANK}/autonomous/WordDice.cpp"
        "${BANK}/autonomous/AcousticDice.cpp"
        "${PHONEME}/Kernel.cpp" "${PHONEME}/Alignment.cpp"
        "${PHONEME}/JokeGuard.cpp" "${PHONEME}/ClockGuard.cpp"
    REQUIRES esp_timer vfs mbedtls
    INCLUDE_DIRS "${ROOT}/main/src" "${KWS}" "${PHONEME}" "${BANK}/guard" "${BANK}/autonomous"
        "${BB_ESMX_ASSETS}" "${BB_ESMX_SPHINX_INCLUDE}" "${BB_ESMX_SPHINX_GENERATED_INCLUDE}"
        "${BB_ESMX_CLOCK_PACK}" "${BB_ESMX_CLOCK_TARGET_CASES}")

add_prebuilt_library(bb_esmx_target_sphinx "${BB_ESMX_SPHINX_LIBRARY}")
target_link_libraries(${COMPONENT_LIB} PRIVATE bb_esmx_target_sphinx)
target_link_options(${COMPONENT_LIB} INTERFACE "-Wl,-u,__wrap_longjmp")
target_compile_definitions(${COMPONENT_LIB} PRIVATE
                           BB_ESMX_PILOT BB_ESMX_JOKE_FALLBACK BB_ESMX_CLOCK_FALLBACK BB_ESMX_CLOCK_ACOUSTIC)
if(BB_ESMX_DICE_ACOUSTIC)
    target_sources(${COMPONENT_LIB} PRIVATE "${PILOT}/DiceAcoustic.cpp")
    target_compile_definitions(${COMPONENT_LIB} PRIVATE BB_ESMX_DICE_ACOUSTIC)
endif()
target_compile_options(${COMPONENT_LIB} PRIVATE -ffp-contract=off)
