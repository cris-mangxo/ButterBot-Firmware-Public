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
target_compile_options(${COMPONENT_LIB} PRIVATE -ffp-contract=off)

option(BB_ESMX_BUTTER_ACOUSTIC "Include the experimental Mantequilla validation cases" OFF)
option(BB_ESMX_BUTTER_WEAK_CONFLICT "Allow the experimental weak D6 conflict for Mantequilla" OFF)
if(BB_ESMX_BUTTER_ACOUSTIC)
    if(NOT IS_ABSOLUTE "${BB_ESMX_BUTTER_ASSETS}" OR
       NOT EXISTS "${BB_ESMX_BUTTER_ASSETS}/ButterModelAssets.h")
        message(FATAL_ERROR "Set BB_ESMX_BUTTER_ASSETS to the local Butter validation assets")
    endif()
    target_compile_definitions(${COMPONENT_LIB} PRIVATE BB_ESMX_BUTTER_ACOUSTIC)
    target_include_directories(${COMPONENT_LIB} PRIVATE "${BB_ESMX_BUTTER_ASSETS}")
    target_sources(${COMPONENT_LIB} PRIVATE "${PILOT}/ButterAcoustic.cpp")
    if(BB_ESMX_BUTTER_WEAK_CONFLICT)
        target_compile_definitions(${COMPONENT_LIB} PRIVATE BB_ESMX_BUTTER_WEAK_CONFLICT)
    endif()
elseif(BB_ESMX_BUTTER_WEAK_CONFLICT)
    message(FATAL_ERROR "BB_ESMX_BUTTER_WEAK_CONFLICT requires BB_ESMX_BUTTER_ACOUSTIC")
endif()
