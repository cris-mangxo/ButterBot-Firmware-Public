file(GLOB_RECURSE LIBS "lib/*/src/**.cpp" "lib/*/src/**.c")

file(GLOB_RECURSE SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/**.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/**.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/**.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/**.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*/**.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*/**.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*/**.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*/**.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

idf_component_register(SRCS ${SOURCES} ${LIBS} INCLUDE_DIRS "src")

option(BB_ESMX_PILOT "Build the experimental es-MX speech pilot" OFF)
option(BB_ESMX_JOKE_FALLBACK "Build the experimental es-MX Chiste fallback" OFF)
option(BB_ESMX_CLOCK_FALLBACK "Build the experimental es-MX Hora fallback" OFF)
option(BB_ESMX_CLOCK_ACOUSTIC "Include the locally supplied acoustic Hora fallback" OFF)
option(BB_ESMX_DICE_ACOUSTIC "Include the experimental D6 positive and negation searches" OFF)
if(BB_ESMX_DICE_ACOUSTIC AND NOT BB_ESMX_CLOCK_ACOUSTIC)
    message(FATAL_ERROR "BB_ESMX_DICE_ACOUSTIC requires the shared Hora acoustic model")
endif()
if(BB_ESMX_CLOCK_ACOUSTIC AND NOT BB_ESMX_JOKE_FALLBACK)
    message(FATAL_ERROR "BB_ESMX_CLOCK_ACOUSTIC requires BB_ESMX_JOKE_FALLBACK")
endif()
if(BB_ESMX_CLOCK_FALLBACK AND NOT BB_ESMX_JOKE_FALLBACK)
    message(FATAL_ERROR "BB_ESMX_CLOCK_FALLBACK requires the shared phoneme bank in BB_ESMX_JOKE_FALLBACK")
endif()
if(BB_ESMX_JOKE_FALLBACK AND NOT BB_ESMX_PILOT)
    message(FATAL_ERROR "BB_ESMX_JOKE_FALLBACK requires BB_ESMX_PILOT")
endif()
if(BB_ESMX_PILOT)
    foreach(header IN ITEMS pilot_model.h phrase_bank.h word_bank.h word_extra.h word_adapted.h
                            acoustic_dice.h consensus_acoustic_dice.h backup_acoustic_dice.h voice_data.h)
        if(NOT IS_ABSOLUTE "${BB_ESMX_ASSETS}" OR NOT EXISTS "${BB_ESMX_ASSETS}/${header}")
            message(FATAL_ERROR "Generate the local es-MX pilot assets and set BB_ESMX_ASSETS. Missing: ${header}")
        endif()
    endforeach()
    set(ESMX_EXPERIMENT "${CMAKE_CURRENT_SOURCE_DIR}/../experiments/esmx-compactas")
    target_compile_definitions(${COMPONENT_LIB} PRIVATE BB_ESMX_PILOT)
    target_sources(${COMPONENT_LIB} PRIVATE
                   "${ESMX_EXPERIMENT}/guard/PhraseGate.cpp"
                   "${ESMX_EXPERIMENT}/autonomous/WordDice.cpp"
                   "${ESMX_EXPERIMENT}/autonomous/AcousticDice.cpp")
    target_include_directories(${COMPONENT_LIB} PRIVATE "${BB_ESMX_ASSETS}" "${ESMX_EXPERIMENT}/guard"
                               "${ESMX_EXPERIMENT}/autonomous" "${CMAKE_CURRENT_SOURCE_DIR}/src/Audio/EsMxKws")
endif()

if(BB_ESMX_JOKE_FALLBACK)
    foreach(header IN ITEMS student_bank.h joke_rule.h)
        if(NOT EXISTS "${BB_ESMX_ASSETS}/${header}")
            message(FATAL_ERROR "Generate the local es-MX Chiste assets. Missing: ${header}")
        endif()
    endforeach()
    set(ESMX_PHONEME "${CMAKE_CURRENT_SOURCE_DIR}/../experiments/esmx-pilot/phoneme_native")
    set(ESMX_JOKE_SOURCES "${ESMX_PHONEME}/Kernel.cpp" "${ESMX_PHONEME}/Alignment.cpp"
                         "${ESMX_PHONEME}/JokeGuard.cpp")
    if(BB_ESMX_CLOCK_FALLBACK)
        if(NOT EXISTS "${BB_ESMX_ASSETS}/clock_rule.h")
            message(FATAL_ERROR "Generate the local es-MX Hora asset clock_rule.h")
        endif()
        list(APPEND ESMX_JOKE_SOURCES "${ESMX_PHONEME}/ClockGuard.cpp")
        target_compile_definitions(${COMPONENT_LIB} PRIVATE BB_ESMX_CLOCK_FALLBACK)
    endif()
    target_sources(${COMPONENT_LIB} PRIVATE ${ESMX_JOKE_SOURCES})
    target_compile_definitions(${COMPONENT_LIB} PRIVATE BB_ESMX_JOKE_FALLBACK)
    target_include_directories(${COMPONENT_LIB} PRIVATE "${ESMX_PHONEME}")
    set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/Audio/EsMxPilot/JokeFallback.cpp"
                        ${ESMX_JOKE_SOURCES} APPEND PROPERTY COMPILE_OPTIONS -ffp-contract=off)
endif()

if(BB_ESMX_CLOCK_ACOUSTIC)
    foreach(input IN ITEMS BB_ESMX_SPHINX_LIBRARY BB_ESMX_SPHINX_INCLUDE BB_ESMX_SPHINX_GENERATED_INCLUDE
                           BB_ESMX_CLOCK_PACK)
        if(NOT IS_ABSOLUTE "${${input}}" OR NOT EXISTS "${${input}}")
            message(FATAL_ERROR "Missing absolute local path for ${input}")
        endif()
    endforeach()
    if(NOT EXISTS "${BB_ESMX_SPHINX_INCLUDE}/pocketsphinx.h" OR
       NOT EXISTS "${BB_ESMX_SPHINX_GENERATED_INCLUDE}/pocketsphinx/sphinx_config.h" OR
       NOT EXISTS "${BB_ESMX_CLOCK_PACK}/ClockModelAssets.h")
        message(FATAL_ERROR "Missing generated Hora model or PocketSphinx headers")
    endif()
    add_prebuilt_library(bb_esmx_pocketsphinx "${BB_ESMX_SPHINX_LIBRARY}")
    target_link_libraries(${COMPONENT_LIB} PRIVATE bb_esmx_pocketsphinx)
    target_link_options(${COMPONENT_LIB} INTERFACE "-Wl,-u,__wrap_longjmp")
    target_compile_definitions(${COMPONENT_LIB} PRIVATE BB_ESMX_CLOCK_ACOUSTIC)
    if(BB_ESMX_DICE_ACOUSTIC)
        if(NOT EXISTS "${BB_ESMX_CLOCK_PACK}/DiceAcousticAssets.h")
            message(FATAL_ERROR "Missing generated D6 acoustic tables")
        endif()
        target_compile_definitions(${COMPONENT_LIB} PRIVATE BB_ESMX_DICE_ACOUSTIC)
    endif()
    target_include_directories(${COMPONENT_LIB} PRIVATE "${BB_ESMX_SPHINX_INCLUDE}"
                               "${BB_ESMX_SPHINX_GENERATED_INCLUDE}" "${BB_ESMX_CLOCK_PACK}")
    set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/Audio/EsMxPilot/ClockModelVfs.cpp"
                 APPEND PROPERTY COMPILE_OPTIONS -ffp-contract=off)
endif()

spiffs_create_partition_image(storage ../spiffs FLASH_IN_PROJECT)

idf_build_get_property(BASEDIR PROJECT_DIR)
esptool_py_flash_to_partition(flash objdet ${BASEDIR}/models/objdet.tflite)
