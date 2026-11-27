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

spiffs_create_partition_image(storage ../spiffs FLASH_IN_PROJECT)

idf_build_get_property(BASEDIR PROJECT_DIR)
esptool_py_flash_to_partition(flash objdet ${BASEDIR}/models/objdet.tflite)
