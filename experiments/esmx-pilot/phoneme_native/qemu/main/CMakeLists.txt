idf_component_register(SRCS "bench.cpp" "../../Kernel.cpp" "../../Alignment.cpp" INCLUDE_DIRS "../.."
                       REQUIRES esp_timer)

foreach(header IN ITEMS student_bank.h student_tests.h)
    if(NOT IS_ABSOLUTE "${PHONEME_ASSETS}" OR NOT EXISTS "${PHONEME_ASSETS}/${header}")
        message(FATAL_ERROR "Generate the local model and tests in PHONEME_ASSETS. Missing: ${header}")
    endif()
endforeach()
target_include_directories(${COMPONENT_LIB} PRIVATE "${PHONEME_ASSETS}")
target_compile_features(${COMPONENT_LIB} PRIVATE cxx_std_20)
target_compile_options(${COMPONENT_LIB} PRIVATE -ffp-contract=off)
