# The opt-in application and its target replay use the same acoustics.
set(BB_BUTTER_BRIDGE "${CMAKE_CURRENT_LIST_DIR}")
get_filename_component(BB_SPHINX_ARCHIVE "${BB_ESMX_SPHINX_LIBRARY}" REALPATH)
get_filename_component(BB_SPHINX_BUILD "${BB_SPHINX_ARCHIVE}" DIRECTORY)
get_filename_component(BB_SPHINX_ROOT "${BB_SPHINX_BUILD}" DIRECTORY)
foreach(input_and_path IN ITEMS
        "BB_ESMX_SPHINX_LIBRARY|build/libpocketsphinx.a"
        "BB_ESMX_SPHINX_INCLUDE|source/include"
        "BB_ESMX_SPHINX_GENERATED_INCLUDE|build/include"
        "BB_ESMX_SPHINX_PRIVATE_INCLUDE|source/src")
    string(REPLACE "|" ";" pair "${input_and_path}")
    list(GET pair 0 input)
    list(GET pair 1 path)
    get_filename_component(actual "${${input}}" REALPATH)
    if(NOT IS_ABSOLUTE "${${input}}" OR NOT EXISTS "${actual}" OR
       NOT actual STREQUAL "${BB_SPHINX_ROOT}/${path}")
        message(FATAL_ERROR "${input} must come from the same PocketSphinx output as the library")
    endif()
endforeach()

foreach(file_and_sha IN ITEMS
        "fsg_search.c|7e6c4dc78fe55cf75ff5790ef0c0dee6a525e00950302aa41816134f9d3bc5ae"
        "fsg_search_internal.h|7ca91296d72185ca1bc6ebb613b7def74617ed402e20985b9aaf0671e5d60a63"
        "fsg_history.c|7f5293e8d49ed41b07dfb03bd73f4a37d11b48a7d4e8865732b0bb8d2733342b"
        "pocketsphinx_internal.h|d17faa10167bda7483da5f887cefd45685ec85f4e76b3a6c1cf7c99242d24097"
        "acmod.h|5f24689453d032f94a1b0db396bc79a9434a1c4959d6353c33105936c0b33941")
    string(REPLACE "|" ";" pair "${file_and_sha}")
    list(GET pair 0 name)
    list(GET pair 1 reviewed)
    set(file "${BB_SPHINX_ROOT}/source/src/${name}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${file}")
    file(SHA256 "${file}" actual)
    if(NOT actual STREQUAL reviewed)
        message(FATAL_ERROR "Unreviewed PocketSphinx source: ${name}. Rebuild with --oppose-pool --skip-empty-history")
    endif()
endforeach()

set(BB_BUTTER_HEADER "${BB_ESMX_BUTTER_ASSETS}/ButterModelAssets.h")
set(BB_CLOCK_HEADER "${BB_ESMX_CLOCK_PACK}/ClockModelAssets.h")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${BB_BUTTER_HEADER}" "${BB_CLOCK_HEADER}")
file(SHA256 "${BB_BUTTER_HEADER}" BB_BUTTER_HEADER_SHA256)
file(SHA256 "${BB_CLOCK_HEADER}" BB_CLOCK_HEADER_SHA256)
if(NOT BB_BUTTER_HEADER_SHA256 STREQUAL "54dfd03c6637ccfa62947444114479525586e73b9a46d91e0cd8a3d74fa3ba62" OR
   NOT BB_CLOCK_HEADER_SHA256 STREQUAL "8ae50e695928dd57acf1dc3b6ffdfb78595fef66f5b3165e45b532fc28a297da")
    message(FATAL_ERROR "The Mantequilla application experiment requires the reviewed Butter and CI16 model headers")
endif()

target_include_directories(${COMPONENT_LIB} PRIVATE "${BB_BUTTER_BRIDGE}" "${BB_ESMX_SPHINX_PRIVATE_INCLUDE}")
target_sources(${COMPONENT_LIB} PRIVATE "${BB_BUTTER_BRIDGE}/cepstra_bridge.c")
target_compile_definitions(${COMPONENT_LIB} PRIVATE
    BB_ESMX_BUTTER_APP_EXPERIMENT BB_ESMX_BUTTER_CEPSTRA BB_ESMX_BUTTER_WEAK_CONFLICT)
