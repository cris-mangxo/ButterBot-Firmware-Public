# The opt-in application and its target replay use the same acoustics.
set(BB_BUTTER_BRIDGE "${CMAKE_CURRENT_LIST_DIR}")
if(NOT IS_ABSOLUTE "${BB_ESMX_SPHINX_PRIVATE_INCLUDE}" OR
   NOT EXISTS "${BB_ESMX_SPHINX_PRIVATE_INCLUDE}/pocketsphinx_internal.h")
    message(FATAL_ERROR "Use the private headers from the PocketSphinx build with --oppose-pool --skip-empty-history")
endif()

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
