cmake_minimum_required (VERSION 3.0.2)

set(PROJECT "Pathtracer")
project (${PROJECT})

find_package(CUDA QUIET REQUIRED)

#set (${PROJECT}_VERSION_MAJOR 1)
#set (${PROJECT}_VERSION_MINOR 0)

# Treat .cpp files as .cu files
set_source_files_properties( ${BASE} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )

# Allow kernel functions to be in a different translation unit
set(CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# Set the GPU architecture
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -arch=sm_61 
)

#configure_file (
#  "${PROJECT_SOURCE_DIR}/${PROJECT}Config.h.in"
#  "${PROJECT_BINARY_DIR}/${PROJECT}Config.h"
#)


add_subdirectory(Source)

include_directories("${PROJECT_BINARY_DIR}")
include_directories(${CMAKE_SOURCE_DIR}/3rdParty/Includes/)

cuda_add_executable(${PROJECT}
    ${BASE}
)

if(WIN32)
   set_target_properties(${PROJECT} PROPERTIES LINK_FLAGS_RELEASE "/SUBSYSTEM:CONSOLE")
endif(WIN32)

#source_group( Source FILES ${BASE})

target_link_libraries(${PROJECT} ${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/Libraries/assimp.lib)
target_link_libraries(${PROJECT} ${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/Libraries/glfw3.lib)

add_custom_command(TARGET ${PROJECT} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${PROJECT_SOURCE_DIR}/3rdParty/Binaries/"
        ${CMAKE_BINARY_DIR}/Release)

add_custom_command(TARGET ${PROJECT} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${PROJECT_SOURCE_DIR}/Resources/"
        ${CMAKE_SOURCE_DIR}/build/res)
