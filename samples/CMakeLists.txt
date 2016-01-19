# Detect if we want to build samples with library binaries or not
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_LIST_DIR)
#
# BUILD CASE 1: Build samples with library sources
#


# ----------------------------------------------------------------------------
#  CMake file for samples. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------

#add_subdirectory(c)
add_subdirectory(examples)
add_subdirectory(eval)
add_subdirectory(tools)
#add_subdirectory(icra16)
#add_subdirectory(ocl)

#
# END OF BUILD CASE 1: Build samples with library sources
#
else()
#
# BUILD CASE 2: Build samples with library binaries
#
cmake_minimum_required(VERSION 2.8)

project(samples C CXX)
option(BUILD_EXAMPLES "Build samples" ON)
option(BUILD_EVALUATION_TOOLS "Build evaluation tools" ON)
option(BUILD_UTILITY_TOOLS "Build utility tools" ON)
#option(BUILD_ICRA16_TOOLS "Build evaluations for RA-L Paper 2016, Faeulhammer" OFF)


find_package(V4R REQUIRED)

#add_subdirectory(c)
add_subdirectory(examples)
add_subdirectory(eval)
add_subdirectory(tools)
#add_subdirectory(icra16)
#add_subdirectory(ocl)

endif()
