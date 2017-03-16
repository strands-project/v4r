cmake_minimum_required(VERSION 2.8.12)
project(gtest NONE)
include(ExternalProject)

# Make a target that downloads gtest
set(_external_target "gtest_external")
ExternalProject_Add(${_external_target}
  URL "https://github.com/google/googletest/archive/release-1.8.0.tar.gz"
  URL_HASH SHA1=e7e646a6204638fe8e87e165292b8dd9cd4c36ed
  TIMEOUT 30
  CMAKE_ARGS "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>" "-DBUILD_GTEST=ON" "-DBUILD_GMOCK=OFF"
)

# INSTALL_DIR is where the library was installed after building
ExternalProject_Get_Property(${_external_target} INSTALL_DIR)

# Let CMake know about the static libraries built by the external project target
# gtest.a
add_library(gtest STATIC IMPORTED GLOBAL)
add_dependencies(gtest ${_external_target})
set_target_properties(gtest
  PROPERTIES
    IMPORTED_LOCATION "${INSTALL_DIR}/lib/libgtest.a"
    OUTPUT_NAME gtest
    ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
)
# gtest_main.a
add_library(gtest_main STATIC IMPORTED GLOBAL)
add_dependencies(gtest_main ${_external_target})
set_target_properties(gtest_main
  PROPERTIES
    IMPORTED_LOCATION "${INSTALL_DIR}/lib/libgtest_main.a"
    OUTPUT_NAME gtest_main
    ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
)

set(GTEST_INCLUDE_DIRS "${INSTALL_DIR}/include" CACHE INTERNAL "Google Test include directories")
set(GTEST_LIBRARIES gtest gtest_main CACHE INTERNAL "Google Test libraries")
set(GTEST_VERSION 1.8.0 CACHE INTERNAL "Google Test version")
