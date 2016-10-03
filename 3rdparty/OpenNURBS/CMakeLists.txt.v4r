cmake_minimum_required(VERSION 2.8.12)
project(${SIFTGPU_LIBRARY})
include(ExternalProject)

# Make a target that downloads and builds SiftGPU from source.
# The install command copies the (only) library header to the root SiftGPU directory. This is needed to allow using V4R
# from without installation (directly from build/ location).
set(_external_target "${SIFTGPU_LIBRARY}_external")
ExternalProject_Add(${_external_target}
  PREFIX "${CMAKE_CURRENT_LIST_DIR}"
  URL "http://wwwx.cs.unc.edu/~ccwu/cgi-bin/siftgpu.cgi?SiftGPU-V400.zip"
  URL_HASH SHA1=eac08305f14cb35d12a22d493f84f9c21f6b7fdb
  TIMEOUT 30
  CONFIGURE_COMMAND ""
  BUILD_COMMAND make siftgpu_enable_cuda=0
  BUILD_IN_SOURCE 1
  INSTALL_COMMAND cp <SOURCE_DIR>/src/SiftGPU/SiftGPU.h ${CMAKE_CURRENT_SOURCE_DIR}
)

# SOURCE_DIR is where the source was unpacked and then built.
# We want to make this available outside of the scope of this file.
ExternalProject_Get_Property(${_external_target} SOURCE_DIR)
set(${SIFTGPU_LIBRARY}_SOURCE_DIR ${SOURCE_DIR} CACHE INTERNAL "SiftGPU source directory")

# Let CMake know about the static library built by the external project target.
add_library(${SIFTGPU_LIBRARY} STATIC IMPORTED GLOBAL)
add_dependencies(${SIFTGPU_LIBRARY} ${_external_target})
set_target_properties(${SIFTGPU_LIBRARY}
  PROPERTIES
    IMPORTED_LOCATION "${SOURCE_DIR}/bin/libsiftgpu.a"
    OUTPUT_NAME "${SIFTGPU_LIBRARY}"
    ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
)

# Setup a rule to install header files
install(FILES "${SOURCE_DIR}/src/SiftGPU/SiftGPU.h" DESTINATION "${V4R_3P_INCLUDE_INSTALL_PATH}/SiftGPU")

if(NOT BUILD_SHARED_LIBS)
  # Normally we should use "v4r_install_target", however "siftgpu" is an imported library, and it will not work.
  # To make CMake happy, install the library excplicitly.
  install(FILES "${SOURCE_DIR}/bin/libsiftgpu.a" DESTINATION ${V4R_3P_LIB_INSTALL_PATH})
endif()
