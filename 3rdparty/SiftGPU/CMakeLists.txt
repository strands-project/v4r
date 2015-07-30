include(ExternalProject)

ExternalProject_Add(SiftGPU
  PREFIX "${CMAKE_CURRENT_LIST_DIR}"
  URL "http://wwwx.cs.unc.edu/~ccwu/cgi-bin/siftgpu.cgi?SiftGPU-V400.zip"
  URL_HASH SHA1=eac08305f14cb35d12a22d493f84f9c21f6b7fdb
  TIMEOUT 30
  CONFIGURE_COMMAND ""
  BUILD_COMMAND make siftgpu_enable_cuda=0
  BUILD_IN_SOURCE 1
  INSTALL_COMMAND ""
)

# SOURCE_DIR is where the source was unpacked and then built
ExternalProject_Get_Property(SiftGPU SOURCE_DIR)
set(${SIFTGPU_LIBRARY}_SOURCE_DIR ${SOURCE_DIR} CACHE INTERNAL "SiftGPU source directory")

add_library(${SIFTGPU_LIBRARY} STATIC IMPORTED GLOBAL)

add_dependencies(${SIFTGPU_LIBRARY} SiftGPU)

set_target_properties(${SIFTGPU_LIBRARY}
  PROPERTIES
    IMPORTED_LOCATION "${SOURCE_DIR}/bin/libsiftgpu.a"
    OUTPUT_NAME "${SIFTGPU_LIBRARY}"
    ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
)

if(NOT BUILD_SHARED_LIBS)
  v4r_install_target(${SIFTGPU_LIBRARY} EXPORT V4RModules ARCHIVE DESTINATION ${V4R_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()
