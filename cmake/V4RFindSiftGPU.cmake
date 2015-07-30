if(WITH_SIFTGPU)
  if(BUILD_SIFTGPU)
    v4r_clear_vars(SIFTGPU_FOUND)
  else()
    message(WARNING "SiftGPU support is enabled, but building from source is disabled. "
                    "This option is not implemented, so you will need to write rules to "
                    "find system-wide installation of SiftGPU yourself.")
  endif()
  if(NOT SIFTGPU_FOUND)
    v4r_clear_vars(SIFTGPU_LIBRARY SIFTGPU_INCLUDE_DIR)
    set(SIFTGPU_LIBRARY siftgpu)
    set(SIFTGPU_LIBRARIES ${SIFTGPU_LIBRARY})
    add_subdirectory("${V4R_SOURCE_DIR}/3rdparty/SiftGPU")
    set(SIFTGPU_INCLUDE_DIR "${${SIFTGPU_LIBRARY}_SOURCE_DIR}/src")
  endif()
  set(SIFTGPU_VERSION "v400") # TODO
  set(HAVE_SIFTGPU YES)
endif()
