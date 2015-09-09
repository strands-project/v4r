if(WITH_LIBSVM)
  if(BUILD_LIBSVM)
    v4r_clear_vars(LIBSVM_FOUND)
  else()
    message(WARNING "LIBSVM support is enabled, but building from source is disabled. "
                    "This option is not implemented, so you will need to write rules to "
                    "find system-wide installation of LIBSVM yourself.")
  endif()
  if(NOT LIBSVM_FOUND)
    v4r_clear_vars(LIBSVM_LIBRARY LIBSVM_INCLUDE_DIRS)
    set(LIBSVM_LIBRARY libsvm)
    set(LIBSVM_LIBRARIES ${LIBSVM_LIBRARY})
    add_subdirectory("${V4R_SOURCE_DIR}/3rdparty/libsvm")
    set(LIBSVM_INCLUDE_DIRS "${V4R_SOURCE_DIR}/3rdparty")
  endif()
  set(LIBSVM_VERSION "3.20") # TODO
  set(HAVE_LIBSVM YES)
endif()
