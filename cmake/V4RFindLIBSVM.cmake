if(WITH_LIBSVM)
  if(BUILD_LIBSVM)
    v4r_clear_vars(LIBSVM_FOUND)
  else()
    find_path(LIBSVM_INCLUDE_PATH libsvm/svm.h
              PATHS /usr/local /opt /usr $ENV{LIBSVM_ROOT}
              DOC "The path to LibSVM header"
              CMAKE_FIND_ROOT_PATH_BOTH)
    find_library(LIBSVM_LIBRARIES
                 NAMES svm
                 PATHS /usr/local /opt /usr $ENV{LIBSVM_ROOT}
                 DOC "The LibSVM library")
    if(LIBSVM_INCLUDE_PATH AND LIBSVM_LIBRARIES)
      set(LIBSVM_INCLUDE_DIRS "${LIBSVM_INCLUDE_PATH}")
      v4r_parse_header("${LIBSVM_INCLUDE_PATH}/libsvm/svm.h" LIBSVM_VERSION_LINES LIBSVM_VERSION)
      set(LIBSVM_FOUND YES)
    endif()
  endif()
  if(NOT LIBSVM_FOUND)
    v4r_clear_vars(LIBSVM_LIBRARY LIBSVM_INCLUDE_DIRS LIBSVM_VERSION)
    set(LIBSVM_LIBRARY libsvm)
    set(LIBSVM_LIBRARIES ${LIBSVM_LIBRARY})
    add_subdirectory("${V4R_SOURCE_DIR}/3rdparty/libsvm")
    set(LIBSVM_INCLUDE_DIRS "${V4R_SOURCE_DIR}/3rdparty")
    set(LIBSVM_VERSION "3.20") # TODO
  endif()
  set(HAVE_LIBSVM YES)
endif()
