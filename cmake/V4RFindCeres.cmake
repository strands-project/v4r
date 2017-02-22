if(WITH_CERES)
  find_package(Ceres)
  if(${Ceres_FOUND})

    # If we build V4R in shared mode, Ceres library should also be a shared object,
    # otherwise we will get a relocation error at linking stage.
    if(BUILD_SHARED_LIBS)
      get_target_property(_ceres_type ceres TYPE)
      if(_ceres_type STREQUAL "STATIC_LIBRARY")
        message(FATAL_ERROR "Found libceres, however it is a static library and can not be linked with V4R shared libraries.\nYour options:\n * Install Ceres shared library\n * Disable WITH_CERES\n * Disable BUILD_SHARED_LIBS")
      endif()
    endif()

    set(CERES_LIBRARIES "${CERES_LIBRARIES}")
    set(CERES_INCLUDE_DIRS "${CERES_INCLUDES}")
    set(HAVE_CERES TRUE)
    if(${Ceres_VERSION} EQUAL 1.8.0)
     add_definitions(-DCERES_VERSION_LESS_1_9_0)
    endif(${Ceres_VERSION} EQUAL 1.8.0)
  else()
    message(FATAL_ERROR "libceres not found. Install it or disable WITH_CERES")
  endif()
else()
  add_definitions(-DKP_NO_CERES_AVAILABLE)
endif()

