if(WITH_GLFW3)
  if(BUILD_GLFW3)
    v4r_clear_vars(GLFW3_FOUND)
  else()
    message(WARNING "GLFW3 support is enabled, but building from source is disabled. "
                    "This option is not implemented, so you will need to write rules to "
                    "find system-wide installation of GLFW3 yourself.")
  endif()
  if(NOT GLFW3_FOUND)
    v4r_clear_vars(GLFW3_LIBRARY GLFW3_INCLUDE_DIRS)
    set(GLFW3_LIBRARY glfw3)
    add_subdirectory("${V4R_SOURCE_DIR}/3rdparty/glfw3")
    set(GLFW3_LIBRARIES ${GLFW3_LIBRARY} Xxf86vm)
    set(GLFW3_INCLUDE_DIRS "${${GLFW3_LIBRARY}_INSTALL_DIR}/include")
  endif()
  set(GLFW3_VERSION "3.1.2") # TODO
  set(HAVE_GLFW3 YES)
endif()
