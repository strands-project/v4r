# ----------------------------------------------------------------------------
#   Uninstall target, for "make uninstall"
# ----------------------------------------------------------------------------
CONFIGURE_FILE(
  "${V4R_SOURCE_DIR}/cmake/templates/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  @ONLY)

ADD_CUSTOM_TARGET(uninstall "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")

# ----------------------------------------------------------------------------
# target building all V4R modules
# ----------------------------------------------------------------------------
add_custom_target(v4r_modules)
if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(v4r_modules PROPERTIES FOLDER "extra")
endif()

# ----------------------------------------------------------------------------
# targets building all tests
# ----------------------------------------------------------------------------
if(BUILD_TESTS)
  add_custom_target(v4r_tests)
endif()
if(BUILD_PERF_TESTS)
  add_custom_target(v4r_perf_tests)
endif()
