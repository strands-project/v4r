if(BUILD_GTEST)
  v4r_clear_vars(GTEST_FOUND)
else()
  message(WARNING "GTest is required, but building it from source is disabled. "
                  "This option is not implemented, so you will need to write rules "
                  "to find system-wide installation of GTest yourself.")
endif()
if(NOT GTEST_FOUND)
  add_subdirectory("${V4R_SOURCE_DIR}/3rdparty/gtest")
endif()
set(HAVE_GTEST YES)
