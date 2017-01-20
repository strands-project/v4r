  v4r_clear_vars(PCL_1_8_FOUND)
  if(NOT PCL_1_8_FOUND)
    v4r_clear_vars(PCL_1_8_LIBRARY PCL_1_8_LIBRARIES PCL_1_8_INCLUDE_DIRS)
    set(PCL_1_8_LIBRARY pcl_1_8)
    set(PCL_LIBRARIES "${PCL_LIBRARIES} ${PCL_1_8_LIBRARY}")
    add_subdirectory("${V4R_SOURCE_DIR}/3rdparty/pcl_1_8")
    set(PCL_INCLUDE_DIRS "${PCL_INCLUDE_DIRS} ${V4R_SOURCE_DIR}/3rdparty")
  endif()

