  v4r_clear_vars(PCL-1_8_FOUND)
  if(NOT PCL-1_8_FOUND)
    v4r_clear_vars(PCL-1_8_LIBRARY PCL-1_8_LIBRARIES PCL-1_8_INCLUDE_DIRS)
    set(PCL-1_8_LIBRARY pcl-1_8)
    set(PCL_LIBRARIES "${PCL_LIBRARIES} ${PCL-1_8_LIBRARY}")
    add_subdirectory("${V4R_SOURCE_DIR}/3rdparty/pcl-1_8")
    set(PCL_INCLUDE_DIRS "${PCL_INCLUDE_DIRS} ${V4R_SOURCE_DIR}/3rdparty")
  endif()

