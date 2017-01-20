project(${PCL-1_8_LIBRARY})
# This project contains files from PCL which are either not available in or have a different ABI than the currently released PCL version. 

v4r_assert(HAVE_PCL)

v4r_include_directories(
${PCL_INCLUDE_DIRS}
${CMAKE_CURRENT_SOURCE_DIR} 
)

set(lib_srcs
  filters/src/uniform_sampling.cpp
  segmentation/src/conditional_euclidean_clustering.cpp
  features/src/organized_edge_detection.cpp
)

SET(lib_hdrs
  filters/include/pcl-1_8/filters/uniform_sampling.h
  segmentation/include/pcl-1_8/segmentation/conditional_euclidean_clustering.h
  features/include/pcl-1_8/features/organized_edge_detection.h
  filters/include/pcl-1_8/filters/impl/uniform_sampling.hpp
  segmentation/include/pcl-1_8/segmentation/impl/conditional_euclidean_clustering.hpp
  features/include/pcl-1_8/features/impl/organized_edge_detection.hpp
)

include_directories(
${PCL_INCLUDE_DIRS}
${CMAKE_CURRENT_SOURCE_DIR}/filters/include/
${CMAKE_CURRENT_SOURCE_DIR}/segmentation/include/
${CMAKE_CURRENT_SOURCE_DIR}/features/include/
)

message ("**********************+** ${PCL_INCLUDE_DIRS} *************************")

add_library(${PCL-1_8_LIBRARY} STATIC ${lib_srcs})
target_link_libraries(${PCL-1_8_LIBRARY} ${PCL_LIBRARIES})

set_target_properties(${PCL-1_8_LIBRARY}
  PROPERTIES
    POSITION_INDEPENDENT_CODE TRUE
    OUTPUT_NAME "${PCL-1_8_LIBRARY}"
    ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
)

install(FILES ${lib_hdrs} DESTINATION "${V4R_3P_INCLUDE_INSTALL_PATH}/pcl-1_8")

if(NOT BUILD_SHARED_LIBS)
  v4r_install_target(${PCL-1_8_LIBRARY} EXPORT V4RModules ARCHIVE DESTINATION ${V4R_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()
