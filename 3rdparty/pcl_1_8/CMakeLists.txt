project(${PCL_1_8_LIBRARY})
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
2d/src/examples.cpp
2d/src/convolution_2d.cpp

)

SET(lib_hdrs
filters/include/pcl_1_8/filters/impl/uniform_sampling.hpp
segmentation/include/pcl_1_8/segmentation/impl/conditional_euclidean_clustering.hpp
features/include/pcl_1_8/features/impl/organized_edge_detection.hpp
2d/include/pcl_1_8/2d/impl/convolution.hpp
2d/include/pcl_1_8/2d/impl/morphology.hpp
2d/include/pcl_1_8/2d/impl/kernel.hpp
2d/include/pcl_1_8/2d/impl/edge.hpp
2d/include/pcl_1_8/2d/impl/keypoint.hpp
filters/include/pcl_1_8/filters/uniform_sampling.h
segmentation/include/pcl_1_8/segmentation/conditional_euclidean_clustering.h
features/include/pcl_1_8/features/organized_edge_detection.h
2d/include/pcl_1_8/2d/edge.h
2d/include/pcl_1_8/2d/convolution.h
2d/include/pcl_1_8/2d/morphology.h
2d/include/pcl_1_8/2d/kernel.h
2d/include/pcl_1_8/2d/keypoint.h
)

include_directories(
${PCL_INCLUDE_DIRS}
${CMAKE_CURRENT_SOURCE_DIR}/2d/include/
${CMAKE_CURRENT_SOURCE_DIR}/features/include/
${CMAKE_CURRENT_SOURCE_DIR}/filters/include/
${CMAKE_CURRENT_SOURCE_DIR}/segmentation/include/
)


add_library(${PCL_1_8_LIBRARY} STATIC ${lib_srcs} ${lib_hdrs})
target_link_libraries(${PCL_1_8_LIBRARY} ${PCL_LIBRARIES})

set_target_properties(${PCL_1_8_LIBRARY}
  PROPERTIES
    POSITION_INDEPENDENT_CODE TRUE
    OUTPUT_NAME "${PCL_1_8_LIBRARY}"
    ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
)

install(FILES ${lib_hdrs} DESTINATION "${V4R_3P_INCLUDE_INSTALL_PATH}/pcl_1_8")

if(NOT BUILD_SHARED_LIBS)
  v4r_install_target(${PCL_1_8_LIBRARY} EXPORT V4RModules ARCHIVE DESTINATION ${V4R_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()
