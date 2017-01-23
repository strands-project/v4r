project(${PCL_1_8_LIBRARY})
# This project contains files from PCL which are either not available in or have a different ABI than the currently released PCL version. 
v4r_assert(HAVE_PCL)
v4r_assert(HAVE_BOOST)

v4r_include_directories(
    ${PCL_INCLUDE_DIRS}
    ${BOOST_INCLUDE_DIRS}
${CMAKE_CURRENT_SOURCE_DIR} 
)

set(lib_srcs
 uniform_sampling.cpp
 conditional_euclidean_clustering.cpp
 organized_edge_detection.cpp
 convolution_2d.cpp

)

SET(lib_hdrs
 pcl_1_8/keypoints/impl/uniform_sampling.hpp
 pcl_1_8/segmentation/impl/conditional_euclidean_clustering.hpp
 pcl_1_8/features/impl/organized_edge_detection.hpp
 pcl_1_8/2d/impl/convolution.hpp
 pcl_1_8/2d/impl/kernel.hpp
 pcl_1_8/2d/impl/edge.hpp
 pcl_1_8/keypoints/uniform_sampling.h
 pcl_1_8/segmentation/conditional_euclidean_clustering.h
 pcl_1_8/features/organized_edge_detection.h
 pcl_1_8/2d/edge.h
 pcl_1_8/2d/convolution.h
 pcl_1_8/2d/kernel.h
)

include_directories(
${PCL_INCLUDE_DIRS}
${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(${PCL_1_8_LIBRARY} STATIC ${lib_srcs} ${lib_hdrs} )
target_link_libraries(${PCL_1_8_LIBRARY} ${PCL_LIBRARIES} )

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
