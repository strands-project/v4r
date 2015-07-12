set(SUBSYS_NAME common)
set(SUBSYS_DESC "V4R common module")
set(SUBSYS_DEPS )
set(V4R_COMMON 1 CACHE BOOL "V4R common module" )
IF(V4R_COMMON)
PROJECT(v4r_${SUBSYS_NAME})

FIND_PACKAGE (PCL)
INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

FIND_PACKAGE (OpenCV)
include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})

SET(SOURCE_CPP
  src/fast_icp_with_gc.cpp
  src/hough_3d.cpp
  src/kmeans.cpp
  src/miscellaneous.cpp
  src/pcl_visualization_utils.cpp
  src/visibility_reasoning.cpp
)

SET(SOURCE_H
  include/v4r/${SUBSYS_NAME}/convolution.h
  include/v4r/${SUBSYS_NAME}/common_data_structures.h
  include/v4r/${SUBSYS_NAME}/edge.h
  include/v4r/${SUBSYS_NAME}/faat_3d_rec_framework_defines.h
  include/v4r/${SUBSYS_NAME}/fast_icp_with_gc.h
  include/v4r/${SUBSYS_NAME}/hough_3d.h
  include/v4r/${SUBSYS_NAME}/kernel.h
  include/v4r/${SUBSYS_NAME}/kmeans.h
  include/v4r/${SUBSYS_NAME}/miscellaneous.h
  include/v4r/${SUBSYS_NAME}/noise_models.h
  include/v4r/${SUBSYS_NAME}/noise_model_based_cloud_integration.h
  include/v4r/${SUBSYS_NAME}/normal_estimator.h
  include/v4r/${SUBSYS_NAME}/occlusion_reasoning.h
  include/v4r/${SUBSYS_NAME}/organized_edge_detection.h
  include/v4r/${SUBSYS_NAME}/pcl_opencv.h
  include/v4r/${SUBSYS_NAME}/pcl_visualization_utils.h
  include/v4r/${SUBSYS_NAME}/uniform_sampling.h
  include/v4r/${SUBSYS_NAME}/visibility_reasoning.h
  
  include/v4r/${SUBSYS_NAME}/impl/convolution.hpp
  include/v4r/${SUBSYS_NAME}/impl/edge.hpp
  include/v4r/${SUBSYS_NAME}/impl/fast_icp_with_gc.hpp
  include/v4r/${SUBSYS_NAME}/impl/hough_3d.hpp
  include/v4r/${SUBSYS_NAME}/impl/kernel.hpp
  include/v4r/${SUBSYS_NAME}/impl/kmeans.hpp
  include/v4r/${SUBSYS_NAME}/impl/miscellaneous.hpp
  include/v4r/${SUBSYS_NAME}/impl/occlusion_reasoning.hpp
  include/v4r/${SUBSYS_NAME}/impl/organized_edge_detection.hpp
  include/v4r/${SUBSYS_NAME}/impl/visibility_reasoning.hpp
)


SET(SOURCE_H_CG
  include/v4r/${SUBSYS_NAME}/correspondence_grouping.h
  include/v4r/${SUBSYS_NAME}/impl/correspondence_grouping.hpp
  include/v4r/${SUBSYS_NAME}/geometric_consistency.h
  include/v4r/${SUBSYS_NAME}/impl/geometric_consistency.hpp
  include/v4r/${SUBSYS_NAME}/graph_geometric_consistency.h
  include/v4r/${SUBSYS_NAME}/impl/graph_geometric_consistency.hpp
)

SET(SOURCE_CPP_CG
  src/geometric_consistency.cpp
  src/graph_geometric_consistency.cpp
  src/noise_models.cpp
  src/noise_model_based_cloud_integration.cpp
  src/organized_edge_detection.cpp
)


include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(${PROJECT_NAME} SHARED ${SOURCE_H} ${SOURCE_CPP} ${SOURCE_H_CG} ${SOURCE_CPP_CG} ${SOURCE_H_CG} ${SOURCE_CPP_CG} ${SOURCE_H_FEATURES} ${SOURCE_CPP_FEATURES} )
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${SUBSYS_DEPS})
v4r_add_library(${PROJECT_NAME} "${SUBSYS_NAME}" "${SOURCE_H} ${SOURCE_H_CG}")


SET(SOURCE_H_FEATURES
  #3D LOCAL
  include/v4r/${SUBSYS_NAME}/features/local_estimator.h 
  include/v4r/${SUBSYS_NAME}/features/normal_estimator.h
  include/v4r/${SUBSYS_NAME}/features/shot_local_estimator.h
  include/v4r/${SUBSYS_NAME}/features/shot_local_estimator_omp.h
  include/v4r/${SUBSYS_NAME}/features/colorshot_local_estimator.h
  #3D GLOBAL
  include/v4r/${SUBSYS_NAME}/features/global_estimator.h
  include/v4r/${SUBSYS_NAME}/features/organized_ourcvfh.h
  include/v4r/${SUBSYS_NAME}/features/color_ourcvfh_estimator.h  
  include/v4r/${SUBSYS_NAME}/features/ourcvfh_estimator.h
  include/v4r/${SUBSYS_NAME}/features/crh_estimator.h  
  include/v4r/${SUBSYS_NAME}/features/cvfh_estimator.h
  include/v4r/${SUBSYS_NAME}/features/esf_estimator.h
  include/v4r/${SUBSYS_NAME}/features/vfh_estimator.h
  include/v4r/${SUBSYS_NAME}/features/organized_color_ourcvfh_estimator.h  
  #image based
  include/v4r/${SUBSYS_NAME}/features/image_local_estimator.h
  include/v4r/${SUBSYS_NAME}/features/vedaldi_sift_local_estimator.h
  include/v4r/${SUBSYS_NAME}/features/sift_local_estimator.h
  include/v4r/${SUBSYS_NAME}/features/opencv_sift_local_estimator.h
  
  include/v4r/${SUBSYS_NAME}/features/faat_3d_rec_framework_defines.h
)

SET(SOURCE_CPP_FEATURES
  src/features/local_estimator.cpp
  src/features/shot_local_estimator.cpp
)

add_library("${PROJECT_NAME}_features" SHARED ${SOURCE_H_FEATURES} ${SOURCE_CPP_FEATURES} )
target_link_libraries("${PROJECT_NAME}_features" ${PCL_LIBRARIES} ${SUBSYS_DEPS})
v4r_add_library("${PROJECT_NAME}_features" "${SUBSYS_NAME}" "${SOURCE_FEATURES}")



SET(SOURCE_H_IO
  include/v4r/${SUBSYS_NAME}/io/filesystem_utils.h
)

SET(SOURCE_CPP_IO
  src/io/filesystem_utils.cpp
)

add_library("${PROJECT_NAME}_io" SHARED ${SOURCE_H_IO} ${SOURCE_CPP_IO} )
target_link_libraries("${PROJECT_NAME}_io" ${PCL_LIBRARIES} ${SUBSYS_DEPS})
v4r_add_library("${PROJECT_NAME}_io" "${SUBSYS_NAME}" "${SOURCE_H_IO}")


ENDIF(V4R_COMMON)
