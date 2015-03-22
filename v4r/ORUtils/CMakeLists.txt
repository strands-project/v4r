cmake_minimum_required(VERSION 2.6)
SET(V4R_ORUTILS 1 CACHE BOOL "Utils for object recognition and classification" )
IF(V4R_ORUTILS)

find_package(Boost REQUIRED COMPONENTS filesystem regex)

include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

PROJECT(v4rORUtils)

SET(SOURCE_CPP
  integral_volume.cpp
  noise_model_based_cloud_integration.cpp
  noise_models.cpp
  pcl_visualization_utils.cpp
  segmentation_utils.cpp
#  organized_edge_detection.cpp
)

SET(SOURCE_VIEWER_CPP
)

SET(SOURCE_H
  common_data_structures.h
  integral_volume.h
  noise_model_based_cloud_integration.h
  noise_models.h
  pcl_opencv.h
  pcl_visualization_utils.h
  registration_utils.h
  segmentation_utils.h
  faat_3d_rec_framework_defines.h
#  organized_edge_detection.h
)


add_library(${PROJECT_NAME} SHARED ${SOURCE_H} ${SOURCE_CPP})
target_link_libraries(${PROJECT_NAME} ${OPENCV_LIBRARIES} ${PCL_LIBRARIES} v4rOREdgeDetector ${Boost_LIBRARIES})

v4r_add_library(${PROJECT_NAME} "${SOURCE_H}")


ENDIF(V4R_ORUTILS)



