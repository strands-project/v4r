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
  filesystem_utils.cpp
  integral_volume.cpp
  noise_model_based_cloud_integration.cpp
  noise_models.cpp
  pcl_visualization_utils.cpp
  segmentation_utils.cpp
#  organized_edge_detection.cpp
)

SET(SOURCE_VIEWER_CPP
  view_all_point_clouds_in_folder.cpp
)

SET(SOURCE_H
  common_data_structures.h
  filesystem_utils.h
  integral_volume.h
  noise_model_based_cloud_integration.h
  noise_models.h
  pcl_opencv.h
  pcl_visualization_utils.h
  registration_utils.h
  segmentation_utils.h
#  organized_edge_detection.h
)


add_library(${PROJECT_NAME} SHARED ${SOURCE_H} ${SOURCE_CPP})
target_link_libraries(${PROJECT_NAME} ${OPENCV_LIBRARIES} ${PCL_LIBRARIES} v4rOREdgeDetector)

v4r_add_library(${PROJECT_NAME} "${SOURCE_H}")

add_executable(view_all_point_clouds_in_folder ${SOURCE_VIEWER_CPP})
target_link_libraries(view_all_point_clouds_in_folder ${OPENCV_LIBRARIES} ${PCL_LIBRARIES} ${PROJECT_NAME}  ${Boost_LIBRARIES})


ENDIF(V4R_ORUTILS)



