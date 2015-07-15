cmake_minimum_required(VERSION 2.6)
SET(APPS_SEGMENTATION 1 CACHE BOOL "App to test segmentation" )
set(SUBSYS_DEPS v4r_segmentation v4r_common_io v4r_common)
IF(APPS_SEGMENTATION)
IF(V4R_SEGMENTATION)
  find_package (OpenCV)
  find_package(PCL REQUIRED )
  find_package(Boost REQUIRED regex)
  
  link_directories(${PCL_LIBRARY_DIRS} ${Boost_LIBRARY_DIR})
  add_definitions(${PCL_DEFINITIONS})

  find_package(Eigen3 REQUIRED)

  include_directories(${EIGEN3_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

  add_executable(slicLabeling labelingSlic.cpp)
  target_link_libraries(slicLabeling ${OpenCV_LIBS} ${PCL_LIBRARIES} ${SUBSYS_DEPS} ${Boost_LIBRARIES})

ENDIF(V4R_SEGMENTATION)
ENDIF(APPS_SEGMENTATION)
