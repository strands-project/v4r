cmake_minimum_required(VERSION 2.6)
SET(V4R_SEGMENTATION 1 CACHE BOOL "Segmentation algorithms" )
IF(V4R_SEGMENTATION)

find_package(Boost REQUIRED COMPONENTS filesystem regex)
find_package(OpenCV)
find_package(PCL REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS} -std=c++0x)

find_package(Eigen3 REQUIRED eigen3)
include_directories(${EIGEN3_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

PROJECT(v4rSegmentation)
SET(SOURCE_CPP
  Slic.cpp
  SlicRGBD.cpp
  SLICO.cpp
)

SET(SOURCE_H
  Slic.h
  SlicRGBD.h
  SLICO.h
)

add_library(${PROJECT_NAME} SHARED ${SOURCE_H} ${SOURCE_CPP})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${Boost_LIBRARIES})
v4r_add_library(${PROJECT_NAME} "${SOURCE_H}")

ENDIF(V4R_SEGMENTATION)

