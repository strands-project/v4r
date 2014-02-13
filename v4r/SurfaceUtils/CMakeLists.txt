SET(V4R_SURFACE_UTILS 1 CACHE BOOL "Utils for surface abstraction." )
IF(V4R_SURFACE_UTILS)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")

IF(V4R_TOMGINE)
  add_definitions(-DV4R_TOMGINE)
ENDIF(V4R_TOMGINE)
IF(V4R_TOMGINE_PCL)
  add_definitions(-DV4R_TOMGINE)
ENDIF(V4R_TOMGINE_PCL)

PROJECT(v4rSurfaceUtils)
SET(SOURCE_CPP
  AddGroundTruth.cpp
  BoundaryDetector.cpp
  ContourDetector.cpp
  FileSystem.cc
#  KinectData.cpp
  MaskDilationErosion.cpp
  SegmentEvaluation.cpp
  SurfaceModel.cpp
  Utils.cc
  EPBase.cpp
)

SET(SOURCE_H
  AddGroundTruth.h
  BoundaryDetector.h
  ContourDetector.h
  FileSystem.hh
#  KinectData.h
  MaskDilationErosion.h
  Relation.h
  SegmentEvaluation.h
  SurfaceModel.hpp
  Utils.hh
  EPBase.hpp
)

IF(V4R_PCL_ADD_ONS)

add_library(${PROJECT_NAME} SHARED ${SOURCE_H} ${SOURCE_CPP})
target_link_libraries(${PROJECT_NAME} ${OPENCV_LIBRARIES} ${PCL_LIBRARIES} v4rPCLAddOns v4rPCLAddOns v4rEPUtils v4rTomGine)
v4r_add_library(${PROJECT_NAME} "${SOURCE_H}")

ELSE(V4R_PCL_ADD_ONS)
  MESSAGE(STATUS "********************************************\n   v4r/SurfaceUtils: missing V4R_PCL_ADD_ONS\n   ********************************************\n")
ENDIF(V4R_PCL_ADD_ONS)

ENDIF(V4R_SURFACE_UTILS)

