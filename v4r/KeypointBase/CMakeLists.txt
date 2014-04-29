SET(V4R_KEYPOINT_BASE 1 CACHE BOOL "Keypoint related data types, detectors, descriptors" )
IF(V4R_KEYPOINT_BASE)

FIND_PACKAGE (OpenCV)
include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})

find_package(PkgConfig REQUIRED)
pkg_search_module(EIGEN3 REQUIRED eigen3)
include_directories(${EIGEN3_INCLUDE_DIRS})
#include_directories("/usr/include/eigen3")

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

PROJECT(v4rKeypointBase)
SET(SOURCE_CPP
  ComputeImGDescOrientations.cc
  ComputeImGradientDescriptors.cc
  Feature.cc
  FeatureDetector_K_HARRIS.cc
  FeatureDetector_KD_CVSURF.cc
  FeatureDetector_KD_CVSIFT.cc
  FeatureDetector_KD_SIFTGPU.cc
  FeatureDetector_D_FREAK.cc
  FeatureDetector_KD_ORB.cc
  FeatureDetector_KD_FAST_IMGD.cc
  ImGradientDescriptor.cc
  ImGDescOrientation.cc
  LinkedKeypoint.cc
  OptimizePointsMultiViewGD.cc
  PSiftGPU.cc
  RefinePatchLocationLK.cc
  RefineProjectedPointLocationLK.cc
)

SET(SOURCE_H
  ComputeImGDescOrientations.hh
  ComputeImGradientDescriptors.hh
  Feature.hh
  FeatureMatch.hh
  FeatureDetector.hh
  FeatureDetector_K_HARRIS.hh
  FeatureDetector_KD_CVSURF.hh
  FeatureDetector_KD_CVSIFT.hh
  FeatureDetector_KD_SIFTGPU.hh
  FeatureDetector_D_FREAK.hh
  FeatureDetector_KD_ORB.hh
  FeatureDetector_KD_FAST_IMGD.hh
  ImGradientDescriptor.hh
  ImGDescOrientation.hh
  LinkedKeypoint.hh
  MultiViewPatchError.hpp
  OptimizePointsMultiViewGD.hh
  PSiftGPU.hh
  RefinePatchLocationLK.hh
  RefineProjectedPointLocationLK.hh
)

add_library(${PROJECT_NAME} SHARED ${SOURCE_H} ${SOURCE_CPP})
target_link_libraries(${PROJECT_NAME} ${OPENCV_LIBRARIES} v4rsiftgpu v4rKeypointTools)
v4r_add_library(${PROJECT_NAME} "${SOURCE_H}")

ENDIF(V4R_KEYPOINT_BASE)

