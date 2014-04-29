SET(V4R_KEYPOINT_TOOLS 1 CACHE BOOL "Some general tools needed for modelling, tracking and recognition" )
IF(V4R_KEYPOINT_TOOLS)

FIND_PACKAGE(OpenCV)
include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})

find_package(PkgConfig REQUIRED)
pkg_search_module(EIGEN3 REQUIRED eigen3)
include_directories(${EIGEN3_INCLUDE_DIRS})

include_directories(${FLANN_INCLUDE_DIRS})
link_directories(${FLANN_LIBRARY_DIRS})

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

PROJECT(v4rKeypointTools)
SET(SOURCE_CPP
  ClusteringMeanShift.cc
  ClusteringRNN.cc
  EuclideanClustering3f.cc
  ImageTransformRANSAC.cc
  PlaneEstimationRANSAC.cc
  RigidTransformationRANSAC.cc
  SearchKdTreeFLANN2f.cc
  SearchKdTreeFLANN3f.cc
)

SET(SOURCE_H
  Clustering.hh
  ClusteringMeanShift.hh
  ClusteringRNN.hh
  convertDataMatrix.hpp
  getImageKPtoCV.hpp
  DataContainer.hh
  DataMatrix2D.hpp
  EuclideanClustering3f.hh
  Histogram.hpp
  Histogram2D.hpp
  ImageTransformRANSAC.hh
  invPose.hpp
  Matrix.hpp
  normalizedCrossCorrelation.hpp
  PointTypes.hpp
  PoseIO.hpp
  PlaneEstimationRANSAC.hh
  projectPointToImage.hpp
  RigidTransformationRANSAC.hh
  rotation.h
  ScopeTime.hpp
  SearchKdTreeFLANN2f.hh
  SearchKdTreeFLANN3f.hh
  SmartPtr.hpp
  SphereHistogram.h
  SphereHistogram.hpp
  toString.hpp
  triple.hpp
  Vector.hpp
  warpPatchHomography.hpp
)

add_library(${PROJECT_NAME} SHARED ${SOURCE_H} ${SOURCE_CPP})
target_link_libraries(${PROJECT_NAME} ${OPENCV_LIBRARIES} ${FLANN_LIBRARIES})
v4r_add_library(${PROJECT_NAME} "${SOURCE_H}")

ENDIF(V4R_KEYPOINT_TOOLS)

