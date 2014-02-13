SET(V4R_SURFACE_CLUSTERING 1 CACHE BOOL "Normals-clustering of point cloud." )
IF(V4R_SURFACE_CLUSTERING)

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

PROJECT(v4rSurfaceClustering)
SET(SOURCE_CPP
  #AssignPointsToPlanes.cc
  ClusterNormalsToPlanes.cc
  #GreedySelection.cc
  #MoSPlanes3D.cc
  #SubsamplePointCloud2.cc
  #UniformSurfaces.cpp
  ZAdaptiveNormals.cc
)

SET(SOURCE_H
  #AssignPointsToPlanes.hh
  ClusterNormalsToPlanes.hh
  #GreedySelection.hh
  #MoSPlanes3D.hh
  PPlane.h
  #SubsamplePointCloud2.hh
  #UniformSurfaces.h
  ZAdaptiveNormals.hh
)

add_library(${PROJECT_NAME} SHARED ${SOURCE_H} ${SOURCE_CPP})
target_link_libraries(${PROJECT_NAME} ${OPENCV_LIBRARIES} ${PCL_LIBRARIES} v4rRGBDSegment v4rTomGinePCL v4rEPUtils)
v4r_add_library(${PROJECT_NAME} "${SOURCE_H}")

ENDIF(V4R_SURFACE_CLUSTERING)

