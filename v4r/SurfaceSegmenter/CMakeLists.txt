cmake_minimum_required (VERSION 2.6)
SET(V4R_SURFACE_SEGMENTER 1 CACHE BOOL "Surface segmenter framework" )
IF(V4R_SURFACE_SEGMENTER)

IF(V4R_GRAPH_CUT)
IF(V4R_SVM)
IF(V4R_SURFACE_RELATIONS)
IF(V4R_SURFACE_MODELING)
IF(V4R_SURFACE_CLUSTERING)
IF(V4R_SURFACE_UTILS)

include_directories(.)
include_directories(${OPENCV_INCLUDE_DIRS})

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

PROJECT(v4rSurfaceSegmenter)
SET(SOURCE_CPP
#   ModelAbstractor.cpp
#   ModelRefinement.cpp
#   PreSegmenter.cpp
#   Segmenter.cpp
#   SegmenterLight.cpp
  #SegmenterThread.cpp
  segmentation.cpp
)

SET(SOURCE_H
#   ModelAbstractor.h
#   ModelRefinement.h
#   PreSegmenter.h
#   Segmenter.h
#   SegmenterLight.h
  #SegmenterThread.h
  segmentation.hpp
)

add_library(${PROJECT_NAME} SHARED ${SOURCE_H} ${SOURCE_CPP})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} v4rSurfaceUtils v4rSurfaceClustering v4rEPUtils v4rSurfaceModeling v4rSurfaceRelations v4rsvm v4rGraphCut v4rAttentionModule rt)
v4r_add_library(${PROJECT_NAME} "${SOURCE_H}")

ELSE(V4R_SURFACE_UTILS)
  MESSAGE(STATUS "********************************************\n   Apps: Segmenter warning: missing V4R_SURFACE_UTILS\n   ********************************************\n")
ENDIF(V4R_SURFACE_UTILS)

ELSE(V4R_SURFACE_CLUSTERING)
  MESSAGE(STATUS "********************************************\n   Apps: Segmenter warning: missing V4R_SURFACE_CLUSTERING\n   ********************************************\n")
ENDIF(V4R_SURFACE_CLUSTERING)

ELSE(V4R_SURFACE_MODELING)
  MESSAGE(STATUS "********************************************\n   Apps: Segmenter warning: missing V4R_SURFACE_MODELING\n   ********************************************\n")
ENDIF(V4R_SURFACE_MODELING)

ELSE(V4R_SURFACE_RELATIONS)
  MESSAGE(STATUS "********************************************\n   Apps: Segmenter warning: missing V4R_SURFACE_RELATIONS\n   ********************************************\n")
ENDIF(V4R_SURFACE_RELATIONS)

ELSE(V4R_SVM)
  MESSAGE(STATUS "********************************************\n   Apps: Segmenter warning: missing V4R_SVM\n   ********************************************\n")
ENDIF(V4R_SVM)

ELSE(V4R_GRAPH_CUT)
  MESSAGE(STATUS "********************************************\n   Apps: Segmenter warning: missing V4R_GRAPH_CUT\n   ********************************************\n")
ENDIF(V4R_GRAPH_CUT)

ENDIF(V4R_SURFACE_SEGMENTER)
