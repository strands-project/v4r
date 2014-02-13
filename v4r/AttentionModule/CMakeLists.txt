SET(V4R_ATTENTION_MODULE 1 CACHE BOOL "Attention module." )
IF(V4R_ATTENTION_MODULE)

FIND_PACKAGE (OpenCV QUIET)
include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})

PROJECT(v4rAttentionModule)

FIND_PACKAGE(PCL REQUIRED)
IF(${PCL_FOUND})
  include_directories(${PCL_INCLUDE_DIRS})
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})

  MESSAGE("PCL_INCLUDE_DIRS: ${PCL_INCLUDE_DIRS}")
  MESSAGE("PCL_LIBRARY_DIRS: ${PCL_LIBRARY_DIRS}")
  MESSAGE("PCL_LIBRARY_DIRS: ${PCL_LIBRARY_DIRS}")

  SET(SOURCE_CPP_PCL
      SurfaceHeightMap.cpp
      RelativeSurfaceOrientationMap.cpp
      SurfaceCurvatureMap.cpp
#       Symmetry3DMap2.cpp
  )
  SET(SOURCE_H_PCL
      SurfaceHeightMap.hpp
      RelativeSurfaceOrientationMap.hpp
      SurfaceCurvatureMap.hpp
#       Symmetry3DMap2.hpp
  )

ENDIF(${PCL_FOUND})

SET(SOURCE_CPP
    WTA.cpp
    pyramids.cpp
    ColorMap.cpp
    OrientationMap.cpp
    SymmetryMap.cpp
    IKNSaliencyMap.cpp
    FrintropSaliencyMap.cpp
    LocationMap.cpp
    MapsCombination.cpp
    TJ.cpp
    MSR.cpp
)
SET(SOURCE_H
    WTA.hpp
    AttentionModuleErrors.hpp
    pyramids.hpp
    ColorMap.hpp
    OrientationMap.hpp
    SymmetryMap.hpp
    IKNSaliencyMap.hpp
    FrintropSaliencyMap.hpp
    LocationMap.hpp
    MapsCombination.hpp
    AttentionModule.hpp
    headers.hpp
    TJ.hpp
    MSR.hpp
)

# add_subdirectory(example)

add_library(${PROJECT_NAME} SHARED ${SOURCE_H} ${SOURCE_CPP} ${SOURCE_CPP_PCL} ${SOURCE_H_PCL})
target_link_libraries(${PROJECT_NAME} ${OPENCV_LIBRARIES} v4rEPUtils)
v4r_add_library(${PROJECT_NAME} "${SOURCE_H};${SOURCE_H_PCL}")



ENDIF(V4R_ATTENTION_MODULE)
