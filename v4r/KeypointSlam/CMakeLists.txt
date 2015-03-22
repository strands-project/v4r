SET(V4R_KEYPOINT_SLAM 1 CACHE BOOL "Keypoint based slam" )
IF(V4R_KEYPOINT_SLAM)

#FIND_PACKAGE (OpenCV)
include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})

find_package(Boost REQUIRED COMPONENTS thread program_options serialization regex filesystem system)
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

FIND_PACKAGE(Ceres)
IF(${Ceres_FOUND})
  INCLUDE_DIRECTORIES(${CERES_INCLUDES})
ELSE(${Ceres_FOUND})
  MESSAGE("\nATTENTION: Ceres solver is not installed!\n")
ENDIF(${Ceres_FOUND})

find_package(PkgConfig REQUIRED)
pkg_search_module(EIGEN3 REQUIRED eigen3)
include_directories(${EIGEN3_INCLUDE_DIRS})


PROJECT(v4rKeypointSlam)

SET(SOURCE_CPP
  ArticulatedObject.cc
  HomographyDetector.cc
  io.cc
  KeypointObjectRecognizerR2.cc
  KeypointPoseDetector.cc
  KeypointPoseDetectorRT.cc
  KeypointSlamRGBD2.cc
  LKHomographyTracker.cc
  LKPoseTracker.cc
  LKPoseTrackerRT.cc
  ObjectTrackerMono.cc
  Part.cc
  PartMotion6D.cc
  PartPlanarMotion3D.cc
  PartRotation1D.cc
  ProjBundleAdjuster.cc
  ProjLKPoseTrackerLM.cc
  ProjLKPoseTrackerR2.cc
  ProjLKPoseTrackerRT.cc
  Object.cpp
  KeyframeManagementRGBD2.cc
)
IF (V4R_TOMGINE_5)
SET(SOURCE_CPP ${SOURCE_CPP} ProjLKPoseTrackerLM_GL.cc)
ENDIF(V4R_TOMGINE_5)

SET(SOURCE_H
  ArticulatedObject.hh
  ArticulatedObject_serialization.hpp
  HomographyDetector.hh
  io.hh
  KeypointObjectRecognizerR2.hh
  KeypointPoseDetector.hh
  KeypointPoseDetectorRT.hh
  KeypointSlamRGBD2.hh
  LKHomographyTracker.hh
  ObjectTrackerMono.hh
  ProjLKPoseTrackerLM.hh
  ProjLKPoseTrackerLM_GL.hh
  LKPoseTracker.hh
  LKPoseTrackerRT.hh
  Part.hh
  PartMotion6D.hh
  PartPlanarMotion3D.hh
  PartRotation1D.hh
  KeyframeManagementRGBD2.hh
  ProjBundleAdjuster.hh
  ProjLKPoseTrackerR2.hh
  ProjLKPoseTrackerRT.hh
  ReprojectionError.hpp
  Object.hpp
)
IF (V4R_TOMGINE_5)
SET(SOURCE_H ${SOURCE_H} ProjLKPoseTrackerLM_GL.hh)
ENDIF(V4R_TOMGINE_5)


add_library(${PROJECT_NAME} SHARED ${SOURCE_H} ${SOURCE_CPP})
target_link_libraries(${PROJECT_NAME} ${OPENCV_LIBRARIES} v4rKeypointTools v4rKeypointBase)
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})

# .. in case of ceres is compiled static we must directly link it to the executable
get_target_property(LIBTYPE ${CERES_LIBRARIES} TYPE)
IF("${LIBTYPE}" STREQUAL "SHARED_LIBRARY")
  target_link_libraries(${PROJECT_NAME} ${CERES_LIBRARIES})
ENDIF("${LIBTYPE}" STREQUAL "SHARED_LIBRARY")

v4r_add_library(${PROJECT_NAME} "${SOURCE_H}")

ENDIF(V4R_KEYPOINT_SLAM)



