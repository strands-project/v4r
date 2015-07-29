set(SUBSYS_NAME object_modelling)
set(SUBSYS_DESC "v4r object modelling module")
set(SUBSYS_DEPS v4r_common v4r_common_io v4r_registration v4r_common_keypoint siftgpu)
SET(V4R_OBJECT_MODELLING 1 CACHE BOOL "Object modelling module" )
IF(V4R_OBJECT_MODELLING)

FIND_PACKAGE (PCL REQUIRED)
INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

FIND_PACKAGE (OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})

PROJECT(v4r_${SUBSYS_NAME})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

SET(SOURCE_CPP
  src/do_learning.cpp
  src/visualization.cpp
)

SET(SOURCE_H
  include/v4r/${SUBSYS_NAME}/do_learning.h
  include/v4r/${SUBSYS_NAME}/model_view.h
)


include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library("${PREFIX_LIB}${SUBSYS_NAME}" SHARED ${SOURCE_H} ${SOURCE_CPP} )
target_link_libraries("${PREFIX_LIB}${SUBSYS_NAME}" ${PCL_LIBRARIES} ${OPENCV_LIBRARIES} ${SUBSYS_DEPS})
_install_library(${PREFIX_LIB}${SUBSYS_NAME} ${SOURCE_H} "/v4r/${SUBSYS_NAME}/")

ENDIF(V4R_OBJECT_MODELLING)
