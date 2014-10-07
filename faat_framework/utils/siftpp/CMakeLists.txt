SET(EXT_SIFTPP 1 CACHE BOOL "Vedaldi's siftpp and an opencv wrapper" )
IF(EXT_SIFTPP)

#FIND_PACKAGE (OpenCV)
include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})

find_package(PkgConfig REQUIRED)
pkg_search_module(EIGEN3 REQUIRED eigen3)
include_directories(${EIGEN3_INCLUDE_DIRS})

add_definitions(-DVL_LOWE_STRICT)
add_definitions(-DVL_USEFASTMATH)

PROJECT(v4rsiftpp)
SET(SOURCE_CPP
  PSiftpp.cc
  sift.cpp
)

SET(SOURCE_H
  PSiftpp.hh
  sift-conv.tpp
  sift.hpp
  sift.ipp
)

#add_library(siftgpu SHARED ${SOURCE_CPP})
#target_link_libraries(siftgpu GLEW glut IL GL X11)

add_library(siftpp SHARED ${SOURCE_H} ${SOURCE_CPP})
target_link_libraries(siftpp ${OpenCV_LIBS})
#v4r_add_library(${PROJECT_NAME} "${SOURCE_H}")

ENDIF(EXT_SIFTPP)

