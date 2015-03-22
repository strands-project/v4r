cmake_minimum_required(VERSION 2.6)
SET(V4R_UTILS 1 CACHE BOOL "Utility functions." )
IF(V4R_UTILS)

PROJECT(v4rUtils)

find_package(Boost REQUIRED COMPONENTS filesystem regex)
find_package(Eigen3 REQUIRED)

set(UTILS_CPP 
 distributions.cpp 
 #filesystemhdl.cpp
 filesystem_utils.cpp
 #timehdl.cpp
)
set(UTILS_H
 distributions.h 
 #filesystemhdl.h
 filesystem_utils.h
 #timehdl.h
 math.h
)

include_directories(${EIGEN_INCLUDE_DIRS})
add_library(${PROJECT_NAME} SHARED ${UTILS_CPP} ${UTILS_H})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})

v4r_add_library(${PROJECT_NAME} "${SOURCE_H}")

#INSTALL(FILES ${UTILS_H} DESTINATION include/v4r/utils)
#INSTALL(TARGETS v4rutils LIBRARY DESTINATION lib/)

ENDIF(V4R_UTILS)

