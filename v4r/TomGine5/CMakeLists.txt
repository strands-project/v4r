cmake_minimum_required(VERSION 2.8)
set(V4R_TOMGINE_5 1 CACHE BOOL "OpenGL-GPU rendering, geometry and image processing library." )

IF(V4R_TOMGINE_5)

project(v4rTomGine5)

# pkg config
find_package(PkgConfig REQUIRED)

# find OpenGL
pkg_search_module(GL REQUIRED gl)
if(NOT GL_FOUND)
        message(FATAL "libgl not found, required by v4rTomGine5")
endif(NOT GL_FOUND)

add_definitions(-DLINUX)
add_definitions(-DGL_GLEXT_PROTOTYPES)

include_directories(${EIGEN3_INCLUDE_DIRS})


set(SOURCE_CPP 
	Camera.cpp
)

set(SOURCE_H
	Camera.h
)

add_library(${PROJECT_NAME} SHARED ${SOURCE_CPP})
target_link_libraries(${PROJECT_NAME} ${GL_LIBRARIES})

if(${V4R_BUILD})

  v4r_add_library(${PROJECT_NAME} "${SOURCE_H}")
  
else(${V4R_BUILD})

  install(DIRECTORY DESTINATION include/v4r/${TOMGINE_INCLUDE_NAME})
  install(FILES ${SOURCE_H} DESTINATION include/v4r/${TOMGINE_INCLUDE_NAME})
  install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)

endif(${V4R_BUILD})

ENDIF(V4R_TOMGINE_5)
