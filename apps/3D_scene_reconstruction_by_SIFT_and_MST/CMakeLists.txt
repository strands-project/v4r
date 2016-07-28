SET(3D_SCENE_RECONSTRUCTION_BY_SIFT_AND_MST_DEPS v4r_core v4r_registration v4r_io v4r_common v4r_features)
v4r_check_dependencies(${3D_SCENE_RECONSTRUCTION_BY_SIFT_AND_MST_DEPS})

if(NOT V4R_DEPENDENCIES_FOUND)
  message(***3D_SCENE_RECONSTRUCTION_BY_SIFT_AND_MST_DEPS does not meed dependencies*****)
  return()
endif()

find_package(PkgConfig REQUIRED)
pkg_search_module(GL REQUIRED gl)

v4r_include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
v4r_include_modules(${3D_SCENE_RECONSTRUCTION_BY_SIFT_AND_MST_DEPS})

if(WIN32)
  link_directories(${CMAKE_CURRENT_BINARY_DIR})
endif()

if(HAVE_PCL)
  v4r_include_directories(${PCL_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${PCL_LIBRARIES})
endif()

if(HAVE_OPENCV)
  v4r_include_directories(${OpenCV_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${OpenCV_LIBS})
endif()

if(HAVE_SIFTGPU)
  v4r_include_directories(${SIFTGPU_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${SIFTGPU_LIBRARIES})
endif()

if(HAVE_OPENGL)
  v4r_include_directories(${OPENGL_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${OPENGL_LIBRARIES})
endif()

add_executable(3dReconstructionBySiftAndMST main.cpp)
target_link_libraries(3dReconstructionBySiftAndMST ${3D_SCENE_RECONSTRUCTION_BY_SIFT_AND_MST_DEPS} ${DEP_LIBS})

INSTALL(TARGETS 3dReconstructionBySiftAndMST
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

