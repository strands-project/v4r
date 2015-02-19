SET(V4R_ORRECOGNITION 1 CACHE BOOL "Recognition module - HV and correspondence grouping" )
IF(V4R_ORRECOGNITION)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


OPTION(BUILD_GPU "Build the GPU-related subsystems" 1)

PROJECT(v4rORRecognition)

SET(SOURCE_CPP_HV
  ghv_opt.cpp
  ghv.cpp
  hv_go_3D.cpp
)

SET(SOURCE_H_HV
  ghv.h
  ghv.hpp
  ghv_opt.h
  ghv_opt.hpp
  hv_go_3D.h
  hypotheses_verification.h
  occlusion_reasoning.h
  occlusion_reasoning.hpp
)

SET(SOURCE_H_CG
  correspondence_grouping.h
  correspondence_grouping.hpp
  geometric_consistency.h
  geometric_consistency.hpp
  hough_3d.h
  hough_3d.hpp
  graph_geometric_consistency.h
  graph_geometric_consistency.hpp
  kmeans.h
  kmeans.hpp
)

SET(SOURCE_CPP_CG
  geometric_consistency.cpp
  hough_3d.cpp
  graph_geometric_consistency.cpp
  kmeans.cpp
)

if(BUILD_GPU)
    set(PCL_SRC_DIR "/home/thomas/Projects/pcl")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                        ${CUDA_TOOLKIT_INCLUDE}
                        ${PCL_SRC_DIR}/gpu/octree/include
                        ${PCL_SRC_DIR}/gpu/containers/include
                        ${PCL_SRC_DIR}/gpu/octree/src/
                        ${PCL_SRC_DIR}/gpu/utils/include/)
    cuda_add_library(${PROJECT_NAME}_cuda_recognition SHARED ghv.cu ghv_cuda.h)
    add_library(${PROJECT_NAME} SHARED ${SOURCE_H_HV} ${SOURCE_CPP_HV} ${SOURCE_H_CG} ${SOURCE_CPP_CG} ghv_cuda_wrapper.h ghv_cuda_wrapper.cpp)
    target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${PROJECT_NAME}_cuda_recognition pcl_gpu_containers pcl_gpu_octree)
else(BUILD_GPU)
    include_directories(include)
    add_library(${PROJECT_NAME} SHARED ${SOURCE_H_HV} ${SOURCE_CPP_HV} ${SOURCE_H_CG} ${SOURCE_CPP_CG})
    target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})
endif(BUILD_GPU)

    v4r_add_library(${PROJECT_NAME} "${SOURCE_H_HV} ${SOURCE_H_CG}")

ENDIF(V4R_ORRECOGNITION)



