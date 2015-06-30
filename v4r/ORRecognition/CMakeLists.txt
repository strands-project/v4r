SET(V4R_ORRECOGNITION 1 CACHE BOOL "Recognition module - HV and correspondence grouping" )
IF(V4R_ORRECOGNITION)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

SET(BUILD_GPU 0 CACHE BOOL "build gpu for recognition module")
if(BUILD_GPU)
	find_package(CUDA)
endif(BUILD_GPU)

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

SET(SOURCE_CPP_MULTIVIEW
    multiview_object_recognizer_service.cpp
    world_representation.cpp
    boost_graph_extension.cpp
    boost_graph_visualization_extension.cpp
    singleview_object_recognizer.cpp
)

SET(SOURCE_H_MULTIVIEW
    include/boost_graph_extension.h
    include/boost_graph_visualization_extension.h
    include/segmenter.h
    include/multiview_object_recognizer_service.h
    include/world_representation.h
    include/segmenter.h
    include/singleview_object_recognizer.h
)

if(CUDA_FOUND AND BUILD_GPU)
# Recent versions of cmake set CUDA_HOST_COMPILER to CMAKE_C_COMPILER which
# on OSX defaults to clang (/usr/bin/cc), but this is not a supported cuda
# compiler.  So, here we will preemptively set CUDA_HOST_COMPILER to gcc if
# that compiler exists in /usr/bin.  This will not override an existing cache
# value if the user has passed CUDA_HOST_COMPILER on the command line.
    if (NOT DEFINED CUDA_HOST_COMPILER AND CMAKE_C_COMPILER_ID STREQUAL "Clang" AND EXISTS /usr/bin/gcc)
      set(CUDA_HOST_COMPILER /usr/bin/gcc CACHE FILEPATH "Host side compiler used by NVCC")
      message(STATUS "Setting CMAKE_HOST_COMPILER to /usr/bin/gcc instead of ${CMAKE_C_COMPILER}.  See http://dev.pointclouds.org/issues/979")
    endif()

    if(MSVC11)
            # Setting this to true brakes Visual Studio builds.
            set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF CACHE BOOL "CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE")
    endif()

    message(" * CUDA ${CUDA_VERSION} was found")

    set(PCL_SRC_DIR "/home/thomas/Projects/pcl")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
						${CUDA_INCLUDE_DIRS}
                        ${CUDA_TOOLKIT_INCLUDE}
                        ${PCL_SRC_DIR}/gpu/octree/include
                        ${PCL_SRC_DIR}/gpu/containers/include
                        ${PCL_SRC_DIR}/gpu/octree/src/
                        ${PCL_SRC_DIR}/gpu/utils/include/)

    cuda_add_library(${PROJECT_NAME}_cuda_recognition SHARED ghv.cu ghv_cuda.h)
    add_library(${PROJECT_NAME} SHARED ${SOURCE_H_HV} ${SOURCE_CPP_HV} ${SOURCE_H_CG} ${SOURCE_CPP_CG} ghv_cuda_wrapper.h ghv_cuda_wrapper.cpp)
    target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${PROJECT_NAME}_cuda_recognition pcl_gpu_containers pcl_gpu_octree)
else(CUDA_FOUND AND BUILD_GPU)
    message("CUDA is not found or GPU build disabled")
    message("Not compiling CUDA libraries.")
endif(CUDA_FOUND AND BUILD_GPU)

    #include_directories(include)
    add_library(${PROJECT_NAME} SHARED ${SOURCE_H_HV} ${SOURCE_CPP_HV} ${SOURCE_H_CG} ${SOURCE_CPP_CG} ${SOURCE_H_MULTIVIEW} ${SOURCE_CPP_MULTIVIEW} ghv.h)
    target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${OpenCV_LIBS} v4rORUtils v4rORFramework v4rORRegistration siftgpu)
    v4r_add_library(${PROJECT_NAME} "${SOURCE_H_HV} ${SOURCE_H_CG} ${SOURCE_H_MULTIVIEW}")

ENDIF(V4R_ORRECOGNITION)
