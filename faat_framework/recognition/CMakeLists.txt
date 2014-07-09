set(SUBSYS_NAME recognition)
set(LIB_NAME ${PROJECT_NAME}_${SUBSYS_NAME})
set(SUBSYS_DESC "FAAT recognition")
#set(SUBSYS_DEPS_FAAT_PCL )
SET(FOO_LIBRARY_TYPE SHARED)

SET(FAAT_PCL_HV_USE_GPU 0)

if(FAAT_PCL_USE_CUDA)

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                        ${CUDA_TOOLKIT_INCLUDE}
                        ${PCL_SRC_DIR}/gpu/octree/include
                        ${PCL_SRC_DIR}/gpu/containers/include
                        ${PCL_SRC_DIR}/gpu/octree/src/
                        ${PCL_SRC_DIR}/gpu/utils/include/)

    cuda_add_library(${PROJECT_NAME}_cuda_recognition SHARED src/hv/cuda/ghv.cu include/faat_pcl/${SUBSYS_NAME}/hv/cuda/ghv.h)

    add_library(${PROJECT_NAME}_recognition_with_cuda ${FOO_LIBRARY_TYPE} include/faat_pcl/${SUBSYS_NAME}/hv/ghv_cuda_wrapper.h src/hv/ghv_cuda_wrapper.cpp)
    target_link_libraries(${PROJECT_NAME}_recognition_with_cuda ${PCL_LIBRARIES} ${PROJECT_NAME}_cuda_recognition pcl_gpu_containers pcl_gpu_octree)

endif(FAAT_PCL_USE_CUDA)

set(hv_incs
  include/faat_pcl/${SUBSYS_NAME}/hv/hv_go.h
  include/faat_pcl/${SUBSYS_NAME}/hv/hv_go_1.h
  include/faat_pcl/${SUBSYS_NAME}/hv/hv_go_3D.h
  include/faat_pcl/${SUBSYS_NAME}/hv/hv_go_opt.h
  include/faat_pcl/${SUBSYS_NAME}/hv/hypotheses_verification.h
  include/faat_pcl/${SUBSYS_NAME}/hv/occlusion_reasoning.h
)

set(hv_impl_incs
  include/faat_pcl/${SUBSYS_NAME}/impl/hv/hv_go.hpp
  include/faat_pcl/${SUBSYS_NAME}/impl/hv/hv_go_1.hpp
  include/faat_pcl/${SUBSYS_NAME}/impl/hv/hv_go_opt.hpp
  include/faat_pcl/${SUBSYS_NAME}/impl/hv/occlusion_reasoning.hpp
)

set(srcs
  src/hv/hv_go.cpp
  src/hv/hv_go_3D.cpp
  src/hv/hv_go_1.cpp
  src/hv/hv_go_opt.cpp
  src/cg/graph_geometric_consistency.cpp
  src/cg/multi_object_graph_CG.cpp
)

set(cg_incs
  include/faat_pcl/${SUBSYS_NAME}/cg/graph_geometric_consistency.h
  include/faat_pcl/${SUBSYS_NAME}/cg/multi_object_graph_CG.h
)

set(cg_impl_incs
  include/faat_pcl/${SUBSYS_NAME}/impl/cg/graph_geometric_consistency.hpp
  include/faat_pcl/${SUBSYS_NAME}/impl/cg/cliques.hpp
  include/faat_pcl/${SUBSYS_NAME}/impl/cg/multi_object_graph_CG.hpp
)

add_definitions(-DFAAT_REC_EXPORTS)

if(FAAT_PCL_HV_USE_GPU)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                        ${CMAKE_CURRENT_SOURCE_DIR}/../utils/include
                        /home/aitor/pcl_git_fork/pcl/gpu/octree/include
                        /home/aitor/pcl_git_fork/pcl/gpu/containers/include)

    add_library(${LIB_NAME} ${FOO_LIBRARY_TYPE} ${srcs} ${hv_incs} ${hv_impl_incs} ${cg_incs} ${cg_impl_incs})
    target_link_libraries(${LIB_NAME} ${PCL_LIBRARIES} ${PROJECT_NAME}_utils pcl_gpu_octree)
else(FAAT_PCL_HV_USE_GPU)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                        ${CMAKE_CURRENT_SOURCE_DIR}/../utils/include)

    add_library(${LIB_NAME} ${FOO_LIBRARY_TYPE} ${srcs} ${hv_incs} ${hv_impl_incs} ${cg_incs} ${cg_impl_incs})
    target_link_libraries(${LIB_NAME} ${PCL_LIBRARIES} ${PROJECT_NAME}_utils)
endif(FAAT_PCL_HV_USE_GPU)

add_executable(queens_to_or_evaluator_format src/scripts/queens_to_or_evaluator_format.cpp)  
target_link_libraries(queens_to_or_evaluator_format ${PCL_LIBRARIES})

add_executable(eccv_to_or_evaluator_format src/scripts/eccv_to_or_evaluator_format.cpp)  
target_link_libraries(eccv_to_or_evaluator_format ${PCL_LIBRARIES})

add_executable(mian_to_or_evaluator_format src/scripts/mian_to_or_evaluator_format.cpp)  
target_link_libraries(mian_to_or_evaluator_format ${PCL_LIBRARIES})

add_executable(willow_challenge_to_or_evaluator_format src/scripts/willow_challenge_to_or_evaluator_format.cpp)
target_link_libraries(willow_challenge_to_or_evaluator_format ${PCL_LIBRARIES})



add_executable(ops_to_pcd src/scripts/ops_to_pcd.cpp)  
target_link_libraries(ops_to_pcd ${PCL_LIBRARIES})

add_executable(vtk2ply src/scripts/vtk2ply.cpp)  
target_link_libraries(vtk2ply ${PCL_LIBRARIES})

#add_executable(local_recognition_ply tests/scene_local_training_recognition.cpp)  
#target_link_libraries(local_recognition_ply ${PCL_LIBRARIES} faat_pcl_recognition faat_pcl_utils faat_pcl_segmentation)
