set(SUBSYS_NAME segmentation)
set(LIB_NAME faat_pcl_${SUBSYS_NAME})
set(SUBSYS_DESC "FAAT segmentation")
set(SUBSYS_DEPS_PCL common io search kdtree octree features filters registration sample_consensus recognition)
#set(SUBSYS_DEPS_FAAT_PCL )
SET(FOO_LIBRARY_TYPE SHARED)

set(objectness_incs
  include/faat_pcl/${SUBSYS_NAME}/objectness_3d/objectness_common.h
  include/faat_pcl/${SUBSYS_NAME}/objectness_3d/objectness_3D.h
  include/faat_pcl/${SUBSYS_NAME}/objectness_3d/bbox_optimizer.h
)

set(objectness_srcs
  src/objectness_3d/objectness_3D.cpp
)

#create cuda library and link it to the non-cuda one         

if(FAAT_PCL_USE_CUDA)

  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include 
                    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/include 
                    ${CMAKE_CURRENT_SOURCE_DIR}/test/include/
                    ${CUDA_TOOLKIT_INCLUDE})
                                                                      
  cuda_add_library(faat_pcl_cuda_segmentation SHARED src/objectness_3d/cuda/objectness_3D_cuda.cu include/faat_pcl/${SUBSYS_NAME}/objectness_3d/cuda/cuda_objectness_3D.h)
  
  add_library(${LIB_NAME} ${FOO_LIBRARY_TYPE} ${objectness_srcs} ${objectness_incs})
  target_link_libraries(${LIB_NAME} ${PCL_LIBRARIES} faat_pcl_utils faat_pcl_cuda_segmentation)
  
  add_executable(objectness_3d test/objectness_3d.cpp)  
  target_link_libraries(objectness_3d ${PCL_LIBRARIES} faat_pcl_recognition faat_pcl_utils faat_pcl_segmentation)
  
  add_executable(objectness_3d_evaluation test/objectness_evaluation.cpp)  
  target_link_libraries(objectness_3d_evaluation ${PCL_LIBRARIES} faat_pcl_recognition faat_pcl_utils faat_pcl_segmentation)
endif(FAAT_PCL_USE_CUDA)  