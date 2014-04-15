set(SUBSYS_NAME utils)
set(LIB_NAME ${PROJECT_NAME}_${SUBSYS_NAME})
set(SUBSYS_DESC "FAAT utils")
SET(FOO_LIBRARY_TYPE SHARED)

set(utils_incs
  include/faat_pcl/${SUBSYS_NAME}/miscellaneous.h
  include/faat_pcl/${SUBSYS_NAME}/integral_volume.h
  include/faat_pcl/${SUBSYS_NAME}/voxel_dist_transform.h
  include/faat_pcl/${SUBSYS_NAME}/noise_models.h
  include/faat_pcl/${SUBSYS_NAME}/noise_model_based_cloud_integration.h
  include/faat_pcl/${SUBSYS_NAME}/common_data_structures.h
  include/faat_pcl/${SUBSYS_NAME}/depth_inpainting.h
  include/faat_pcl/${SUBSYS_NAME}/filesystem_utils.h
  include/faat_pcl/${SUBSYS_NAME}/pcl_opencv.h
  include/faat_pcl/${SUBSYS_NAME}/registration_utils.h
  include/faat_pcl/${SUBSYS_NAME}/segmentation_utils.h
  include/faat_pcl/${SUBSYS_NAME}/EDT/3rdparty/propagation_distance_field.h
  include/faat_pcl/${SUBSYS_NAME}/EDT/3rdparty/distance_field.h
  include/faat_pcl/${SUBSYS_NAME}/EDT/3rdparty/voxel_grid.h
)

#SET_TARGET_PROPERTIES (faat_pcl_${SUBSYS_NAME} PROPERTIES DEFINE_SYMBOL  "FAAT_UTILS_EXPORTS" )
add_definitions(-DFAAT_UTILS_EXPORTS)

set(srcs
  src/integral_volume.cpp
  src/voxel_dist_transform.cpp
  src/EDT/3rdparty/propagation_distance_field.cpp
  src/noise_models.cpp
  src/noise_model_based_cloud_integration.cpp
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(${LIB_NAME} ${FOO_LIBRARY_TYPE} ${srcs} ${utils_incs})
target_link_libraries(${LIB_NAME} ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_executable(NMBasedCloudIntegration tests/test_NMBasedCloudIntegration.cpp)
target_link_libraries(NMBasedCloudIntegration ${PCL_LIBRARIES} ${OpenCV_LIBS} ${PROJECT_NAME}_utils boost_regex)

add_subdirectory(levmar-2.6)
add_subdirectory(sparselm-1.3)
add_subdirectory(SiftGPU)
