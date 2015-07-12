set(SUBSYS_NAME recognition)
set(SUBSYS_DESC "V4R object recognition module")
set(SUBSYS_DEPS v4r_common v4r_common_features v4r_common_io v4r_segmentation EDT siftgpu)
set(V4R_RECOGNITION 1 CACHE BOOL "Object recognition module" )
IF(V4R_RECOGNITION)


FIND_PACKAGE (PCL)
INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

PROJECT(v4r_${SUBSYS_NAME})

SET(SOURCE_CPP_HV
  src/ghv_opt.cpp
  src/ghv.cpp
  src/hv_go_3D.cpp
)

SET(SOURCE_H_HV
  include/v4r/${SUBSYS_NAME}/ghv.h
  include/v4r/${SUBSYS_NAME}/impl/ghv.hpp
  include/v4r/${SUBSYS_NAME}/ghv_opt.h
  include/v4r/${SUBSYS_NAME}/impl/ghv_opt.hpp
  include/v4r/${SUBSYS_NAME}/hv_go_3D.h
  include/v4r/${SUBSYS_NAME}/hypotheses_verification.h
)



SET(SOURCE_H_UTILS
  include/v4r/${SUBSYS_NAME}/metrics.h
  include/v4r/${SUBSYS_NAME}/vtk_model_sampling.h
  include/v4r/${SUBSYS_NAME}/voxel_based_correspondence_estimation.h
  include/v4r/${SUBSYS_NAME}/multiplane_segmentation.h
  include/v4r/${SUBSYS_NAME}/impl/voxel_based_correspondence_estimation.hpp
  include/v4r/${SUBSYS_NAME}/impl/multiplane_segmentation.hpp
)

SET(SOURCE_CPP_UTILS
  src/voxel_based_correspondence_estimation.cpp
  src/multiplane_segmentation.cpp
)


SET(SOURCE_CPP_MULTIVIEW
    src/multiview_object_recognizer_service.cpp
#    src/world_representation.cpp
    src/boost_graph_extension.cpp
    src/boost_graph_visualization_extension.cpp
    src/singleview_object_recognizer.cpp
)

SET(SOURCE_H_MULTIVIEW
    include/v4r/${SUBSYS_NAME}/boost_graph_extension.h
    include/v4r/${SUBSYS_NAME}/boost_graph_visualization_extension.h
    include/v4r/${SUBSYS_NAME}/segmenter.h
    include/v4r/${SUBSYS_NAME}/multiview_object_recognizer_service.h
#    include/v4r/${SUBSYS_NAME}/world_representation.h
    include/v4r/${SUBSYS_NAME}/segmenter.h
    include/v4r/${SUBSYS_NAME}/singleview_object_recognizer.h
)

SET(SOURCE_H_DATA_SOURCES
  include/v4r/${SUBSYS_NAME}/source.h
  include/v4r/${SUBSYS_NAME}/unregistered_views_source.h
  include/v4r/${SUBSYS_NAME}/partial_pcd_source.h  
  include/v4r/${SUBSYS_NAME}/impl/partial_pcd_source.hpp
  include/v4r/${SUBSYS_NAME}/model_only_source.h
  include/v4r/${SUBSYS_NAME}/registered_views_source.h
  include/v4r/${SUBSYS_NAME}/registered_views_pp_source.h
  include/v4r/${SUBSYS_NAME}/source2d.h
  include/v4r/${SUBSYS_NAME}/mesh_source.h
)

SET(SOURCE_CPP_DATA_SOURCES
  src/mesh_source.cpp             
  src/model_only_source.cpp                           
  src/partial_pcd_source.cpp               
)

SET(SOURCE_H_PIPELINES
  include/v4r/${SUBSYS_NAME}/recognizer.h
  include/v4r/${SUBSYS_NAME}/global_nn_classifier.h   
  include/v4r/${SUBSYS_NAME}/global_nn_recognizer_cvfh.h          
  include/v4r/${SUBSYS_NAME}/local_recognizer.h
  include/v4r/${SUBSYS_NAME}/multi_pipeline_recognizer.h                 
  #linemod3d_recognizer.h
  include/v4r/${SUBSYS_NAME}/hough_grouping_local_recognizer.h
  #implementations
  include/v4r/${SUBSYS_NAME}/impl/global_nn_classifier.hpp
  include/v4r/${SUBSYS_NAME}/impl/global_nn_recognizer_crh.hpp 
  include/v4r/${SUBSYS_NAME}/impl/multi_pipeline_recognizer.hpp
  include/v4r/${SUBSYS_NAME}/impl/local_recognizer.hpp
  include/v4r/${SUBSYS_NAME}/impl/global_nn_recognizer_cvfh.hpp
)

SET(SOURCE_CPP_PIPELINES
 src/multi_pipeline_recognizer.cpp               
 #linemod3d_recognizer.cpp #only works with PCL-trunk (not in releases)
 src/global_nn_classifier.cpp         
 src/local_recognizer.cpp
 src/global_nn_recognizer_cvfh.cpp                          
)

include_directories(${PCL_INCLUDE_DIRS} "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(${PROJECT_NAME} SHARED ${SOURCE_H_HV} ${SOURCE_CPP_HV} ${SOURCE_H_MULTIVIEW} ${SOURCE_CPP_MULTIVIEW} 
          ${SOURCE_H_UTILS} ${SOURCE_CPP_UTILS} ${SOURCE_CPP_MULTIVIEW} ${SOURCE_H_MULTIVIEW} ${SOURCE_H_DATA_SOURCES} ${SOURCE_CPP_DATA_SOURCES} 
          ${SOURCE_H_PIPELINES} ${SOURCE_CPP_PIPELINES})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${SUBSYS_DEPS})
v4r_add_library(${PROJECT_NAME}  "${SUBSYS_NAME}" "${SOURCE_H_HV} ${SOURCE_H_MULTIVIEW} ${SOURCE_H_UTILS} ${SOURCE_H_DATA_SOURCES} ${SOURCE_H_PIPELINES}")

ENDIF(V4R_RECOGNITION)
