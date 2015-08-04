set(the_description "Recognition")
v4r_add_module(recognition v4r_common v4r_io v4r_features v4r_segmentation)

if(HAVE_SIFTGPU)
  v4r_include_directories(${SIFTGPU_INCLUDE_DIR})
  list(APPEND DEP_LIBS ${SIFTGPU_LIBRARIES})
endif()

if(HAVE_EDT)
  v4r_include_directories(${EDT_INCLUDE_DIR})
  list(APPEND DEP_LIBS ${EDT_LIBRARIES})
endif()

if(HAVE_PCL)
  v4r_include_directories(${PCL_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${PCL_LIBRARIES})
  add_definitions(${PCL_DEFINITIONS} ${Boost_LIB_DIAGNOSTIC_DEFINITIONS})
endif()

if(HAVE_OpenCV)
  v4r_include_directories(${OpenCV_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${OpenCV_LIBS})
endif()


include_directories(${CMAKE_CURRENT_LIST_DIR}/../../3rdparty/metslib) # for the time being - metslib still needs to be implemented

set(SOURCE_CPP_HV
  src/ghv_opt.cpp
  src/ghv.cpp
  src/hv_go_3D.cpp
)

SET(SOURCE_H_HV
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/ghv.h"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/impl/ghv.hpp"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/ghv_opt.h"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/impl/ghv_opt.hpp"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/hv_go_3D.h"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/hypotheses_verification.h"
)

SET(SOURCE_H_UTILS
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/metrics.h"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/vtk_model_sampling.h"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/voxel_based_correspondence_estimation.h"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/multiplane_segmentation.h"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/impl/voxel_based_correspondence_estimation.hpp"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/impl/multiplane_segmentation.hpp"
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
    "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/boost_graph_extension.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/boost_graph_visualization_extension.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/segmenter.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/multiview_object_recognizer_service.h"
#    "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/world_representation.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/segmenter.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/singleview_object_recognizer.h"
)

SET(SOURCE_H_DATA_SOURCES
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/source.h"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/unregistered_views_source.h"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/partial_pcd_source.h"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/impl/partial_pcd_source.hpp"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/model_only_source.h"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/registered_views_source.h"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/registered_views_pp_source.h"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/source2d.h"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/mesh_source.h"
)

SET(SOURCE_CPP_DATA_SOURCES
  src/mesh_source.cpp             
  src/model_only_source.cpp                           
  src/partial_pcd_source.cpp               
)

SET(SOURCE_H_PIPELINES
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/recognizer.h"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/global_nn_classifier.h" 
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/global_nn_recognizer_cvfh.h"         
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/local_recognizer.h"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/multi_pipeline_recognizer.h"               
  #linemod3d_recognizer.h"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/hough_grouping_local_recognizer.h"
  #implementations
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/impl/global_nn_classifier.hpp"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/impl/global_nn_recognizer_crh.hpp"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/impl/multi_pipeline_recognizer.hpp"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/impl/local_recognizer.hpp"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/impl/global_nn_recognizer_cvfh.hpp"
)

SET(SOURCE_CPP_PIPELINES
 src/multi_pipeline_recognizer.cpp               
 #linemod3d_recognizer.cpp #only works with PCL-trunk (not in releases)
 src/global_nn_classifier.cpp         
 src/local_recognizer.cpp
 src/global_nn_recognizer_cvfh.cpp                          
)


v4r_set_module_sources(HEADERS ${SOURCE_H_HV} ${SOURCE_H_MULTIVIEW} ${SOURCE_H_DATA_SOURCES} ${SOURCE_H_UTILS} ${SOURCE_H_PIPELINES}
                       SOURCES ${SOURCE_CPP_HV}  ${SOURCE_CPP_MULTIVIEW} ${SOURCE_CPP_UTILS} ${SOURCE_CPP_DATA_SOURCES} ${SOURCE_CPP_PIPELINES} )
v4r_module_include_directories()
v4r_create_module(${DEP_LIBS})

set_target_properties(${the_module} PROPERTIES LINK_INTERFACE_LIBRARIES "")
