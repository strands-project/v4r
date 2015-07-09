SET(V4R_RECOGNITION 1 CACHE BOOL "Object recognition module" )
IF(V4R_RECOGNITION)

include_directories(${PCL_INCLUDE_DIRS} include/recognition)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

PROJECT(v4rRecognition)

SET(SOURCE_CPP_HV
  src/ghv_opt.cpp
  src/ghv.cpp
  src/hv_go_3D.cpp
)

SET(SOURCE_H_HV
  include/recognition/ghv.h
  include/recognition/impl/ghv.hpp
  include/recognition/hv_opt.h
  include/recognition/impl/ghv_opt.hpp
  include/recognition/hv_go_3D.h
  include/recognition/hypotheses_verification.h
  include/recognition/occlusion_reasoning.h
  include/recognition/impl/occlusion_reasoning.hpp
)

SET(SOURCE_H_CG
  include/recognition/correspondence_grouping.h
  include/recognition/correspondence_grouping.hpp
  include/recognition/geometric_consistency.h
  include/recognition/geometric_consistency.hpp
  include/recognition/hough_3d.h
  include/recognition/hough_3d.hpp
  include/recognition/graph_geometric_consistency.h
  include/recognition/graph_geometric_consistency.hpp
  include/recognition/kmeans.h
  include/recognition/kmeans.hpp
)

SET(SOURCE_CPP_CG
  src/geometric_consistency.cpp
  src/hough_3d.cpp
  src/graph_geometric_consistency.cpp
  src/kmeans.cpp
)

SET(SOURCE_CPP_MULTIVIEW
    src/multiview_object_recognizer_service.cpp
    src/world_representation.cpp
    src/boost_graph_extension.cpp
    src/boost_graph_visualization_extension.cpp
    src/singleview_object_recognizer.cpp
)

SET(SOURCE_H_MULTIVIEW
    include/recognition/boost_graph_extension.h
    include/recognition/boost_graph_visualization_extension.h
    include/recognition/segmenter.h
    include/recognition/multiview_object_recognizer_service.h
    include/recognition/world_representation.h
    include/recognition/segmenter.h
    include/recognition/singleview_object_recognizer.h
)

add_library(${PROJECT_NAME} SHARED ${SOURCE_H_HV} ${SOURCE_CPP_HV} ${SOURCE_H_CG} ${SOURCE_CPP_CG} ${SOURCE_H_MULTIVIEW} ${SOURCE_CPP_MULTIVIEW} ghv.h)
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${OpenCV_LIBS} siftgpu)
v4r_add_library(${PROJECT_NAME} "${SOURCE_H_HV} ${SOURCE_H_CG} ${SOURCE_H_MULTIVIEW}")

ENDIF(V4R_RECOGNITION)
