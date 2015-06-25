SET(V4R_OCTOPDOWN_SEGMENTER 1 CACHE BOOL "Segmentation and stuff" )
IF(V4R_OCTOPDOWN_SEGMENTER)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

MESSAGE(STATUS ${EIGEN_INCLUDE_DIRS})
if(SUITESPARSE_FOUND)
  include_directories(${SUITESPARSE_INCLUDE_DIRS})
endif()

MESSAGE(STATUS ${EIGEN_INCLUDE_DIRS})

PROJECT(v4rOCTopDownSegmenter)

SET(SOURCE_CPP_PRESEGMENTER
  edge_based_presegmenter.cpp
  sv_ms_presegmenter.cpp
  ms_utils.cpp
  mv_MS_presegmenter.cpp
)

SET(SOURCE_H_PRESEGMENTER
  pre_segmenter.h
  edge_based_presegmenter.h
  sv_ms_presegmenter.h
  mv_MS_presegmenter.h
  ms_utils.h
)

add_definitions(-DDEBUG)

add_library(${PROJECT_NAME} SHARED ${SOURCE_H_PRESEGMENTER} ${SOURCE_CPP_PRESEGMENTER})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${OpenCV_LIBS} v4ron_nurbs v4rSegmentation)
v4r_add_library(${PROJECT_NAME} "${SOURCE_H_PRESEGMENTER} ${SOURCE_CPP_PRESEGMENTER}")

add_executable(app_test app_test.cpp)
target_link_libraries(app_test ${PCL_LIBRARIES} ${OpenCV_LIBS} v4rOCTopDownSegmenter v4ron_nurbs v4rORUtils v4rSegmentation boost_regex )

add_executable(pcd2detph pcd2depth.cpp)
target_link_libraries(pcd2detph ${PCL_LIBRARIES} ${OpenCV_LIBS} boost_regex )

ENDIF(V4R_OCTOPDOWN_SEGMENTER)



