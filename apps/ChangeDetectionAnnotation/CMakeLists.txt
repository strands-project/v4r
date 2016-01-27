SET(CHANGE_DETECTION_ANNOTATION_DEPS v4r_core v4r_io v4r_common)
v4r_check_dependencies(${CHANGE_DETECTION_ANNOTATION_DEPS})

if(NOT V4R_DEPENDENCIES_FOUND)
  message(***ChangeDetectionAnnotation does not meed dependencies*****)
  return()
endif()

v4r_include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
v4r_include_modules(${CHANGE_DETECTION_ANNOTATION_DEPS})

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

if(HAVE_BOOST)
  v4r_include_directories(${BOOST_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${BOOST_LIBRARIES})
endif()

if(HAVE_GLOG)
  v4r_include_directories(${GLOG_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${GLOG_LIBRARIES})
endif()

add_executable(change_annotator change_annotator.cpp)
target_link_libraries(change_annotator ${CHANGE_DETECTION_ANNOTATION_DEPS} ${DEP_LIBS})

add_executable(create_penalties create_penalties.cpp)
target_link_libraries(create_penalties ${CHANGE_DETECTION_ANNOTATION_DEPS} ${DEP_LIBS})

INSTALL(TARGETS change_annotator create_penalties
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
