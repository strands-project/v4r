SET(OR_DEPS v4r_core v4r_common v4r_io v4r_features v4r_keypoints v4r_segmentation v4r_recognition v4r_ml)
v4r_check_dependencies(${OR_DEPS})

if(NOT V4R_DEPENDENCIES_FOUND)
  message(***MVObjectRecognizer does not meet dependencies*****)
  return()
endif()

v4r_include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
v4r_include_modules(${OR_DEPS})

if(HAVE_PCL)
  v4r_include_directories(${PCL_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${PCL_LIBRARIES})
endif()

if(HAVE_OPENCV)
  v4r_include_directories(${OpenCV_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${OpenCV_LIBS})
endif()

if(HAVE_EDT)
  v4r_include_directories(${EDT_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${EDT_LIBRARIES})
endif()

if(HAVE_CERES)
  v4r_include_directories(${CERES_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${CERES_LIBRARIES})
endif()

if(HAVE_LIBSVM)
  v4r_include_directories(${LIBSVM_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${LIBSVM_LIBRARIES})
endif()

if(HAVE_GLOG)
  v4r_include_directories(${GLOG_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${GLOG_LIBRARIES})
endif()

if(HAVE_BOOST)
  v4r_include_directories(${BOOST_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${BOOST_LIBRARIES})
endif()

set(incs visualization.h)
set(srcs visualization.cpp main.cpp)

add_executable(MVObjectRecognizer ${incs} ${srcs})
target_link_libraries(MVObjectRecognizer ${OR_DEPS} ${DEP_LIBS})

INSTALL(TARGETS MVObjectRecognizer
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
