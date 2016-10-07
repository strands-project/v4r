SET(AttentionSegmentation_DEPS v4r_core v4r_common v4r_attention_segmentation )
v4r_check_dependencies(${AttentionSegmentation_DEPS})

if(NOT V4R_DEPENDENCIES_FOUND)
  message(***AttentionSegmentation does not meet dependencies*****)
  return()
endif()

v4r_include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
v4r_include_directories(${OPENNURBS_INCLUDE_DIRS})
v4r_include_directories(${ON_NURBS_INCLUDE_DIRS})
v4r_include_modules(${AttentionSegmentation_DEPS})

if(HAVE_PCL)
  v4r_include_directories(${PCL_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${PCL_LIBRARIES})
endif()

if(HAVE_OPENCV)
  v4r_include_directories(${OPENCV_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${OPENCV_LIBRARIES})
endif()

if(HAVE_BOOST)
  v4r_include_directories(${BOOST_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${BOOST_LIBRARIES})
endif()

if(HAVE_OPENNURBS)
  v4r_include_directories(${OPENNURBS_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${OPENNURBS_LIBRARIES})
endif()

if(HAVE_ON_NURBS)
  v4r_include_directories(${ON_NURBS_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${ON_NURBS_LIBRARIES})
endif()

#add_definitions(-std=c++0x)
add_definitions(-DEIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET)
add_definitions(-Dlinux -D__x86_64__)

set(FORMS
        mainwindow.ui
        params.ui
)

SET(BINARY_NAME createTrainingSet)
add_executable(${BINARY_NAME} createTrainingSet.cpp)
target_link_libraries(${BINARY_NAME} ${AttentionSegmentation_DEPS} ${DEP_LIBS}) 

SET(BINARY_NAME scaleTrainingSet)
add_executable(${BINARY_NAME} scaleTrainingSet.cpp)
target_link_libraries(${BINARY_NAME} ${AttentionSegmentation_DEPS} ${DEP_LIBS}) 

SET(BINARY_NAME trainSVM)
add_executable(${BINARY_NAME} trainSVM.cpp)
target_link_libraries(${BINARY_NAME} ${AttentionSegmentation_DEPS} ${DEP_LIBS}) 

SET(BINARY_NAME segment)
add_executable(${BINARY_NAME} segment.cpp)
target_link_libraries(${BINARY_NAME} ${AttentionSegmentation_DEPS} ${DEP_LIBS}) 

SET(BINARY_NAME segmentAttention)
add_executable(${BINARY_NAME} segmentAttention.cpp)
target_link_libraries(${BINARY_NAME} ${AttentionSegmentation_DEPS} ${DEP_LIBS}) 

SET(BINARY_NAME segmentIncremental)
add_executable(${BINARY_NAME} segmentIncremental.cpp)
target_link_libraries(${BINARY_NAME} ${AttentionSegmentation_DEPS} ${DEP_LIBS}) 


INSTALL(TARGETS segmentAttention
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
