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

add_executable(segmentAttention segmentAttention.cpp)
target_link_libraries(segmentAttention ${AttentionSegmentation_DEPS} ${DEP_LIBS}) 

INSTALL(TARGETS segmentAttention
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
