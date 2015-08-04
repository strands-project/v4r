if(NOT HAVE_PCL)
  v4r_module_disable(reconstruction)
endif()

if(HAVE_CERES)
  v4r_include_directories(${CERES_INCLUDES})
  list(APPEND DEP_LIBS ${CERES_LIBRARIES})
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


set(the_description "Reconstruction")

v4r_add_module(reconstruction v4r_core v4r_io v4r_keypoints)
v4r_module_include_directories(${PCL_INCLUDE_DIRS})
v4r_glob_module_sources()

v4r_create_module(${DEP_LIBS})
