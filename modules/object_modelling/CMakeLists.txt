if(NOT HAVE_PCL)
  v4r_module_disable(object_modelling)
endif()

if(NOT HAVE_OpenCV)
  v4r_module_disable(object_modelling)
endif()

set(the_description "Object_Modelling")

v4r_add_module(object_modelling v4r_core v4r_io v4r_common v4r_features v4r_registration v4r_keypoints)

#v4r_module_include_directories(${PCL_INCLUDE_DIRS})

if(HAVE_EDT)
  v4r_include_directories(${EDT_INCLUDE_DIR})
  list(APPEND DEP_LIBS ${EDT_LIBRARIES})
endif()

if(HAVE_SIFTGPU)
  v4r_include_directories(${SIFTGPU_INCLUDE_DIR})
  list(APPEND DEP_LIBS ${SIFTGPU_LIBRARIES})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#v4r_set_module_sources(HEADERS ${lib_hdrs} ${lib_hdrs_detail}
 #                     SOURCES ${lib_srcs} ${lib_int_hdrs} ${lib_cuda_srcs} ${lib_cuda_hdrs})

v4r_module_include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
v4r_create_module(${DEP_LIBS} ${PCL_LIBRARIES} ${OpenCV_LIBS})
v4r_glob_module_sources()


