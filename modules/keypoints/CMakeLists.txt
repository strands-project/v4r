if(NOT HAVE_PCL)
  v4r_module_disable(keypoints)
endif()

set(the_description "Keypoints")

v4r_add_module(keypoints v4r_core v4r_common v4r_features)
v4r_module_include_directories(${PCL_INCLUDE_DIRS})
v4r_glob_module_sources()

v4r_create_module(${PCL_LIBRARIES})
