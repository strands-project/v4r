if(NOT HAVE_PCL)
  v4r_module_disable(segmentation)
endif()

set(the_description "Segmentation")

v4r_add_module(segmentation v4r_core v4r_io)
v4r_module_include_directories(${PCL_INCLUDE_DIRS})
v4r_glob_module_sources()

v4r_create_module(${PCL_LIBRARIES})
