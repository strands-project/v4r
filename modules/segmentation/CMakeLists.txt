if(NOT HAVE_PCL)
  v4r_module_disable(segmentation)
endif()

set(the_description "Segmentation")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

v4r_add_module(segmentation v4r_core v4r_io)
v4r_module_include_directories(${PCL_INCLUDE_DIRS})
v4r_glob_module_sources()

v4r_create_module(${PCL_LIBRARIES})
