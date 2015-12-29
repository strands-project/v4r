set(the_description "Common stuff")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
v4r_define_module(common REQUIRED v4r_core pcl opencv)
