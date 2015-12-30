set(the_description "Registration")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
v4r_define_module(registration REQUIRED v4r_core v4r_io v4r_features pcl ceres opencv OPTIONAL edt siftgpu)
