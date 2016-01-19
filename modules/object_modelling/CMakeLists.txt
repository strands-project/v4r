set(the_description "Object Modelling")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
v4r_define_module(object_modelling REQUIRED v4r_core v4r_io v4r_common v4r_keypoints v4r_features v4r_registration pcl opencv OPTIONAL siftgpu)
