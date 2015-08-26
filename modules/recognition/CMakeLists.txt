set(the_description "Recognition")
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O1")
v4r_define_module(recognition REQUIRED v4r_core v4r_common v4r_io v4r_features v4r_segmentation v4r_registration pcl opencv OPTIONAL edt siftgpu mets)
