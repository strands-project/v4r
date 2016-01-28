set(the_description "Recognition")
v4r_define_module(recognition REQUIRED v4r_core v4r_common v4r_io v4r_features v4r_segmentation v4r_keypoints v4r_registration v4r_rendering v4r_change_detection pcl opencv edt metslib glog boost OPTIONAL siftgpu)
