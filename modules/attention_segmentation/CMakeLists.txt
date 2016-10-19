set(the_description "Attention-based Segmentation")
v4r_include_directories(${OPENNURBS_INCLUDE_DIRS})
v4r_define_module(attention_segmentation REQUIRED v4r_core v4r_common v4r_io pcl opencv opennurbs on_nurbs)
