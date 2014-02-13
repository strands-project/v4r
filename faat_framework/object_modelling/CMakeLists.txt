set(SUBSYS_NAME object_modelling)
set(LIB_NAME faat_pcl_${SUBSYS_NAME})
set(SUBSYS_DESC "FAAT modelling")
set(SUBSYS_DEPS_PCL common io search kdtree octree features filters registration sample_consensus recognition)
#set(SUBSYS_DEPS_FAAT_PCL )
SET(FOO_LIBRARY_TYPE SHARED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include 
                    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/include
                    ${CUDA_TOOLKIT_INCLUDE}                    
                    ../utils/)

set(om_incs
  #include/faat_pcl/${SUBSYS_NAME}/object_modeller.h
  #include/faat_pcl/${SUBSYS_NAME}/ggo.h
  include/faat_pcl/${SUBSYS_NAME}/merge_sequence.h
  include/faat_pcl/${SUBSYS_NAME}/get_nice_model.h
)

set(om_impl_incs
  #include/faat_pcl/${SUBSYS_NAME}/impl/object_modeller.hpp
)

set(srcs
    #src/object_modeller.cpp
    #src/ggo.cpp
    src/merge_sequence.cpp
    src/get_nice_model.cpp
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include 
                    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../3d_rec_framework/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../recognition/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../registration/include)

add_library(${LIB_NAME} ${FOO_LIBRARY_TYPE} ${srcs} ${om_incs} ${om_impl_incs})
target_link_libraries(${LIB_NAME} ${PCL_LIBRARIES} faat_pcl_utils faat_pcl_recognition faat_pcl_registration)
                    
#add_executable(object_modelling test/do_modelling.cpp)  
#target_link_libraries(object_modelling ${PCL_LIBRARIES} faat_pcl_utils faat_pcl_recognition faat_pcl_object_modelling)

#add_executable(object_modelling_no_seg_xyz test/do_modelling_no_segmentation_xyz.cpp)  
#target_link_libraries(object_modelling_no_seg_xyz ${PCL_LIBRARIES} faat_pcl_utils faat_pcl_recognition faat_pcl_object_modelling)

#add_executable(ggo test/global_graph_optimizer.cpp)  
#target_link_libraries(ggo ${PCL_LIBRARIES} faat_pcl_utils faat_pcl_recognition faat_pcl_object_modelling levmar)

#add_executable(red_green_blue_to_xyzrgb test/red_green_blue_to_xyzrgb.cpp)  
#target_link_libraries(red_green_blue_to_xyzrgb ${PCL_LIBRARIES} faat_pcl_utils)

add_executable(render_from_ply test/render_from_ply.cpp)  
target_link_libraries(render_from_ply ${PCL_LIBRARIES} faat_pcl_utils boost_regex)

add_executable(object_modelling_willow test/do_modelling_willow.cpp)  
target_link_libraries(object_modelling_willow ${PCL_LIBRARIES} faat_pcl_utils faat_pcl_recognition faat_pcl_registration boost_regex)

add_executable(get_nice_model test/get_nice_model.cpp)  
target_link_libraries(get_nice_model ${PCL_LIBRARIES} ${OpenCV_LIBS} faat_pcl_utils faat_pcl_recognition faat_pcl_registration boost_regex)

add_executable(stable_planes test/stable_planes.cpp)  
target_link_libraries(stable_planes ${PCL_LIBRARIES} ${OpenCV_LIBS} faat_pcl_utils faat_pcl_recognition faat_pcl_registration boost_regex)

add_executable(merge_multiple test/merge_multiple_sequences.cpp)
target_link_libraries(merge_multiple ${PCL_LIBRARIES} ${OpenCV_LIBS} faat_pcl_utils faat_pcl_recognition faat_pcl_registration faat_pcl_object_modelling boost_regex)

add_executable(object_modelling_willow_unknown_poses test/do_modelling_willow_unknown_initial_poses.cpp include/seg_do_modelling.h)
target_link_libraries(object_modelling_willow_unknown_poses ${PCL_LIBRARIES} ${OpenCV_LIBS} faat_pcl_utils faat_pcl_recognition faat_pcl_registration boost_regex)

add_executable(multiview test/multiview.cpp include/seg_do_modelling.h)
target_link_libraries(multiview ${PCL_LIBRARIES} ${OpenCV_LIBS} faat_pcl_utils faat_pcl_recognition faat_pcl_registration boost_regex)

add_executable(visualize_directory test/visualize.cpp include/seg_do_modelling.h)
target_link_libraries(visualize_directory ${PCL_LIBRARIES} ${OpenCV_LIBS} faat_pcl_utils faat_pcl_recognition faat_pcl_registration boost_regex)
