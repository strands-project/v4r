set(SUBSYS_NAME local_recognition_ply)
set(SUBSYS_DESC "rec test")
set(SUBSYS_DEPS_PCL recognition)

set(V4R_FOUND 0)
find_package(V4R)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include 
                    ${CMAKE_CURRENT_SOURCE_DIR}/../3d_rec_framework/include 
                    ${CMAKE_CURRENT_SOURCE_DIR}/../recognition/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../registration/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/
                    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/include)

message(STATUS "PCL LIBRARIES:" ${PCL_LIBRARIES})
add_executable(${SUBSYS_NAME} scene_local_training_recognition.cpp)  
target_link_libraries(${SUBSYS_NAME} ${PCL_LIBRARIES} ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework)

add_executable(fix_test_glover fix_test_glover.cpp)
target_link_libraries(fix_test_glover ${PCL_LIBRARIES} ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework)

add_executable(eval_cg eval_correspondence_grouping.cpp)
target_link_libraries(eval_cg ${PCL_LIBRARIES} ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework)

add_executable(eval_hv_opt_strategies evaluate_hv_opt_strategies.cpp)
target_link_libraries(eval_hv_opt_strategies ${PCL_LIBRARIES} ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework ${PROJECT_NAME}_registration)

#add_executable(est_reg_views_source test_reg_views_source.cpp)  
#target_link_libraries(est_reg_views_source ${PCL_LIBRARIES} ${OpenCV_LIBS} faat_pcl_recognition faat_pcl_utils faat_pcl_3d_rec_framework)

#add_executable(shrec_2013 shrec_2013.cpp)  
#target_link_libraries(shrec_2013 ${PCL_LIBRARIES} faat_pcl_recognition faat_pcl_utils faat_pcl_3d_rec_framework)

add_executable(mit mit.cpp)
target_link_libraries(mit ${PCL_LIBRARIES} ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework)

add_executable(mitGT_to_or_format mitGT_to_or_format.cpp)
target_link_libraries(mitGT_to_or_format ${PCL_LIBRARIES} ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework)

add_executable(mp_training_recognition multi_pipeline_training_recognition.cpp)  
target_link_libraries(mp_training_recognition ${PCL_LIBRARIES} ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework)

#add_executable(visualize_gt visualize_gt.cpp)
#target_link_libraries(visualize_gt ${PCL_LIBRARIES} ${OpenCV_LIBS} faat_pcl_recognition faat_pcl_utils faat_pcl_3d_rec_framework)

add_executable(kinect kinect.cpp)  
target_link_libraries(kinect ${PCL_LIBRARIES} ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework)

add_executable(labeling_from_orgt labeling_from_orgt.cpp)  
target_link_libraries(labeling_from_orgt ${PCL_LIBRARIES} ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework)

add_executable(save_histograms_from_ply render_from_ply_and_save_histograms.cpp)
target_link_libraries(save_histograms_from_ply ${PCL_LIBRARIES} ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework boost_regex)

add_executable(esf_histograms_from_whole_model esf_histograms_from_whole_model.cpp)
target_link_libraries(esf_histograms_from_whole_model ${PCL_LIBRARIES} ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework boost_regex)

add_executable(GO3D GO3D.cpp)  
target_link_libraries(GO3D ${PCL_LIBRARIES} ${OpenCV_LIBS} ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework boost_regex)

add_executable(noise_model test_noise_models.cpp)  
target_link_libraries(noise_model ${PCL_LIBRARIES} ${OpenCV_LIBS} ${PROJECT_NAME}_utils boost_regex)

add_executable(test_partial_pcd_source test_partial_pcd_source.cpp)  
target_link_libraries(test_partial_pcd_source ${PCL_LIBRARIES}  ${OpenCV_LIBS} ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework boost_regex)

add_executable(linemod test_linemod.cpp)  
target_link_libraries(linemod ${PCL_LIBRARIES} ${OpenCV_LIBS} ${PROJECT_NAME}_utils boost_regex)

add_executable(test_gc_constraints test_gc_constraints.cpp)  
target_link_libraries(test_gc_constraints ${PCL_LIBRARIES} ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework)

add_executable(linemod_framework test_linemod_3d_rec_framework.cpp)  
target_link_libraries(linemod_framework ${PCL_LIBRARIES} ${OpenCV_LIBS} ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework boost_regex)

add_executable(ourcvfh_test ourcvfh_test)  
target_link_libraries(ourcvfh_test ${PCL_LIBRARIES} ${OpenCV_LIBS} ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework)

IF(V4R_FOUND)

  add_executable(train_semantic_segmenter semantic_segmenter/semantic_segmenter_training.cpp)
  target_link_libraries(train_semantic_segmenter ${PCL_LIBRARIES} ${OpenCV_LIBS} v4rRandomForest ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework boost_regex)

  add_executable(classify_semantic_segmenter semantic_segmenter/semantic_segmenter_classify.cpp)
  target_link_libraries(classify_semantic_segmenter ${PCL_LIBRARIES} ${OpenCV_LIBS} v4rRandomForest ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework boost_regex)

  add_executable(mp_willow multi_pipeline_willow_dataset.cpp)
  target_link_libraries(mp_willow ${PCL_LIBRARIES} ${OpenCV_LIBS} ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework v4rSurfaceSegmenter v4rSurfaceUtils v4rTomGine)

ENDIF(V4R_FOUND)

#add_subdirectory(retrieval_3d)

#FIND_PACKAGE(Qt4)
#include(${QT_USE_FILE})
#QT4_WRAP_CPP(foo_MOC_SRCS include/main_window.h)
#add_executable(GT6DOF GT6DOF.cpp ${foo_MOC_SRCS} main_window.cpp)
#target_link_libraries(GT6DOF ${PCL_LIBRARIES} ${QT_LIBRARIES} QVTK faat_pcl_recognition faat_pcl_utils faat_pcl_3d_rec_framework faat_pcl_registration)

#add_executable(tomita test_tomita.cpp)  
#target_link_libraries(tomita ${PCL_LIBRARIES} faat_pcl_recognition faat_pcl_utils faat_pcl_3d_rec_framework)
