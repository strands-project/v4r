add_executable(visualize_gt visualize_gt.cpp)
target_link_libraries(visualize_gt ${PCL_LIBRARIES} ${OpenCV_LIBS} ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework)

add_executable(compute_occlusion_values_gt compute_occlusion_values.cpp)
target_link_libraries(compute_occlusion_values_gt ${PCL_LIBRARIES} ${OpenCV_LIBS} ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework)

add_executable(create_images create_images.cpp)
target_link_libraries(create_images ${PCL_LIBRARIES} ${OpenCV_LIBS} ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework)

add_executable(old_willow_to_new old_willow_models_to_new.cpp)
target_link_libraries(old_willow_to_new ${PCL_LIBRARIES} ${OpenCV_LIBS} ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework)

add_executable(gt_vs_gt evaluate_gt_vs_gt.cpp)
target_link_libraries(gt_vs_gt ${PCL_LIBRARIES} ${OpenCV_LIBS} ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework)

add_executable(count_gt_objects_per_sequence count_gt_objects_per_sequence.cpp)
target_link_libraries(count_gt_objects_per_sequence ${PCL_LIBRARIES} ${OpenCV_LIBS} ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework)

add_executable(merge_hypotheses merge_hypotheses.cpp)
target_link_libraries(merge_hypotheses ${PCL_LIBRARIES} ${OpenCV_LIBS} ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework)


#FIND_PACKAGE(Qt4)
#include(${QT_USE_FILE})
#QT4_WRAP_CPP(foo_MOC_SRCS main_window.h)
#add_executable(GT6DOF GT6DOF.cpp ${foo_MOC_SRCS} main_window.cpp)
#target_link_libraries(GT6DOF ${PCL_LIBRARIES} ${QT_LIBRARIES} QVTK ${PROJECT_NAME}_recognition ${PROJECT_NAME}_utils ${PROJECT_NAME}_3d_rec_framework ${PROJECT_NAME}_registration)
