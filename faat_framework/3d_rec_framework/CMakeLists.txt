set(SUBSYS_NAME 3d_rec_framework)
set(SUBSYS_DESC "3D recognition framework")
set(DEFAULT FALSE)

# Find VTK 
# if(NOT VTK_FOUND)
    # set(DEFAULT FALSE)
    # set(REASON "VTK was not found.")
# else(NOT VTK_FOUND)
    # set(DEFAULT TRUE)
    # set(REASON)
    # include (${VTK_USE_FILE})
# endif(NOT VTK_FOUND)

# OpenNI found?
if(NOT OPENNI_FOUND OR NOT BUILD_OPENNI)
    set(DEFAULT FALSE)
    set(REASON "OpenNI was not found or was disabled by the user.")
else()
    set(DEFAULT TRUE)
    set(REASON)
endif()

#PCL_SUBSYS_OPTION(build ${SUBSYS_NAME} ${SUBSYS_DESC} ${DEFAULT} ${REASON})
#PCL_SUBSYS_DEPEND(build ${SUBSYS_NAME} DEPS ${SUBSYS_DEPS} EXT_DEPS vtk openni)

include_directories (${OpenCV_INCLUDE_DIRS})
include_directories (${CMAKE_CURRENT_BINARY_DIR})
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/include 
                     ${CMAKE_CURRENT_SOURCE_DIR}/../utils/include
                     ${CMAKE_CURRENT_SOURCE_DIR}/../utils/
                     ${CMAKE_CURRENT_SOURCE_DIR}/../recognition/include)

set(incs_fw include/faat_pcl/${SUBSYS_NAME}/feature_wrapper/normal_estimator.h)
set(incs_fw_global include/faat_pcl/${SUBSYS_NAME}/feature_wrapper/global/cvfh_estimator.h 
                   include/faat_pcl/${SUBSYS_NAME}/feature_wrapper/global/vfh_estimator.h
                   include/faat_pcl/${SUBSYS_NAME}/feature_wrapper/global/esf_estimator.h
                   include/faat_pcl/${SUBSYS_NAME}/feature_wrapper/global/crh_estimator.h
                   include/faat_pcl/${SUBSYS_NAME}/feature_wrapper/global/global_estimator.h
                   include/faat_pcl/${SUBSYS_NAME}/feature_wrapper/global/ourcvfh_estimator.h
                   include/faat_pcl/${SUBSYS_NAME}/feature_wrapper/global/color_ourcvfh_estimator.h
                   include/faat_pcl/${SUBSYS_NAME}/feature_wrapper/global/organized_color_ourcvfh_estimator.h
                   include/faat_pcl/${SUBSYS_NAME}/feature_wrapper/global/organized_ourcvfh.h)

set(incs_fw_local include/faat_pcl/${SUBSYS_NAME}/feature_wrapper/local/local_estimator.h
                  include/faat_pcl/${SUBSYS_NAME}/feature_wrapper/local/fpfh_local_estimator.h
                  include/faat_pcl/${SUBSYS_NAME}/feature_wrapper/local/fpfh_local_estimator_omp.h
                  include/faat_pcl/${SUBSYS_NAME}/feature_wrapper/local/shot_local_estimator.h
                  include/faat_pcl/${SUBSYS_NAME}/feature_wrapper/local/colorshot_local_estimator.h
                  include/faat_pcl/${SUBSYS_NAME}/feature_wrapper/local/shot_local_estimator_omp.h
                  include/faat_pcl/${SUBSYS_NAME}/feature_wrapper/local/image/image_local_estimator.h
                  include/faat_pcl/${SUBSYS_NAME}/feature_wrapper/local/image/sift_local_estimator.h
                  include/faat_pcl/${SUBSYS_NAME}/feature_wrapper/local/image/opencv_sift_local_estimator.h
                  include/faat_pcl/${SUBSYS_NAME}/feature_wrapper/local/image/vedaldi_sift_local_estimator.h)

set(incs_pc_source include/faat_pcl/${SUBSYS_NAME}/pc_source/source.h
                   include/faat_pcl/${SUBSYS_NAME}/pc_source/mesh_source.h
                   include/faat_pcl/${SUBSYS_NAME}/pc_source/partial_pcd_source.h
                   include/faat_pcl/${SUBSYS_NAME}/pc_source/registered_views_source.h
                   include/faat_pcl/${SUBSYS_NAME}/pc_source/model_only_source.h
                   include/faat_pcl/${SUBSYS_NAME}/pc_source/registered_views_pp_source.h
                   include/faat_pcl/${SUBSYS_NAME}/pc_source/unregistered_views_source.h)

set(incs_img_source include/faat_pcl/${SUBSYS_NAME}/img_source/source2d.h)

set(impl_incs_pc_source include/faat_pcl/${SUBSYS_NAME}/pc_source/impl/partial_pcd_source.hpp)
                   
set(incs_pipelines  include/faat_pcl/${SUBSYS_NAME}/pipeline/global_nn_classifier.h
                    include/faat_pcl/${SUBSYS_NAME}/pipeline/global_nn_recognizer_cvfh.h
                    include/faat_pcl/${SUBSYS_NAME}/pipeline/local_recognizer.h
                    include/faat_pcl/${SUBSYS_NAME}/pipeline/recognizer.h
                    include/faat_pcl/${SUBSYS_NAME}/pipeline/multi_pipeline_recognizer.h
                    include/faat_pcl/${SUBSYS_NAME}/pipeline/linemod3d_recognizer.h)
         
set(incs_registration include/faat_pcl/${SUBSYS_NAME}/registration/voxel_based_correspondence_estimation.h)

set(incs_segmentation include/faat_pcl/${SUBSYS_NAME}/segmentation/multiplane_segmentation.h
                      include/faat_pcl/${SUBSYS_NAME}/segmentation/impl/multiplane_segmentation.hpp)

set(incc_tools_framework include/faat_pcl/${SUBSYS_NAME}/tools/openni_frame_source.h
                         include/faat_pcl/${SUBSYS_NAME}/tools/or_evaluator.h
                         include/faat_pcl/${SUBSYS_NAME}/tools/impl/or_evaluator.hpp)

set(incs_utils include/faat_pcl/${SUBSYS_NAME}/utils/metrics.h
         include/faat_pcl/${SUBSYS_NAME}/utils/persistence_utils.h
         include/faat_pcl/${SUBSYS_NAME}/utils/vtk_model_sampling.h
        )
       
set(incs_defines include/faat_pcl/${SUBSYS_NAME}/defines/faat_3d_rec_framework_defines.h)
	   
set(srcs src/pipeline/global_nn_classifier.cpp
         src/pipeline/global_nn_recognizer_cvfh.cpp
         src/pipeline/local_recognizer.cpp 
         src/pipeline/multi_pipeline_recognizer.cpp
         src/pipeline/linemod3d_recognizer.cpp
         src/feature_wrapper/local/local_estimator.cpp
         src/feature_wrapper/local/shot_local_estimator.cpp
         src/pc_source/mesh_source.cpp
         src/pc_source/partial_pcd_source.cpp
         src/pc_source/model_only_source.cpp
         src/tools/openni_frame_source.cpp
         src/tools/or_evaluator.cpp
         src/registration/voxel_based_correspondence_estimation.cpp
         src/ml/unsupervised/rnn_clustering.cpp
         src/ml/unsupervised/3rdparty/fast_rnn.cpp
         src/segmentation/multiplane_segmentation.cpp)
         
set(impl_incs_pipeline  include/faat_pcl/${SUBSYS_NAME}/pipeline/impl/global_nn_classifier.hpp
                        include/faat_pcl/${SUBSYS_NAME}/pipeline/impl/global_nn_recognizer_cvfh.hpp
                        include/faat_pcl/${SUBSYS_NAME}/pipeline/impl/local_recognizer.hpp
                        include/faat_pcl/${SUBSYS_NAME}/pipeline/impl/multi_pipeline_recognizer.hpp)
    
set(impl_incs_registration include/faat_pcl/${SUBSYS_NAME}/registration/impl/voxel_based_correspondence_estimation.hpp)

set(incs_ml include/faat_pcl/${SUBSYS_NAME}/ml/unsupervised/rnn_clustering.h
            include/faat_pcl/${SUBSYS_NAME}/ml/unsupervised/3rdparty/fast_rnn.h)

# Install include files
#PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_NAME}/feature_wrapper ${incs_fw})
#PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_NAME}/feature_wrapper/global ${incs_fw_global})
#PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_NAME}/feature_wrapper/local ${incs_fw_local})
#PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_NAME}/tools ${incc_tools_framework})
#PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_NAME}/pipeline ${incs_pipelines})
#PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_NAME}/registration ${incs_registration})
#PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_NAME}/pc_source ${incs_pc_source})
#PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_NAME}/utils ${incs_utils})

#PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_NAME}/pipeline/impl ${impl_incs_pipeline})
#PCL_ADD_INCLUDES(${SUBSYS_NAME} ${SUBSYS_NAME}/registration/impl ${impl_incs_registration})

if(MSVC)
add_definitions(-DFAAT_3D_FRAMEWORK_EXPORTS)
endif(MSVC)
	 
	 
set(LIB_NAME ${PROJECT_NAME}_${SUBSYS_NAME})
add_library(${LIB_NAME} ${srcs} ${incs_segmentation} ${impl_incs_pc_source} ${incs_img_source} ${impl_incs_registration} ${impl_incs_pipeline} ${incs_ml} ${incs_utils} ${incs_fw} ${incs_fw_global} ${incs_fw_local} ${incc_tools_framework} ${incs_define} ${incs_registration} ${incs_pipelines} ${incs_pc_source})

target_link_libraries(${LIB_NAME} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${PROJECT_NAME}_utils ${PROJECT_NAME}_recognition siftgpu siftpp boost_regex)

#if(OPENNI_FOUND)
#  target_link_libraries(${LIB_NAME} ${OPENNI_LIBRARIES})
#  if(NOT WIN32)
#    find_package(libusb-1.0 REQUIRED)
#    target_link_libraries(${LIB_NAME} ${LIBUSB_1_LIBRARIES})
#  endif()
#endif(OPENNI_FOUND)
    
#PCL_MAKE_PKGCONFIG(${LIB_NAME} ${SUBSYS_NAME} "${SUBSYS_DESC}" "" "" "" "" "")

add_subdirectory(tests)
