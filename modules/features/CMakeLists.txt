set(the_description "Features")
v4r_add_module(features REQUIRED v4r_core v4r_common v4r_keypoints pcl OPTIONAL siftgpu caffe)

    list(APPEND feat_srcs
${CMAKE_CURRENT_LIST_DIR}/src/ComputeImGradientDescriptors.cpp
${CMAKE_CURRENT_LIST_DIR}/src/ourcvfh_estimator.cpp
${CMAKE_CURRENT_LIST_DIR}/src/ImGDescOrientation.cpp
${CMAKE_CURRENT_LIST_DIR}/src/FeatureDetector_KD_ORB.cpp
${CMAKE_CURRENT_LIST_DIR}/src/pcl_ourcvfh.cpp
${CMAKE_CURRENT_LIST_DIR}/src/FeatureDetector_K_HARRIS.cpp
${CMAKE_CURRENT_LIST_DIR}/src/sift_local_estimator.cpp
${CMAKE_CURRENT_LIST_DIR}/src/esf_estimator.cpp
${CMAKE_CURRENT_LIST_DIR}/src/ImGradientDescriptor.cpp
${CMAKE_CURRENT_LIST_DIR}/src/FeatureDetector_D_FREAK.cpp
${CMAKE_CURRENT_LIST_DIR}/src/FeatureDetector_KD_FAST_IMGD.cpp
${CMAKE_CURRENT_LIST_DIR}/src/shot_local_estimator.cpp
${CMAKE_CURRENT_LIST_DIR}/src/FeatureSelection.cpp
${CMAKE_CURRENT_LIST_DIR}/src/ComputeImGDescOrientations.cpp
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/impl/pcl_ourcvfh.hpp
    )


    list(APPEND feat_headers
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/all_headers.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/ComputeImGradientDescriptors.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/FeatureDetector_KD_ORB.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/types.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/crh_estimator.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/ourcvfh_estimator.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/ComputeImGDescOrientations.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/FeatureSelection.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/shot_local_estimator.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/esf_estimator.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/ImGDescOrientation.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/color_ourcvfh_estimator.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/image_local_estimator.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/FeatureDetector_KD_FAST_IMGD.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/vedaldi_sift_local_estimator.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/FeatureDetector_K_HARRIS.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/pcl_ourcvfh.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/sift_local_estimator.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/FeatureDetector.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/FeatureDetector_D_FREAK.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/ImGradientDescriptor.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/FeatureDetectorHeaders.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/local_estimator.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/global_estimator.h
${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/colorshot_local_estimator.h
)

if(HAVE_CAFFE)
    list(APPEND FEATURE_LIBRARIES ${CAFFE_LIBRARIES})
    v4r_include_directories(${CAFFE_INCLUDE_DIRS})
    list(APPEND feat_srcs
            ${CMAKE_CURRENT_LIST_DIR}/src/global_alexnet_cnn_estimator.cpp
        )
    list(APPEND feat_headers
            ${CMAKE_CURRENT_LIST_DIR}/include/v4r/features/global_alexnet_cnn_estimator.h
        )
endif()

v4r_set_module_sources(HEADERS ${feat_headers} SOURCES ${feat_srcs} ${feat_headers})
v4r_module_include_directories()

v4r_create_module(${FEATURE_LIBRARIES})
