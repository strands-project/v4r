set(the_description "Recognition")
v4r_add_module(recognition v4r_core v4r_common v4r_io v4r_features v4r_segmentation v4r_keypoints v4r_ml pcl pcl_1_8 opencv metslib glog boost OPTIONAL v4r_rendering siftgpu v4r_reconstruction)

    list(APPEND rec_srcs
        ${CMAKE_CURRENT_LIST_DIR}/src/ghv_opt.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/hypotheses_verification.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/hypotheses_verification_visualization.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/global_recognizer.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/global_recognition_pipeline.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/local_feature_matching.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/local_recognition_pipeline.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/local_rec_object_hypotheses.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/multi_pipeline_recognizer.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/multiview_recognizer.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/model.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/object_hypothesis.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/recognition_model_hv.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/recognition_pipeline.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/source.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/hypotheses_verification.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/recognition_pipeline.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/ghv_opt.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/IMKObjectVotesClustering.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/IMKRecognizer.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/IMKRecognizerIO.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/RansacSolvePnP.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/RansacSolvePnPdepth.cpp
    )

    list(APPEND rec_headers
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/ghv_opt.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/global_recognizer.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/global_recognition_pipeline.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/hypotheses_verification.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/hypotheses_verification_param.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/hypotheses_verification_visualization.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/local_feature_matching.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/local_recognition_pipeline.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/local_rec_object_hypotheses.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/metrics.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/model.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/multi_pipeline_recognizer.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/multiview_recognizer.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/object_hypothesis.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/recognition_pipeline.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/metrics.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/ghv_opt.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/object_hypothesis.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/IMKObjectVotesClustering.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/IMKObjectVotesClustering.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/IMKRecognizer.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/IMKRecognizerIO.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/IMKRecognizer_serialization.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/IMKView.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/RansacSolvePnP.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/RansacSolvePnPdepth.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/source.h
)

v4r_set_module_sources(HEADERS ${rec_headers} SOURCES ${rec_srcs} ${rec_headers})
v4r_module_include_directories()

v4r_create_module(${RECOGNITION_LIBRARIES})
