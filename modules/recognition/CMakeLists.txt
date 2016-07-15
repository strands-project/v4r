set(the_description "Recognition")
v4r_add_module(recognition v4r_core v4r_common v4r_io v4r_features v4r_segmentation v4r_keypoints v4r_registration v4r_change_detection v4r_ml pcl opencv metslib glog boost OPTIONAL v4r_rendering siftgpu)

    list(APPEND rec_srcs
        #${CMAKE_CURRENT_LIST_DIR}/src/multiview_representation.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/global_nn_classifier.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/model_only_source.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/voxel_based_correspondence_estimation.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/partial_pcd_source.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/local_recognizer.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/local_rec_object_hypotheses.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/global_recognizer.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/recognition_model_hv.cpp
        #${CMAKE_CURRENT_LIST_DIR}/src/multiview_object_recognizer_change_detection.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/multi_pipeline_recognizer.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/model.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/source.cpp
        #${CMAKE_CURRENT_LIST_DIR}/src/multiview_object_recognizer.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/hypotheses_verification.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/recognizer.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/registered_views_source.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/ghv_opt.cpp
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/impl/multi_pipeline_recognizer.hpp
        #${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/impl/multiview_object_recognizer_init.hpp
        #${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/impl/multiview_object_recognizer.hpp
        #${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/impl/multiview_object_recognizer_change_detection.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/impl/multi_pipeline_recognizer_init.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/impl/ghv_opt.hpp
    )


    list(APPEND rec_headers
        #${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/multiview_representation.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/local_recognizer.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/partial_pcd_source.h
        #${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/multiview_object_recognizer_change_detection.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/registered_views_source.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/multi_pipeline_recognizer.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/hypotheses_verification.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/source.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/local_rec_object_hypotheses.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/vtk_model_sampling.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/voxel_based_correspondence_estimation.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/model_only_source.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/mesh_source.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/model.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/global_recognizer.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/global_nn_classifier.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/recognizer.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/metrics.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/segmenter.h
        #${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/multiview_object_recognizer.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/ghv_opt.h
        ${CMAKE_CURRENT_LIST_DIR}/include/v4r/recognition/object_hypothesis.h
)


if(HAVE_v4r_rendering)
    list(APPEND rec_srcs
        ${CMAKE_CURRENT_LIST_DIR}/src/mesh_source.cpp
)
else()
message("*Rendering is disabled. Therefore not able to compile mesh source.*")
endif()

v4r_set_module_sources(HEADERS ${rec_headers} SOURCES ${rec_srcs} ${rec_headers})
v4r_module_include_directories()

v4r_create_module(${RECOGNITION_LIBRARIES})
