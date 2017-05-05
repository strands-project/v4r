if(BUILD_EXAMPLES)

  if(CMAKE_COMPILER_IS_GNUCXX AND NOT ENABLE_NOISY_WARNINGS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
  endif()

  SET(sample_kind example)
  SET(sample_KIND EXAMPLE)

  SET(V4R_DEPS v4r_ml)
  V4R_DEFINE_CPP_EXAMPLE(classify_with_RF)

  SET(V4R_DEPS v4r_ml)
  V4R_DEFINE_CPP_EXAMPLE(classify_with_SVM_tmp)

  SET(V4R_DEPS v4r_io v4r_rendering)
  V4R_DEFINE_CPP_EXAMPLE(depth_map_renderer)

  SET(V4R_DEPS v4r_ml v4r_recognition v4r_rendering v4r_segmentation)
  V4R_DEFINE_CPP_EXAMPLE(esf_object_classifier)

  SET(V4R_DEPS v4r_tracking v4r_recognition v4r_keypoints v4r_features)
  V4R_DEFINE_CPP_EXAMPLE(imkRecognizeObject-file)

  SET(V4R_DEPS v4r_object_modelling)
  V4R_DEFINE_CPP_EXAMPLE(incremental_object_learning)

  SET(V4R_DEPS v4r_io v4r_keypoints v4r_features)
  V4R_DEFINE_CPP_EXAMPLE(keypoint_extraction)

  SET(V4R_DEPS v4r_registration)
  V4R_DEFINE_CPP_EXAMPLE(noise_model_based_cloud_integration)

  #SET(V4R_DEPS v4r_recognition)
  #V4R_DEFINE_CPP_EXAMPLE(object_recognizer_multiview)

  SET(V4R_DEPS v4r_tracking v4r_reconstruction v4r_keypoints)
  V4R_DEFINE_CPP_EXAMPLE(object_tracker_mono)


  SET(V4R_DEPS v4r_segmentation v4r_apps)
  V4R_DEFINE_CPP_EXAMPLE(segment_pcd)

  SET(V4R_DEPS v4r_keypoints)
  V4R_DEFINE_CPP_EXAMPLE(temporalFiltering)

  SET(V4R_DEPS v4r_attention_segmentation)
  V4R_DEFINE_CPP_EXAMPLE(Exmpl3DSymSalMap)
  SET(V4R_DEPS v4r_attention_segmentation)
  V4R_DEFINE_CPP_EXAMPLE(ExmplColorMap)
  SET(V4R_DEPS v4r_attention_segmentation)
  V4R_DEFINE_CPP_EXAMPLE(ExmplFrintropSaliencyMap)
  SET(V4R_DEPS v4r_attention_segmentation)
  V4R_DEFINE_CPP_EXAMPLE(ExmplHitRatio)
  SET(V4R_DEPS v4r_attention_segmentation)
  V4R_DEFINE_CPP_EXAMPLE(ExmplIttiSaliencyMap)
  SET(V4R_DEPS v4r_attention_segmentation)
  V4R_DEFINE_CPP_EXAMPLE(ExmplLocationMap)
  #SET(V4R_DEPS v4r_attention_segmentation)
  #V4R_DEFINE_CPP_EXAMPLE(ExmplMapCombination)
  SET(V4R_DEPS v4r_attention_segmentation)
  V4R_DEFINE_CPP_EXAMPLE(ExmplMSR)
  SET(V4R_DEPS v4r_attention_segmentation)
  V4R_DEFINE_CPP_EXAMPLE(ExmplOrientationMap)
  SET(V4R_DEPS v4r_attention_segmentation)
  V4R_DEFINE_CPP_EXAMPLE(ExmplSalLine)
  SET(V4R_DEPS v4r_attention_segmentation)
  V4R_DEFINE_CPP_EXAMPLE(ExmplSurfCurvatureMap)
  SET(V4R_DEPS v4r_attention_segmentation)
  V4R_DEFINE_CPP_EXAMPLE(ExmplSurfHeightMap)
  SET(V4R_DEPS v4r_attention_segmentation)
  V4R_DEFINE_CPP_EXAMPLE(ExmplSurfOrientationMap)
  SET(V4R_DEPS v4r_attention_segmentation)
  V4R_DEFINE_CPP_EXAMPLE(ExmplSymSalMap)
  SET(V4R_DEPS v4r_attention_segmentation)
  V4R_DEFINE_CPP_EXAMPLE(ExmplWTA)

endif()

if (INSTALL_C_EXAMPLES AND NOT WIN32)
  file(GLOB C_SAMPLES *.c *.cpp *.jpg *.png *.data makefile.* build_all.sh *.dsp *.cmd )
  install(FILES ${C_SAMPLES}
          DESTINATION ${V4R_SAMPLES_SRC_INSTALL_PATH}/cpp
          PERMISSIONS OWNER_READ GROUP_READ WORLD_READ COMPONENT samples)
endif()
