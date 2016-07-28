if(BUILD_EVALUATION_TOOLS)
  SET(sample_kind eval)
  SET(sample_KIND EVAL)

  SET(V4R_DEPS v4r_recognition v4r_io)
  V4R_DEFINE_CPP_EXAMPLE(compute_recognition_rate)

  SET(V4R_DEPS v4r_recognition v4r_io)
  V4R_DEFINE_CPP_EXAMPLE(compute_recognition_rate_over_occlusion)

  #SET(V4R_DEPS v4r_ml v4r_recognition v4r_rendering v4r_segmentation)
  #V4R_DEFINE_CPP_EXAMPLE(esf_object_classifier)

  SET(V4R_DEPS v4r_object_modelling)
  V4R_DEFINE_CPP_EXAMPLE(incremental_object_learning)

  SET(V4R_DEPS v4r_recognition)
  V4R_DEFINE_CPP_EXAMPLE(object_recognizer)

  #SET(V4R_DEPS v4r_recognition)
  #V4R_DEFINE_CPP_EXAMPLE(object_recognizer_multiview)

endif()

if (INSTALL_C_EVALUATION_TOOLS AND NOT WIN32)
  file(GLOB C_SAMPLES *.c *.cpp *.jpg *.png *.data makefile.* build_all.sh *.dsp *.cmd )
  install(FILES ${C_SAMPLES}
          DESTINATION ${V4R_SAMPLES_SRC_INSTALL_PATH}/cpp
          PERMISSIONS OWNER_READ GROUP_READ WORLD_READ COMPONENT samples)
endif()
