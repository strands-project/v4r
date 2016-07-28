if(BUILD_UTILITY_TOOLS)

  if(CMAKE_COMPILER_IS_GNUCXX AND NOT ENABLE_NOISY_WARNINGS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
  endif()

  SET(sample_kind tool)
  SET(sample_KIND TOOL)

  SET(V4R_DEPS v4r_common v4r_io)
  V4R_DEFINE_CPP_EXAMPLE(convertOldModelDatabaseToNew)

  SET(V4R_DEPS v4r_recognition)
  V4R_DEFINE_CPP_EXAMPLE(create_annotated_images_from_recognition_gt_data)

  SET(V4R_DEPS v4r_common v4r_io)
  V4R_DEFINE_CPP_EXAMPLE(pcd2png)

  SET(V4R_DEPS v4r_recognition)
  V4R_DEFINE_CPP_EXAMPLE(pcd_ground_truth_labelling)

  SET(V4R_DEPS v4r_common v4r_io)
  V4R_DEFINE_CPP_EXAMPLE(save_pose_into_pcd)

  SET(V4R_DEPS v4r_common v4r_io)
  V4R_DEFINE_CPP_EXAMPLE(view_all_point_clouds_in_folder)

  SET(V4R_DEPS v4r_recognition)
  V4R_DEFINE_CPP_EXAMPLE(visualize_hypothesis)

endif()

if (INSTALL_C_UTILITY_TOOLS AND NOT WIN32)
  file(GLOB C_SAMPLES *.c *.cpp *.jpg *.png *.data makefile.* build_all.sh *.dsp *.cmd )
  install(FILES ${C_SAMPLES}
          DESTINATION ${V4R_SAMPLES_SRC_INSTALL_PATH}/cpp
          PERMISSIONS OWNER_READ GROUP_READ WORLD_READ COMPONENT samples)
endif()
