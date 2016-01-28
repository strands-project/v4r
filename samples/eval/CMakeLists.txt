# ----------------------------------------------------------------------------
#  CMake file for C samples. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------
SET(V4R_EVAL_SAMPLES_REQUIRED_DEPS v4r_common v4r_core v4r_features v4r_io v4r_keypoints v4r_ml v4r_recognition v4r_reconstruction v4r_registration v4r_segmentation )
SET(V4R_EVAL_SAMPLES_REQUIRED_DEPS ${V4R_EVAL_SAMPLES_REQUIRED_DEPS} v4r_object_modelling)
SET(V4R_EVAL_SAMPLES_REQUIRED_DEPS ${V4R_EVAL_SAMPLES_REQUIRED_DEPS} v4r_change_detection)

if(HAVE_PCL)
  v4r_include_directories(${PCL_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${PCL_LIBRARIES})
endif()

if(HAVE_OPENCV)
  v4r_include_directories(${OpenCV_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${OpenCV_LIBS})
endif()

if(HAVE_EDT)
  v4r_include_directories(${EDT_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${EDT_LIBRARIES})
endif()

if(HAVE_BOOST)
  v4r_include_directories(${BOOST_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${BOOST_LIBRARIES})
endif()

if(HAVE_CERES)
  v4r_include_directories(${CERES_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${CERES_LIBRARIES})
endif()

if(HAVE_LIBSVM)
  v4r_include_directories(${LIBSVM_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${LIBSVM_LIBRARIES})
endif()

if(HAVE_OPENGL)
  v4r_include_directories(${OPENGL_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${OPENGL_LIBRARIES})
endif()

if(HAVE_ASSIMP)
  v4r_include_directories(${ASSIMP_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${ASSIMP_LIBRARIES})
endif()

if(HAVE_GLOG)
  v4r_include_directories(${GLOG_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${GLOG_LIBRARIES})
endif()

v4r_check_dependencies(${V4R_EVAL_SAMPLES_REQUIRED_DEPS})

if(BUILD_EVALUATION_TOOLS AND V4R_DEPENDENCIES_FOUND)
  project(cpp_samples)
  v4r_include_modules(${V4R_EVAL_SAMPLES_REQUIRED_DEPS})

  if(CMAKE_COMPILER_IS_GNUCXX AND NOT ENABLE_NOISY_WARNINGS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
  endif()

  # ---------------------------------------------
  #      Define executable targets
  # ---------------------------------------------
  MACRO(V4R_DEFINE_CPP_EXAMPLE name srcs)

      set(sample_kind eval)
      set(sample_KIND EVAL)
      set(sample_subfolder "cpp")

    set(the_target "${sample_kind}_${name}")
    add_executable(${the_target} ${srcs})
    target_link_libraries(${the_target} ${V4R_LINKER_LIBS} ${V4R_EVAL_SAMPLES_REQUIRED_DEPS} ${DEP_LIBS})

    set_target_properties(${the_target} PROPERTIES
      OUTPUT_NAME "${sample_kind}-${name}"
      PROJECT_LABEL "(${sample_KIND}) ${name}")

    if(ENABLE_SOLUTION_FOLDERS)
      set_target_properties(${the_target} PROPERTIES FOLDER "samples/${sample_subfolder}")
    endif()

    if(WIN32)
      if (MSVC AND NOT BUILD_SHARED_LIBS)
        set_target_properties(${the_target} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:atlthunk.lib /NODEFAULTLIB:atlsd.lib /DEBUG")
      endif()
      install(TARGETS ${the_target}
              RUNTIME DESTINATION "${V4R_SAMPLES_BIN_INSTALL_PATH}/${sample_subfolder}" COMPONENT samples)
    endif()

    if (INSTALL_C_EVALUATION_TOOLS AND NOT WIN32)
      install(TARGETS ${the_target}
              RUNTIME DESTINATION bin
              LIBRARY DESTINATION lib
              ARCHIVE DESTINATION lib
              )
    endif()
  ENDMACRO()

  file(GLOB_RECURSE cpp_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
  
foreach(sample_filename ${cpp_samples})
    get_filename_component(sample ${sample_filename} NAME_WE)
    V4R_DEFINE_CPP_EXAMPLE(${sample}  ${sample_filename})
  endforeach()
endif()

if (INSTALL_C_EVALUATION_TOOLS AND NOT WIN32)
  file(GLOB C_SAMPLES *.c *.cpp *.jpg *.png *.data makefile.* build_all.sh *.dsp *.cmd )
  install(FILES ${C_SAMPLES}
          DESTINATION ${V4R_SAMPLES_SRC_INSTALL_PATH}/eval
          PERMISSIONS OWNER_READ GROUP_READ WORLD_READ COMPONENT samples)
endif()

