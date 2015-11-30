# ----------------------------------------------------------------------------
#  CMake file for C samples. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------

SET(V4R_ICRA16_SAMPLES_REQUIRED_DEPS v4r_common v4r_features v4r_keypoints v4r_recognition v4r_registration v4r_segmentation v4r_core v4r_io v4r_object_modelling)

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

v4r_check_dependencies(${V4R_ICRA16_SAMPLES_REQUIRED_DEPS})

if(BUILD_EXAMPLES AND V4R_DEPENDENCIES_FOUND)
  project(cpp_samples)
  v4r_include_modules(${V4R_ICRA16_SAMPLES_REQUIRED_DEPS})

  if(CMAKE_COMPILER_IS_GNUCXX AND NOT ENABLE_NOISY_WARNINGS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
  endif()

  # ---------------------------------------------
  #      Define executable targets
  # ---------------------------------------------
  MACRO(V4R_DEFINE_CPP_EXAMPLE name srcs)

    if("${srcs}" MATCHES "tutorial_code")
      set(sample_kind tutorial)
      set(sample_KIND TUTORIAL)
      set(sample_subfolder "tutorials")
    else()
      set(sample_kind example)
      set(sample_KIND EXAMPLE)
      set(sample_subfolder "cpp")
    endif()

    set(the_target "${sample_kind}_${name}")
    add_executable(${the_target} ${srcs})
    target_link_libraries(${the_target} ${V4R_LINKER_LIBS} ${V4R_ICRA16_SAMPLES_REQUIRED_DEPS} ${DEP_LIBS})

    set_target_properties(${the_target} PROPERTIES
      OUTPUT_NAME "icra16-${sample_kind}-${name}"
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
  ENDMACRO()

  file(GLOB_RECURSE cpp_samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
  
foreach(sample_filename ${cpp_samples})
    get_filename_component(sample ${sample_filename} NAME_WE)
    V4R_DEFINE_CPP_EXAMPLE(${sample}  ${sample_filename})
  endforeach()
endif()

if (INSTALL_C_EXAMPLES AND NOT WIN32)
  file(GLOB C_SAMPLES *.c *.cpp *.jpg *.png *.data makefile.* build_all.sh *.dsp *.cmd )
  install(FILES ${C_SAMPLES}
          DESTINATION ${V4R_SAMPLES_SRC_INSTALL_PATH}/icra16
          PERMISSIONS OWNER_READ GROUP_READ WORLD_READ COMPONENT samples)
endif()
