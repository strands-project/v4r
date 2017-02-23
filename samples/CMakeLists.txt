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

if(HAVE_GLFW3)
  v4r_include_directories(${GLFW3_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${GLFW3_LIBRARIES})
endif()

if(HAVE_GLEW)
  v4r_include_directories(${GLEW_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${GLEW_LIBRARIES})
endif()

if(HAVE_ASSIMP)
  v4r_include_directories(${ASSIMP_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${ASSIMP_LIBRARIES})
endif()

if(HAVE_GLOG)
  v4r_include_directories(${GLOG_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${GLOG_LIBRARIES})
endif()

if(HAVE_BOOST)
  v4r_include_directories(${BOOST_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${BOOST_LIBRARIES})
endif()

#if(HAVE_SIFTGPU)
#  v4r_include_directories(${SIFTGPU_INCLUDE_DIRS})
#  list(APPEND DEP_LIBS ${SIFTGPU_LIBRARIES})
#endif()


if(TRUE)#HAVE_PCL_1_8)
  v4r_include_directories(${PCL_1_8_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${PCL_1_8_LIBRARY})
endif()

if(CMAKE_COMPILER_IS_GNUCXX AND NOT ENABLE_NOISY_WARNINGS)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
endif()

  # ---------------------------------------------
  #      Define executable targets
  # ---------------------------------------------
MACRO(V4R_DEFINE_CPP_EXAMPLE name)
      set(sample_subfolder "cpp")
      v4r_check_dependencies(${V4R_DEPS})

     if(V4R_DEPENDENCIES_FOUND)
      v4r_include_modules_recurse(${V4R_DEPS})


    set(the_target "${sample_kind}_${name}")
    add_executable(${the_target} "${name}.cpp")
    target_link_libraries(${the_target} ${V4R_LINKER_LIBS} ${V4R_DEPS} ${DEP_LIBS})

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

    if (INSTALL_C_EXAMPLES AND NOT WIN32)
      install(TARGETS ${the_target}
              RUNTIME DESTINATION bin
              LIBRARY DESTINATION lib
              ARCHIVE DESTINATION lib
              )
    endif()
    else()
        message("*** Cannot build ${sample_kind}_${name} because required dependencies (${V4R_DEPS}) are not found")
    endif()

  ENDMACRO()

project(cpp_samples)

# Detect if we want to build samples with library binaries or not
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_LIST_DIR)
#
# BUILD CASE 1: Build samples with library sources
#

#add_subdirectory(c)
add_subdirectory(examples)
add_subdirectory(eval)
add_subdirectory(tools)
#add_subdirectory(icra16)
#add_subdirectory(ocl)

#
# END OF BUILD CASE 1: Build samples with library sources
#
else()
#
# BUILD CASE 2: Build samples with library binaries
#
cmake_minimum_required(VERSION 2.8)

project(samples C CXX)
option(BUILD_EXAMPLES "Build samples" ON)
option(BUILD_EVALUATION_TOOLS "Build evaluation tools" ON)
option(BUILD_UTILITY_TOOLS "Build utility tools" ON)
#option(BUILD_ICRA16_TOOLS "Build evaluations for RA-L Paper 2016, Faeulhammer" OFF)


find_package(V4R REQUIRED)

#add_subdirectory(c)
add_subdirectory(examples)
add_subdirectory(eval)
add_subdirectory(tools)
#add_subdirectory(icra16)
#add_subdirectory(ocl)

endif()
