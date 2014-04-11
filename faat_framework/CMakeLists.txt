cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(FAAT_PCL)
set(FAAT_PCL_VERSION 0.1)

MESSAGE(STATUS ${PCL_DIR})
# Find modules
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH ${PCL_DIR} ${CMAKE_MODULE_PATH})
set(FAAT_PCL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
MESSAGE(STATUS ${CMAKE_MODULE_PATH})

# Allow the developer to select if Dynamic or Static libraries are built
OPTION(BUILD_SHARED_LIBS "Build Shared Libraries" ON)
IF(BUILD_SHARED_LIBS)
  SET(FAAT_LIB_TYPE SHARED)
ELSE(BUILD_SHARED_LIBS)
  SET(FAAT_LIB_TYPE STATIC)
ENDIF(BUILD_SHARED_LIBS)

# Special flags
set(FAAT_C_FLAGS)
set(FAAT_CXX_FLAGS)
set(FAAT_CXX_FLAGS_RELEASE)
set(FAAT_EXE_LINKER_FLAGS)
set(FAAT_EXE_LINKER_FLAGS_RELEASE)
set(FAAT_SHARED_LINKER_FLAGS)
set(FAAT_SHARED_LINKER_FLAGS_RELEASE)
if(CMAKE_COMPILER_IS_GNUCC)
  #set(FAAT_CXX_FLAGS "${FAAT_CXX_FLAGS} -fexceptions -Wunused-function  -Wunused-label     
  #        -Wunused-value -Wunused-variable -Wuninitialized  -Wignored-qualifiers -Wsign-compare -fopenmp")
  #set(FAAT_C_FLAGS "${FAAT_C_FLAGS} -Wuninitialized -Wignored-qualifiers -Wsign-compare")
  set(FAAT_CXX_FLAGS "${FAAT_CXX_FLAGS} -fopenmp")
  set(FAAT_C_FLAGS "${FAAT_C_FLAGS} -fopenmp")
  set(FAAT_EXE_LINKER_FLAGS "${FAAT_EXE_LINKER_FLAGS} -Wl,--warn-unresolved-symbols -Wl,--as-needed")
  set(FAAT_SHARED_LINKER_FLAGS "${FAAT_SHARED_LINKER_FLAGS} -Wl,--warn-unresolved-symbols -Wl,--as-needed")
elseif(MSVC)
  add_definitions ("-DBOOST_ALL_NO_LIB -D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS -DNOMINMAX")
  #SET(FAAT_CXX_FLAGS "${FAAT_CXX_FLAGS} /bigobj /EHsc /fp:fast /wd4800")
  SET(FAAT_CXX_FLAGS "${FAAT_CXX_FLAGS} /EHsc /fp:fast /wd4800 /openmp")
  SET(FAAT_CXX_FLAGS_RELEASE "${FAAT_CXX_FLAGS_RELEASE} /GL /wd4800 ")
  SET(FAAT_SHARED_LINKER_FLAGS_RELEASE "${FAAT_SHARED_LINKER_FLAGS_RELEASE} /LTCG ")
  SET(FAAT_EXE_LINKER_FLAGS_RELEASE "${FAAT_EXE_LINKER_FLAGS_RELEASE} /LTCG ")
  if(MSVC90 OR MSVC10)
    SET(FAAT_CXX_FLAGS "${FAAT_CXX_FLAGS} /MP")
  endif(MSVC90 OR MSVC10)
endif(CMAKE_COMPILER_IS_GNUCC)

if(NOT BUILD_SHARED_LIBS)
  add_definitions("-DBOOST_LIB_DIAGNOSTIC -DBOOST_THREAD_USE_LIB")
endif(NOT BUILD_SHARED_LIBS)

set(FAAT_CXX_FLAGS "${FAAT_CXX_FLAGS}" CACHE STRING "FAAT cxx options")
set(FAAT_CXX_FLAGS_RELEASE "${FAAT_CXX_FLAGS_RELEASE}" CACHE STRING "FAAT cxx release options")
set(FAAT_C_FLAGS "${FAAT_C_FLAGS}" CACHE STRING "FAAT c options")
set(FAAT_EXE_LINKER_FLAGS "${FAAT_EXE_LINKER_FLAGS}" CACHE STRING "FAAT exe linking options")
set(FAAT_EXE_LINKER_FLAGS_RELEASE "${FAAT_EXE_LINKER_FLAGS_RELEASE}" CACHE STRING "FAAT exe release linking options")
set(FAAT_SHARED_LINKER_FLAGS "${FAAT_SHARED_LINKER_FLAGS}" CACHE STRING "FAAT shared lib linking options")
set(FAAT_SHARED_LINKER_FLAGS_RELEASE "${FAAT_SHARED_LINKER_FLAGS_RELEASE}" CACHE STRING "FAAT shared release lib linking options")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FAAT_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${FAAT_CXX_FLAGS_RELEASE}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${FAAT_C_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${FAAT_EXE_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${FAAT_EXE_LINKER_FLAGS_RELEASE}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${FAAT_SHARED_LINKER_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} ${FAAT_SHARED_LINKER_FLAGS_RELEASE}")

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  add_definitions("-DBOOST_DISABLE_ASSERTS -DEIGEN_NO_DEBUG")
endif()

#put all exe files in /bin, dll and lib files in /lib
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
#
find_package( OpenCV 2.4 REQUIRED )

find_package(PCL 1.7 REQUIRED COMPONENTS common io features segmentation geometry visualization recognition keypoints surface apps)
if(PCL_FOUND)
  include_directories(${PCL_INCLUDE_DIRS})
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})
  
 # if (MSVC)
 # list(APPEND PCL_LIBRARIES optimized "${VTK_DIR}/vtkWidgets.lib" debug "${VTK_DIR}/vtkWidgets-gd.lib")
 # endif(MSVC)
  
endif(PCL_FOUND)

include(cmake/pcl_find_cuda.cmake)
set(FAAT_PCL_USE_CUDA 0)

IF(CUDA_FOUND)
  MESSAGE(STATUS "CUDA WAS INDEED FOUND!!")
  #set(FAAT_PCL_USE_CUDA 1)
ELSE(CUDA_FOUND)
  MESSAGE(STATUS "CUDA WAS NOT FOUND!!")
  set(FAAT_PCL_USE_CUDA 0)
ENDIF(CUDA_FOUND)

add_subdirectory(utils)
add_subdirectory(registration)
add_subdirectory(recognition)
add_subdirectory(3d_rec_framework)

set(CONF_INCLUDE_DIRS   ${PROJECT_SOURCE_DIR}/3d_rec_framework/include
                        ${PROJECT_SOURCE_DIR}/recognition/include
                        ${PROJECT_SOURCE_DIR}/registration/include
                        ${PROJECT_SOURCE_DIR}/utils/
                        ${PROJECT_SOURCE_DIR}/utils/include)

set(CONF_LIBRARY_DIRS   ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/lib${PROJECT_NAME}_utils.so
                        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/lib${PROJECT_NAME}_registration.so
                        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/lib${PROJECT_NAME}_recognition.so
                        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/lib${PROJECT_NAME}_3d_rec_framework.so)

if(FAAT_PCL_USE_CUDA)
	add_subdirectory(segmentation)
	add_subdirectory(object_modelling)

        set(CONF_INCLUDE_DIRS   ${CONF_INCLUDE_DIRS}
                                ${PROJECT_SOURCE_DIR}/segmentation/include
                                ${PROJECT_SOURCE_DIR}/object_modelling/include)

        set(CONF_LIBRARY_DIRS   ${CONF_LIBRARY_DIRS}
                                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/lib${PROJECT_NAME}_segmentation.so
                                ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/lib${PROJECT_NAME}_object_modelling.so)

endif(FAAT_PCL_USE_CUDA)

#add_subdirectory(tests)

export(PACKAGE ${PROJECT_NAME})

#append external libraries in utils
list(APPEND CONF_LIBRARY_DIRS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/liblevmar.so)
list(APPEND CONF_LIBRARY_DIRS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libsplm.so)
list(APPEND CONF_LIBRARY_DIRS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libsiftgpu.so)

configure_file(${PROJECT_NAME}Config.cmake.in
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" @ONLY)

configure_file(${PROJECT_NAME}ConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake" @ONLY)

message(STATUS "faat libraries:" ${FAAT_LIBRARIES})
message(STATUS "faat include dirs:" ${FAAT_INCLUDE_DIRS})
