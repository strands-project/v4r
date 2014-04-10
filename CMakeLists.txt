cmake_minimum_required ( VERSION 2.8.3 )
PROJECT(strandsv4r)
find_package(catkin REQUIRED roscpp rospy)

message(STATUS "PCL_DIR:" ${PCL_DIR})
message(STATUS "V4R_PCL_DIR:" ${V4R_PCL_DIR})

set(PCL_DIR_BEFORE ${PCL_DIR})
set(PCL_DIR ${V4R_PCL_DIR})

unset(PCL_FOUND)
unset(PCL_LIBRARIES)
unset(PCL_INCLUDE_DIRS)
unset(PCL_LIBRARY_DIRS)
unset(PCL_DEFINITIONS)

set(all_components_in_pcl  common geometry kdtree octree search sample_consensus filters io 2d features ml segmentation keypoints registration recognition surface visualization outofcore people tracking stereo apps gpu_containers gpu_utils gpu_octree gpu_segmentation gpu_kinfu_large_scale gpu_kinfu gpu_features )
list(LENGTH all_components_in_pcl PCL_NUM_COMPONENTS)

foreach(component ${all_components_in_pcl})
	string(TOUPPER "${component}" COMPONENT)
	#MESSAGE(STATUS "component" ${component})
	#MESSAGE(STATUS "PCL_${COMPONENT}_INCLUDE_DIR: " ${PCL_${COMPONENT}_INCLUDE_DIR})
	if(DEFINED PCL_${COMPONENT}_INCLUDE_DIR)
		#MESSAGE(STATUS "defined")
		unset(PCL_${COMPONENT}_INCLUDE_DIR CACHE)
		unset(PCL_${COMPONENT}_LIBRARIES CACHE)
		unset(PCL_${COMPONENT}_LIBRARY_DIRS CACHE)
		unset(PCL_${COMPONENT}_DEFINITIONS CACHE)
	endif(DEFINED PCL_${COMPONENT}_INCLUDE_DIR)
	#unset(PCL_${COMPONENT}_LIBRARIES)
	#MESSAGE(STATUS "PCL_${COMPONENT}_INCLUDE_DIR: " ${PCL_${COMPONENT}_INCLUDE_DIR})
endforeach(component)

###########################################
## Do not waste the top level cmake !!! ###
###########################################

# the gcc-4.2.1 coming with MacOS X is not compatible with the OpenMP pragmas we use, so disabling OpenMP for it
if((NOT APPLE) OR (NOT CMAKE_COMPILER_IS_GNUCXX) OR (GCC_VERSION VERSION_GREATER 4.2.1))
  find_package(OpenMP)
endif()
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  message (STATUS "Found OpenMP")
  if(MSVC90 OR MSVC10)
    if(MSVC90)
      set(OPENMP_DLL VCOMP90)
    elseif(MSVC10)
      set(OPENMP_DLL VCOMP100)
    endif(MSVC90)
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /DELAYLOAD:${OPENMP_DLL}D.dll")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DELAYLOAD:${OPENMP_DLL}.dll")
  endif(MSVC90 OR MSVC10)
else(OPENMP_FOUND)
  message (STATUS "Not found OpenMP")
endif()

include(opencv.cmake)
include(v4rutils.cmake)

SET(LIBRARY_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/lib")
SET(EXECUTABLE_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/bin")

set(V4R_BUILD 1)

#add_definitions(-DDEBUG) # this should be defined in each module individually [tm]
add_definitions(-DLINUX)
add_definitions("-Wall -Wl,--no-undefined -fno-strict-aliasing -pthread")

include_directories(.)

add_subdirectory(v4r)
add_subdirectory(faat_framework)

message(STATUS "V4R_LIBRARIES: " ${V4R_LIBRARIES})
message(STATUS "V4R_LIBRARIES_PC: " ${V4R_LIBRARIES_PC})

catkin_package()


# install target
configure_file(v4r.pc.in "${PROJECT_BINARY_DIR}/v4r.pc" @ONLY)
INSTALL(FILES ${PROJECT_BINARY_DIR}/v4r.pc DESTINATION lib/pkgconfig)

# uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)
add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)

set(PCL_DIR ${PCL_DIR_BEFORE})
