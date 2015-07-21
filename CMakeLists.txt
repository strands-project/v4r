cmake_minimum_required ( VERSION 2.6 )
PROJECT(V4R)

###########################################
## Do not waste the top level cmake !!! ###
###########################################

include(cmake/openmp.cmake)
include(cmake/opencv.cmake)
include(cmake/pcl.cmake)
include(cmake/suitesparse.cmake)
include(cmake/v4rutils.cmake)

_version(2014 11 30)
_output_prefix()    
_set_output_paths()
_installation_destinations() # must be before add_subdirectory

#add_definitions(-DDEBUG) # this should be defined in each module individually [tm]
add_definitions(-DLINUX)
add_definitions("-Wall -Wno-unused-function -Wl,--no-undefined -fno-strict-aliasing -pthread")

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/modules/common/include)
include_directories(${CMAKE_SOURCE_DIR}/modules/object_modelling/include)
include_directories(${CMAKE_SOURCE_DIR}/modules/recognition/include)
include_directories(${CMAKE_SOURCE_DIR}/modules/registration/include)
include_directories(${CMAKE_SOURCE_DIR}/modules/segmentation/include)

add_subdirectory(3rdparty)
add_subdirectory(modules)
add_subdirectory(apps)

message(STATUS "V4R_LIBRARIES:  kj" ${V4R_LIBRARIES})
message(STATUS "V4R_LIBRARIES_PC: " ${V4R_LIBRARIES_PC})

_generate_find_package() # must be after add_subdirectory
_uninstall() 

# install target
configure_file(v4r.pc.in "${PROJECT_BINARY_DIR}/v4r.pc" @ONLY)
INSTALL(FILES ${PROJECT_BINARY_DIR}/v4r.pc DESTINATION lib/pkgconfig)
mark_as_advanced(V4R_LIBRARIES)

