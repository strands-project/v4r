set(the_description "Registration")

v4r_add_module(registration v4r_common v4r_io v4r_features)


if(HAVE_EDT)
  v4r_debug_message(" ->>>>>>>>>>>>>>>>>>>>>>>>>>>>" ${EDT_INCLUDE_DIR})
  v4r_include_directories(${EDT_INCLUDE_DIR})
  list(APPEND DEP_LIBS ${EDT_LIBRARY})
endif()

# depends on siftgpu and opencv ceres

#FIND_PACKAGE (OpenCV REQUIRED)
#include_directories(${OPENCV_INCLUDE_DIRS})
#link_directories(${OPENCV_LIBRARY_DIRS})

#FIND_PACKAGE(Ceres)
#IF(${Ceres_FOUND})
  #INCLUDE_DIRECTORIES(${CERES_INCLUDES})
  #IF(${Ceres_VERSION} EQUAL 1.8.0)
    #add_definitions(-DCERES_VERSION_LESS_1_9_0)
  #ENDIF(${Ceres_VERSION} EQUAL 1.8.0)
#ELSE(${Ceres_FOUND})
  #MESSAGE("\nATTENTION: Ceres is not installed! Some parts of V4R_registration will not be compiled!\n")
  #add_definitions(-DKP_NO_CERES_AVAILABLE)
#ENDIF(${Ceres_FOUND})

#PROJECT(v4r_${SUBSYS_NAME})

set(lib_srcs
  src/fast_icp_with_gc.cpp
  src/MultiSessionModelling.cpp
  src/FeatureBasedRegistration.cpp
  src/StablePlanesRegistration.cpp
  src/PartialModelRegistrationBase.cpp
)

set(lib_hdrs
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/fast_icp_with_gc.h"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/impl/fast_icp_with_gc.hpp"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/uniform_sampling.h"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/MultiSessionModelling.h"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/FeatureBasedRegistration.h"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/StablePlanesRegistration.h"
  "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/PartialModelRegistrationBase.h"
)

#IF(${Ceres_FOUND})
  #SET(SOURCE_H ${SOURCE_H}
    #include/v4r/${SUBSYS_NAME}/MvLMIcp.h
  #)

  #SET(SOURCE_CPP ${SOURCE_CPP}
    #src/MvLMIcp.cpp
  #)

#ENDIF(${Ceres_FOUND})


# unable to compile everything now
#v4r_glob_module_sources()

v4r_set_module_sources(HEADERS ${lib_hdrs} ${lib_hdrs_detail}
                       SOURCES ${lib_srcs} ${lib_int_hdrs} ${lib_cuda_srcs} ${lib_cuda_hdrs})

v4r_module_include_directories()
v4r_create_module()


# in case of ceres is a static lib it must be linked to the executable
#IF(${Ceres_FOUND})
    #get_target_property(LIBTYPE ${CERES_LIBRARIES} TYPE)

    #IF("${LIBTYPE}" STREQUAL "SHARED_LIBRARY")
        #target_link_libraries(${PREFIX_LIB}${SUBSYS_NAME} ${CERES_LIBRARIES})
    #ELSE("${LIBTYPE}" STREQUAL "SHARED_LIBRARY")
        #message(STATUS "libceres is static")
    #ENDIF("${LIBTYPE}" STREQUAL "SHARED_LIBRARY")

#ENDIF(${Ceres_FOUND})

