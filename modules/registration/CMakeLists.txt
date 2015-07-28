set(SUBSYS_NAME registration)
set(SUBSYS_DESC "v4r registration module")
set(SUBSYS_DEPS v4r_common siftgpu)
SET(V4R_REGISTRATION 1 CACHE BOOL "Registration module" )
IF(V4R_REGISTRATION)

FIND_PACKAGE (PCL REQUIRED)
INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

FIND_PACKAGE (OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})

FIND_PACKAGE(Ceres)
IF(${Ceres_FOUND})
  message("****************${Ceres_VERSION}**************")
  INCLUDE_DIRECTORIES(${CERES_INCLUDES})
  IF(${Ceres_VERSION} EQUAL 1.8.0)
    add_definitions(-DCERES_VERSION_LESS_1_9_0)
  ENDIF(${Ceres_VERSION} EQUAL 1.8.0)
ELSE(${Ceres_FOUND})
  MESSAGE("\nATTENTION: Ceres is not installed! Some parts of V4R_registration will not be compiled!\n")
  add_definitions(-DKP_NO_CERES_AVAILABLE)
ENDIF(${Ceres_FOUND})


PROJECT(v4r_${SUBSYS_NAME})

SET(SOURCE_CPP
  src/fast_icp_with_gc.cpp
  src/MultiSessionModelling.cpp
  src/FeatureBasedRegistration.cpp
  src/StablePlanesRegistration.cpp
  src/PartialModelRegistrationBase.cpp
)

SET(SOURCE_H
  include/v4r/${SUBSYS_NAME}/fast_icp_with_gc.h
  include/v4r/${SUBSYS_NAME}/impl/fast_icp_with_gc.hpp
  include/v4r/${SUBSYS_NAME}/uniform_sampling.h
  include/v4r/${SUBSYS_NAME}/MultiSessionModelling.h
  include/v4r/${SUBSYS_NAME}/FeatureBasedRegistration.h
  include/v4r/${SUBSYS_NAME}/StablePlanesRegistration.h
  include/v4r/${SUBSYS_NAME}/PartialModelRegistrationBase.h
)

IF(${Ceres_FOUND})
  SET(SOURCE_H ${SOURCE_H}
    include/v4r/${SUBSYS_NAME}/MvLMIcp.h
  )

  SET(SOURCE_CPP ${SOURCE_CPP}
    src/MvLMIcp.cpp
  )

ENDIF(${Ceres_FOUND})


include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library("${PREFIX_LIB}${SUBSYS_NAME}" SHARED ${SOURCE_H} ${SOURCE_CPP} )
target_link_libraries("${PREFIX_LIB}${SUBSYS_NAME}" ${PCL_LIBRARIES} ${OPENCV_LIBRARIES} ${SUBSYS_DEPS})

# in case of ceres is a static lib it must be linked to the executable
IF(${Ceres_FOUND})
    get_target_property(LIBTYPE ${CERES_LIBRARIES} TYPE)

    IF("${LIBTYPE}" STREQUAL "SHARED_LIBRARY")
        target_link_libraries(${PREFIX_LIB}${SUBSYS_NAME} ${CERES_LIBRARIES})
    ELSE("${LIBTYPE}" STREQUAL "SHARED_LIBRARY")
        message(STATUS "libceres is static")
    ENDIF("${LIBTYPE}" STREQUAL "SHARED_LIBRARY")

ENDIF(${Ceres_FOUND})

_install_library(${PREFIX_LIB}${SUBSYS_NAME} ${SOURCE_H} "/v4r/${SUBSYS_NAME}/")

ENDIF(V4R_REGISTRATION)
