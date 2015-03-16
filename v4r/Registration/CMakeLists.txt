SET(V4R_REGISTRATION 1 CACHE BOOL "Registration tools for Point clouds" )
IF(V4R_REGISTRATION)

FIND_PACKAGE (PCL)
INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})

FIND_PACKAGE (OpenCV)
include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})

FIND_PACKAGE(Ceres)
IF(${Ceres_FOUND})
  INCLUDE_DIRECTORIES(${CERES_INCLUDES})
ELSE(${Ceres_FOUND})
  MESSAGE("\nATTENTION: Ceres is not installed! Some parts of V4R_registration will not be compiled!\n")
  add_definitions(-DKP_NO_CERES_AVAILABLE)
ENDIF(${Ceres_FOUND})

PROJECT(v4rRegistration)

SET(SOURCE_H
MultiSessionModelling.h
FeatureBasedRegistration.h
StablePlanesRegistration.h
PartialModelRegistrationBase.h
VisibilityReasoning.h
VisibilityReasoning.hpp
)

SET(SOURCE_CPP
MultiSessionModelling.cpp
FeatureBasedRegistration.cpp
StablePlanesRegistration.cpp
PartialModelRegistrationBase.cpp
VisibilityReasoning.cpp
)

IF(${Ceres_FOUND})
  SET(SOURCE_H ${SOURCE_H}
    MvLMIcp.h
  )

  SET(SOURCE_CPP ${SOURCE_CPP}
    MvLMIcp.cpp
  )

ENDIF(${Ceres_FOUND})

add_library(${PROJECT_NAME} SHARED ${SOURCE_H} ${SOURCE_CPP})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} v4rEDT v4rORRecognition v4rORUtils ${OPENCV_LIBRARIES} siftgpu)

# in case of ceres is a static lib it must be linked to the executable
IF(${Ceres_FOUND})
    get_target_property(LIBTYPE ${CERES_LIBRARIES} TYPE)

    IF("${LIBTYPE}" STREQUAL "SHARED_LIBRARY")
        target_link_libraries(${PROJECT_NAME} ${CERES_LIBRARIES})
    ELSE("${LIBTYPE}" STREQUAL "SHARED_LIBRARY")
        message(STATUS "libceres is static")
    ENDIF("${LIBTYPE}" STREQUAL "SHARED_LIBRARY")

ENDIF(${Ceres_FOUND})

v4r_add_library(${PROJECT_NAME} "${SOURCE_H}")

#executables...
#add_executable(testMVICP testMVICP.cpp)
#target_link_libraries(testMVICP ${PCL_LIBRARIES} v4rRegistration v4rORUtils v4rORRecognition boost_regex ${CERES_LIBRARIES})

#add_executable(testMultiSessionModelling testMultiSessionModelling.cpp)
#target_link_libraries(testMultiSessionModelling ${PCL_LIBRARIES} v4rRegistration v4rORRecognition v4rORUtils boost_regex ${CERES_LIBRARIES})

#add_executable(convertJPFormatToNew convertJPFormatToNew.cpp)
#target_link_libraries(convertJPFormatToNew ${PCL_LIBRARIES} v4rRegistration v4rORUtils v4rORRecognition boost_regex ${CERES_LIBRARIES})

ENDIF(V4R_REGISTRATION)



