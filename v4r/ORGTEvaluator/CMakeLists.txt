SET(V4R_ORGT_EVALUATOR 1 CACHE BOOL "Ground-truth evaluator tools for object recognition" )

IF(V4R_ORGT_EVALUATOR)

find_package(Boost REQUIRED COMPONENTS filesystem regex)

include_directories(${OPENCV_INCLUDE_DIRS}
                    ${PCL_INCLUDE_DIRS}
                     ${Boost_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS} ${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS} ${Boost_LIB_DIAGNOSTIC_DEFINITIONS})

PROJECT(v4rORGTEvaluator)

SET(SOURCE_CPP
  or_evaluator.cpp
)


SET(SOURCE_H
  or_evaluator.h
  or_evaluator.hpp
)


add_library(${PROJECT_NAME} SHARED ${SOURCE_H} ${SOURCE_CPP})
target_link_libraries(${PROJECT_NAME} ${OPENCV_LIBRARIES} ${PCL_LIBRARIES} v4rORUtils ${Boost_LIBRARIES})
v4r_add_library(${PROJECT_NAME} "${SOURCE_H}")

add_executable(create_images create_images.cpp)
target_link_libraries(create_images v4rORUtils ${OPENCV_LIBRARIES} ${PCL_LIBRARIES} v4rORFramework ${PROJECT_NAME} ${Boost_LIBRARIES})

add_executable(visualize_gt visualize_gt.cpp)
target_link_libraries(visualize_gt v4rORUtils ${OPENCV_LIBRARIES} ${PCL_LIBRARIES} v4rORFramework ${PROJECT_NAME} ${Boost_LIBRARIES})

FIND_PACKAGE(Qt4)
include(${QT_USE_FILE})
add_definitions(-DQT_NO_KEYWORDS)
QT4_WRAP_CPP(foo_MOC_SRCS main_window.h)
add_executable(GT6DOF GT6DOF.cpp ${foo_MOC_SRCS} main_window.cpp)
target_link_libraries(GT6DOF v4rORGTEvaluator v4rORFramework v4rORUtils ${PCL_LIBRARIES} ${QT_LIBRARIES} QVTK  ${Boost_LIBRARIES})

ENDIF(V4R_ORGT_EVALUATOR)



