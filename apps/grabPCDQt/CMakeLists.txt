set(APPS_GRAB_PCD_QT 0 CACHE BOOL "Qt GUI grabing PCD files from Kinect-like sensors" )
if(APPS_GRAB_PCD_QT)

project(grabPCDQt)


find_package(Qt4 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})

# OpenNI 2
find_package(PkgConfig REQUIRED)
pkg_search_module(OpenNI2 REQUIRED libopenni2)
include_directories(${OpenNI2_INCLUDE_DIRS})
link_directories(${OpenNI2_LIBRARY_DIRS})

# PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(QT_USE_QTOPENGL TRUE)
include(${QT_USE_FILE})

add_definitions(${QT_DEFINITIONS})
add_definitions(-Dlinux -D__x86_64__)

include_directories(${QT_INCLUDE_DIR})
include_directories(${QT_INCLUDE_DIR}/QtOpenGL)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

message(STATUS "QT_INCLUDE_DIR: ${QT_INCLUDE_DIR}")
message(STATUS "QT_USE_FILE: ${QT_USE_FILE}")
message(STATUS "QT_DEFINITIONS: ${QT_DEFINITIONS}")
message(STATUS "QT_LIBRARIES: ${QT_LIBRARIES}")


set(SOURCES
        glviewer.cpp
        main.cpp
        mainwindow.cpp
        params.cpp
        sensor.cpp
)

set(HEADERS
        glviewer.h
        mainwindow.h
        params.h
        sensor.h
)

set(FORMS
        mainwindow.ui
        params.ui
)

QT4_WRAP_CPP(HEADERS_MOC ${HEADERS})
QT4_WRAP_UI(FORMS_MOC ${FORMS})

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS_MOC} ${FORMS_MOC})
target_link_libraries(${PROJECT_NAME} ${OPENCV_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${QT_LIBRARIES})
target_link_libraries(${PROJECT_NAME} v4rTomGine5 v4rCalibrationRGBD)

endif(APPS_GRAB_PCD_QT)
