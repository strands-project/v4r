cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

PROJECT(object_modeller_ui)

set(LIB_NAME "object_modeller_lib")

find_package(PCL REQUIRED)
find_package(V4R REQUIRED)
find_package(FAAT_PCL REQUIRED)
find_package(OpenCV)

include_directories(${PCL_INCLUDE_DIRS}
                    ${FAAT_PCL_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


set(lib_incs
    mainwindow.h
)

set(lib_srcs
    mainwindow.cpp
    main.cpp
)

add_executable(${PROJECT_NAME} ${lib_incs} ${libsrcs})
target_link_libraries(${PROJECT_NAME} ${LIB_NAME} ${OPENCV_LIBRARIES} ${PCL_LIBRARIES} ${FAAT_PCL_LIBRARIES} v4rKeypointCameraTrackerPCL v4rKeypointBase v4rKeypointCameraTracker v4rTomGine v4rTomGinePCL pthread boost_system boost_filesystem boost_regex opencv_core opencv_highgui opencv_imgproc opencv_calib3d opencv_features2d opencv_nonfree opencv_video)
