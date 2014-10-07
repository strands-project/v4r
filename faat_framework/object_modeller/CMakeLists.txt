cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

PROJECT(object_modeller)

set(LIB_NAME "object_modeller_lib")

find_package(PCL REQUIRED)
IF(V4R_FAAT_TRUNK)

ELSE(V4R_FAAT_TRUNK)
    find_package(V4R REQUIRED)
    find_package(FAAT_PCL REQUIRED)
ENDIF()

find_package(Qt4 REQUIRED)
find_package(OpenCV)
find_package(OpenGL)
FIND_PACKAGE(VTK)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(Ceres)
if(${Ceres_FOUND})
    include_directories(${CERES_INCLUDES})
else(${Ceres_FOUND})
    add_definitions(-DKP_NO_CERES_AVAILABLE)
endif(${Ceres_FOUND})

IF(VTK_FOUND)
  # Use Qt that the QVTK library was built with
  SET(QT_QMAKE_EXECUTABLE ${VTK_QT_QMAKE_EXECUTABLE} CACHE FILEPATH "")
  SET(DESIRED_QT_VERSION ${VTK_DESIRED_QT_VERSION} CACHE FILEPATH "")
  FIND_PACKAGE(Qt4)
ENDIF(VTK_FOUND)

IF(VTK_FOUND AND QT_FOUND)
  INCLUDE(${VTK_USE_FILE})
ENDIF(VTK_FOUND AND QT_FOUND)

# qt
set(QT_USE_QTOPENGL TRUE)
include(${QT_USE_FILE})

add_definitions(${QT_DEFINITIONS})
add_definitions(-Dlinux -D__x86_64__)

include_directories(${QT_INCLUDE_DIR})
include_directories(${QT_INCLUDE_DIR}/QtOpenGL)
include_directories(${CMAKE_CURRENT_BINARY_DIR})



include_directories(${PCL_INCLUDE_DIRS}
                    ${FAAT_PCL_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/test)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


set(lib_incs
    include/module.h
    include/multisequence.h
    include/config.h
    include/inputModule.h
    include/outputModule.h
    include/ioModule.h
    include/pipeline.h
    include/reader/fileReader.h
    include/reader/poseReader.h
    include/reader/cameraReader.h
    include/reader/trackingCameraReader.h
    include/reader/turntableReader.h
    include/reader/memory_friendly_fileReader.h
    include/output/rendererArgs.h
    include/output/meshRenderer.h
    include/output/texturedMeshRenderer.h
    include/output/pclRenderer.h
    include/output/windowedPclRenderer.h
    include/output/pointCloudWriter.h
    include/output/indicesWriter.h
    include/output/posesWriter.h
    include/output/renderer.h
    include/output/roi.h
    include/output/pointCloudRenderer.h
    include/texturing/shadingTexture.h
    include/texturing/pclTexture.h
    include/registration/cameraTracker.h
    include/registration/icp.h
    include/registration/checkerboard.h
    include/registration/globalRegistration.h
    include/segmentation/dominantPlaneExtraction.h
    include/segmentation/checkerboardSegmentation.h
    include/multisequence/siftFeatureMatcher.h
    include/segmentation/notNaNSegmentation.h
    include/segmentation/ROISegmentation.h
    include/util/mask.h
    include/util/transform.h
    include/util/convert.h
    include/util/distanceFilter.h
    include/util/boxFilter.h
    include/util/multiplyMatrix.h
    include/util/normalEstimationOmp.h
    include/util/integralImageNormalEstimation.h
    include/util/vectorMask.h
    include/util/nguyenNoiseWeights.h
    include/modelling/nmBasedCloudIntegration.h
    include/modelling/poissonReconstruction.h
    include/result.h
    include/resultAggregator.h
    include/eventManager.h
    include/pipelineFactory.h
    include/visualConfig.h
)

set(lib_srcs
    src/pipeline.cpp
    src/config.cpp
    src/module.cpp
    src/result.cpp
    src/reader/fileReader.cpp
    src/reader/poseReader.cpp
    src/reader/cameraReader.cpp
    src/reader/trackingCameraReader.cpp
    src/reader/turntableReader.cpp
    src/reader/memory_friendly_fileReader.cpp
    src/output/pclRenderer.cpp
    src/output/windowedPclRenderer.cpp
    src/output/meshRenderer.cpp
    src/output/texturedMeshRenderer.cpp
    src/output/pointCloudWriter.cpp
    src/output/indicesWriter.cpp
    src/output/posesWriter.cpp
    src/output/roi.cpp
    src/output/pointCloudRenderer.cpp
    src/texturing/shadingTexture.cpp
    src/texturing/pclTexture.cpp
    src/registration/icp.cpp
    src/registration/cameraTracker.cpp
    src/registration/checkerboard.cpp
    src/registration/globalRegistration.cpp
    src/segmentation/dominantPlaneExtraction.cpp
    src/segmentation/checkerboardSegmentation.cpp
    src/multisequence/siftFeatureMatcher.cpp
    src/segmentation/notNaNSegmentation.cpp
    src/segmentation/ROISegmentation.cpp
    src/util/mask.cpp
    src/util/transform.cpp
    src/util/distanceFilter.cpp
    src/util/boxFilter.cpp
    src/util/multiplyMatrix.cpp
    src/util/normalEstimationOmp.cpp
    src/util/integralImageNormalEstimation.cpp
    src/util/vectorMask.cpp
    src/util/nguyenNoiseWeights.cpp
    src/modelling/nmBasedCloudIntegration.cpp
    src/modelling/poissonReconstruction.cpp
    src/pipelineFactory.cpp
    src/visualConfig.cpp
)

add_library(${LIB_NAME} SHARED ${lib_srcs} ${lib_incs})

target_link_libraries(${LIB_NAME} ${OPENCV_LIBRARIES} ${PCL_LIBRARIES} ${FAAT_PCL_LIBRARIES} ${LIB_NAME} v4rKeypointCameraTrackerPCL v4rKeypointBase v4rKeypointCameraTracker v4rTomGine v4rTomGinePCL pthread boost_system boost_filesystem boost_regex opencv_core opencv_highgui opencv_imgproc opencv_calib3d opencv_features2d opencv_nonfree opencv_video)

#target_link_libraries(${LIB_NAME} ${OPENCV_LIBRARIES} ${PCL_LIBRARIES} ${FAAT_PCL_LIBRARIES} ${LIB_NAME} v4rKeypointCameraTrackerPCL v4rKeypointBase v4rKeypointCameraTracker v4rTomGine v4rTomGinePCL boost_system boost_filesystem boost_regex opencv_core opencv_highgui opencv_imgproc opencv_calib3d opencv_features2d opencv_nonfree opencv_video)
#target_link_libraries(${LIB_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${FAAT_PCL_LIBRARIES})


add_executable(${PROJECT_NAME} test/object_modeller_cmd.cpp)

target_link_libraries(${PROJECT_NAME} ${LIB_NAME} ${OPENCV_LIBRARIES} ${PCL_LIBRARIES} ${FAAT_PCL_LIBRARIES} v4rKeypointCameraTrackerPCL v4rKeypointBase v4rKeypointCameraTracker pthread boost_system boost_filesystem boost_regex opencv_core opencv_highgui opencv_imgproc opencv_calib3d opencv_features2d opencv_nonfree opencv_video)


set(lib_incs_ui
    test/ui/mainwindow.h
    test/ui/settings.h
    test/ui/manager.h
    test/ui/glrenderer.h
    test/ui/parameteredit.h
)

set(lib_srcs_ui
    test/ui/mainwindow.cpp
    test/ui/settings.cpp
    test/ui/manager.cpp
    test/ui/glrenderer.cpp
    test/ui/main.cpp
)

set(forms
        test/ui/mainwindow.ui
        test/ui/settings.ui
)

QT4_WRAP_CPP(lib_incs_ui_moc ${lib_incs_ui})
QT4_WRAP_UI(forms_moc ${forms})

add_executable("object_modeller_ui" ${lib_srcs_ui} ${lib_incs_ui_moc} ${forms_moc})
target_link_libraries("object_modeller_ui" ${LIB_NAME} QVTK vtkRendering ${OPENGL_LIBRARIES} ${OPENCV_LIBRARIES} ${PCL_LIBRARIES} ${QT_LIBRARIES} ${FAAT_PCL_LIBRARIES} v4rKeypointCameraTrackerPCL v4rKeypointBase v4rKeypointCameraTracker pthread boost_system boost_filesystem boost_regex opencv_core opencv_highgui opencv_imgproc opencv_calib3d opencv_features2d opencv_nonfree opencv_video)

#target_link_libraries(${PROJECT_NAME} ${LIB_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${FAAT_PCL_LIBRARIES} v4rKeypointCameraTrackerPCL v4rKeypointBase v4rKeypointCameraTracker v4rTomGine v4rTomGinePCL boost_system boost_filesystem boost_regex)

add_executable(hobbit_object_modeller test/hobbit_object_modeller_cmd.cpp)
target_link_libraries(hobbit_object_modeller ${LIB_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${FAAT_PCL_LIBRARIES} v4rKeypointCameraTrackerPCL v4rKeypointBase v4rKeypointCameraTracker boost_system boost_filesystem boost_regex)

add_executable(turn_table_pose test/turn_table_pose.cpp)
target_link_libraries(turn_table_pose ${PCL_LIBRARIES})

