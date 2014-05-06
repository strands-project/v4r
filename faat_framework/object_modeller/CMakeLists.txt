cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

PROJECT(object_modeller)

set(LIB_NAME "object_modeller_lib")

find_package(PCL REQUIRED)
#find_package(V4R REQUIRED)
#find_package(FAAT_PCL REQUIRED)
find_package(OpenCV)

include_directories(${PCL_INCLUDE_DIRS}
                    ${FAAT_PCL_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


set(lib_incs
    include/module.h
    include/config.h
    include/inputModule.h
    include/outputModule.h
    include/ioModule.h
    include/pipeline.h
    include/reader/fileReader.h
    include/output/tgRenderer.h
    include/output/rendererArgs.h
    include/output/meshRenderer.h
    include/output/pclRenderer.h
    include/output/pointCloudWriter.h
    include/output/indicesWriter.h
    include/output/posesWriter.h
    include/output/renderer.h
    include/output/pointCloudRenderer.h
    include/registration/cameraTracker.h
    include/registration/checkerboard.h
    include/registration/globalRegistration.h
    include/segmentation/dominantPlaneExtraction.h
    include/segmentation/checkerboardSegmentation.h
    include/util/mask.h
    include/util/transform.h
    include/util/distanceFilter.h
    include/util/multiplyMatrix.h
    include/util/normalEstimationOmp.h
    include/util/integralImageNormalEstimation.h
    include/util/vectorMask.h
    include/util/nguyenNoiseWeights.h
    include/modelling/nmBasedCloudIntegration.h
    include/modelling/poissonReconstruction.h
    include/result.h
)

set(lib_srcs
    src/pipeline.cpp
    src/config.cpp
    src/module.cpp
    src/reader/fileReader.cpp
    src/output/tgRenderer.cpp
    src/output/pclRenderer.cpp
    src/output/meshRenderer.cpp
    src/output/pointCloudWriter.cpp
    src/output/indicesWriter.cpp
    src/output/posesWriter.cpp
    src/output/pointCloudRenderer.cpp
    src/registration/cameraTracker.cpp
    src/registration/checkerboard.cpp
    src/registration/globalRegistration.cpp
    src/segmentation/dominantPlaneExtraction.cpp
    src/segmentation/checkerboardSegmentation.cpp
    src/util/mask.cpp
    src/util/transform.cpp
    src/util/distanceFilter.cpp
    src/util/multiplyMatrix.cpp
    src/util/normalEstimationOmp.cpp
    src/util/integralImageNormalEstimation.cpp
    src/util/vectorMask.cpp
    src/util/nguyenNoiseWeights.cpp
    src/modelling/nmBasedCloudIntegration.cpp
    src/modelling/poissonReconstruction.cpp
)

add_library(${LIB_NAME} SHARED ${lib_srcs} ${lib_incs})
target_link_libraries(${LIB_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${FAAT_PCL_LIBRARIES} ${LIB_NAME} v4rKeypointCameraTrackerPCL v4rKeypointBase v4rKeypointCameraTracker v4rTomGine v4rTomGinePCL boost_system boost_filesystem boost_regex)

add_executable(${PROJECT_NAME} test/object_modeller_cmd.cpp)
target_link_libraries(${PROJECT_NAME} ${LIB_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${FAAT_PCL_LIBRARIES} v4rKeypointCameraTrackerPCL v4rKeypointBase v4rKeypointCameraTracker v4rTomGine v4rTomGinePCL boost_system boost_filesystem boost_regex)
