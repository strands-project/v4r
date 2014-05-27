SET(V4R_SVM 1 CACHE BOOL "Support Vector Machine - based on libsvm." )
SET(CMAKE_BUILD_TYPE "Debug")

IF(V4R_SVM)

PROJECT(v4rsvm)

find_package(PCL REQUIRED)
find_package(OpenCV 2.4 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

SET(SOURCE_CPP
  svm.cpp
  SVMFileCreator.cpp
  SVMPredictorSingle.cpp
  SVMScale.cpp
  SVMTrainModel.cpp
  svmWrapper.cpp
)

SET(SOURCE_H
  svm.h
  SVMFileCreator.h
  SVMPredictorSingle.h
  SVMScale.h
  SVMTrainModel.h
  svmWrapper.h
)

SET(SOURCE_MODEL
  model/param.txt
  model/param.txt.light
  model/param2.txt
  model/PP-Trainingsset.txt.scaled.model
  model/PP-Trainingsset.txt.scaled.model.light
  model/PP2-Trainingsset.txt.scaled.model
)

add_library(${PROJECT_NAME} SHARED ${SOURCE_H} ${SOURCE_CPP})
target_link_libraries(${PROJECT_NAME} ${OPENCV_LIBRARIES} ${PCL_LIBRARIES})
v4r_add_library(${PROJECT_NAME} "${SOURCE_H}")

# copy model to system
string(REGEX REPLACE "v4r" "" SVM_INCLUDE_NAME ${PROJECT_NAME})
set(SVM_MODEL ${CMAKE_INSTALL_PREFIX}/include/v4r/${SVM_INCLUDE_NAME}/model/)
add_definitions(-DSVM_MODEL="${SVM_MODEL}")
message(STATUS "SVM_MODEL is ${SVM_MODEL}")

install(DIRECTORY DESTINATION include/v4r/${SVM_INCLUDE_NAME}/model)
install(FILES ${SOURCE_MODEL} DESTINATION include/v4r/${SVM_INCLUDE_NAME}/model)

ENDIF(V4R_SVM)

