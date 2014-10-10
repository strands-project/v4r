find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

FIND_PACKAGE(PCL QUIET)
IF(${PCL_FOUND})
  include_directories(${PCL_INCLUDE_DIRS})
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})

  add_executable(ExmplDrawPoints ExmplDrawPoints.cpp)
  target_link_libraries(ExmplDrawPoints ${OpenCV_LIBS} v4rEPUtils v4rAttentionModule)
  
  add_executable(ExmplXYZRGB2Disparity ExmplXYZRGB2Disparity.cpp)
  target_link_libraries(ExmplXYZRGB2Disparity ${OpenCV_LIBS} v4rEPUtils v4rAttentionModule)

ELSE (${PCL_FOUND})
  MESSAGE("\nCould not locate PCL. Skip attention module examples\n")
ENDIF(${PCL_FOUND})

