find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

FIND_PACKAGE(PCL QUIET)
IF(${PCL_FOUND})
  include_directories(${PCL_INCLUDE_DIRS})
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})

  #add_executable(GaborFilters GaborFilters.cpp)
  #target_link_libraries(GaborFilters ${OpenCV_LIBS} v4rEPUtils v4rAttentionModule)

#   add_executable(GaborFilters2 GaborFilters2.cpp)
#   target_link_libraries(GaborFilters2 ${OpenCV_LIBS} v4rEPUtils)
# 
#   add_executable(exampleSurfaceHeightMapSingleImage exampleSurfaceHeightMapSingleImage.cpp)
#   target_link_libraries(exampleSurfaceHeightMapSingleImage ${OpenCV_LIBS} ${PCL_LIBRARIES} v4rEPUtils v4rAttentionModule)
#  
#   add_executable(exampleSurfaceHeightMapPyramidSingleImage exampleSurfaceHeightMapPyramidSingleImage.cpp)
#   target_link_libraries(exampleSurfaceHeightMapPyramidSingleImage ${OpenCV_LIBS} ${PCL_LIBRARIES} v4rEPUtils v4rAttentionModule)
# 
#   add_executable(exampleSurfaceHeightMapManyImages exampleSurfaceHeightMapManyImages.cpp)
#   target_link_libraries(exampleSurfaceHeightMapManyImages ${OpenCV_LIBS} ${PCL_LIBRARIES} v4rEPUtils v4rAttentionModule)
# 
#   add_executable(exampleRelativeSurfaceOrientationMapSingleImage exampleRelativeSurfaceOrientationMapSingleImage.cpp)
#   target_link_libraries(exampleRelativeSurfaceOrientationMapSingleImage ${OpenCV_LIBS} ${PCL_LIBRARIES} v4rEPUtils v4rAttentionModule)
# 
#   add_executable(exampleRelativeSurfaceOrientationMapPyramidSingleImage exampleRelativeSurfaceOrientationMapPyramidSingleImage.cpp)
#   target_link_libraries(exampleRelativeSurfaceOrientationMapPyramidSingleImage ${OpenCV_LIBS} ${PCL_LIBRARIES} v4rEPUtils v4rAttentionModule)
# 
#   add_executable(exampleSurfaceCurvatureMapSingleImage exampleSurfaceCurvatureMapSingleImage.cpp)
#   target_link_libraries(exampleSurfaceCurvatureMapSingleImage ${OpenCV_LIBS} ${PCL_LIBRARIES} v4rEPUtils v4rAttentionModule)
# 
#   add_executable(exampleSurfaceCurvatureMapPyramidSingleImage exampleSurfaceCurvatureMapPyramidSingleImage.cpp)
#   target_link_libraries(exampleSurfaceCurvatureMapPyramidSingleImage ${OpenCV_LIBS} ${PCL_LIBRARIES} v4rEPUtils v4rAttentionModule)

#   add_executable(exampleIKNSaliencyMapSingleImage exampleIKNSaliencyMapSingleImage.cpp)
#   target_link_libraries(exampleIKNSaliencyMapSingleImage ${OpenCV_LIBS} v4rEPUtils v4rAttentionModule)
# 
#   add_executable(exampleMapCombination exampleMapCombination.cpp)
#   target_link_libraries(exampleMapCombination ${OpenCV_LIBS} ${PCL_LIBRARIES} v4rEPUtils v4rAttentionModule)

  #add_executable(Exmpl3DSymSalMap Exmpl3DSymSalMap.cpp)
  #target_link_libraries(Exmpl3DSymSalMap ${OpenCV_LIBS} ${PCL_LIBRARIES} v4rEPUtils v4rAttentionModule)
  
  add_executable(ExmplSalLine ExmplSalLine.cpp)
  target_link_libraries(ExmplSalLine ${OpenCV_LIBS} ${PCL_LIBRARIES} v4rEPUtils v4rAttentionModule)

  add_executable(ExmplWTA ExmplWTA.cpp)
  target_link_libraries(ExmplWTA ${OpenCV_LIBS} ${PCL_LIBRARIES} v4rEPUtils v4rAttentionModule)

  add_executable(ExmplMSR ExmplMSR.cpp)
  target_link_libraries(ExmplMSR ${OpenCV_LIBS} ${PCL_LIBRARIES} v4rEPUtils v4rAttentionModule)

ELSE (${PCL_FOUND})
  MESSAGE("\nCould not locate PCL. Skip attention module examples\n")
ENDIF(${PCL_FOUND})


