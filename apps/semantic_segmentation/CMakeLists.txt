SET(SEMSEG_DEPS v4r_core v4r_semantic_segmentation)
v4r_check_dependencies(${SEMSEG_DEPS})

if(NOT V4R_DEPENDENCIES_FOUND)
	message(***semantic_segmentation does not meet dependencies*****)
	return()
endif()

v4r_include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
v4r_include_modules(${SEMSEG_DEPS})

if(HAVE_PCL)
  v4r_include_directories(${PCL_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${PCL_LIBRARIES})
endif()

if(HAVE_OPENCV)
  v4r_include_directories(${OpenCV_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${OpenCV_LIBS})
endif()

if(HAVE_BOOST)
  v4r_include_directories(${BOOST_INCLUDE_DIRS})
  list(APPEND DEP_LIBS ${BOOST_LIBRARIES})
endif()

add_executable(train3DEF train3DEF.cpp )
add_executable(evaluate3DEF evaluate3DEF.cpp )
add_executable(merge3DEF merge3DEF.cpp )
add_executable(updateleafs3DEF updateleafs3DEF.cpp )
add_executable(analyze3DEF analyze3DEF.cpp )
add_executable(merged_supervoxels_demo segmentation_demo.cpp)
add_executable(create_3DEF_trainingdata create_trainingdata.cpp)
add_executable(semseg_demo semseg_demo.cpp)

target_link_libraries(train3DEF ${SEMSEG_DEPS} ${DEP_LIBS})
target_link_libraries(evaluate3DEF ${SEMSEG_DEPS} ${DEP_LIBS})
target_link_libraries(merge3DEF ${SEMSEG_DEPS} ${DEP_LIBS})
target_link_libraries(updateleafs3DEF ${SEMSEG_DEPS} ${DEP_LIBS})
target_link_libraries(analyze3DEF ${SEMSEG_DEPS} ${DEP_LIBS})
target_link_libraries(merged_supervoxels_demo ${SEMSEG_DEPS} ${DEP_LIBS})
target_link_libraries(create_3DEF_trainingdata ${SEMSEG_DEPS} ${DEP_LIBS})
target_link_libraries(semseg_demo ${SEMSEG_DEPS} ${DEP_LIBS})

INSTALL(TARGETS train3DEF evaluate3DEF merge3DEF updateleafs3DEF analyze3DEF merged_supervoxels_demo create_3DEF_trainingdata semseg_demo
  RUNTIME DESTINATION bin 
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
