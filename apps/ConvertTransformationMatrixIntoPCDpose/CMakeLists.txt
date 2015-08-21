SET(V4R_SAVE_POSE_INTO_PCD_DEPS v4r_io v4r_core v4r_common)
v4r_check_dependencies(${V4R_SAVE_POSE_INTO_PCD_DEPS})

if(NOT V4R_DEPENDENCIES_FOUND)
  return()
endif()

v4r_include_modules(${V4R_SAVE_POSE_INTO_PCD_DEPS})

if(HAVE_PCL)
  v4r_include_directories(${PCL_INCLUDE_DIRS})
  list(APPEND V4R_SAVE_POSE_INTO_PCD_DEPS ${PCL_LIBRARIES} ${BOOST_LIBRARIES})
endif()


project(save_transformation_matrix_into_pcd_header)

set(the_target v4r_save_transformation_matrix_into_pcd_header)


add_executable(${the_target} save_pose_into_pcd.cpp)
target_link_libraries(${the_target} ${V4R_SAVE_POSE_INTO_PCD_DEPS})

set_target_properties(${the_target} PROPERTIES
                      DEBUG_POSTFIX "${V4R_DEBUG_POSTFIX}"
                      ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
                      RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
                      INSTALL_NAME_DIR lib
                      OUTPUT_NAME ${the_target})

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${the_target} PROPERTIES FOLDER "applications")
endif()

if(INSTALL_CREATE_DISTRIB)
  if(BUILD_SHARED_LIBS)
    install(TARGETS ${the_target} RUNTIME DESTINATION ${V4R_BIN_INSTALL_PATH} CONFIGURATIONS Release COMPONENT dev)
  endif()
else()
  install(TARGETS ${the_target} RUNTIME DESTINATION ${V4R_BIN_INSTALL_PATH} COMPONENT dev)
endif()
