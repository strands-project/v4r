project(${EDT_LIBRARY})

v4r_assert(HAVE_PCL)

v4r_include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${PCL_INCLUDE_DIRS})

set(lib_srcs
  propagation_distance_field.cpp
)

SET(lib_hdrs
  distance_field.h
  voxel_grid.h
  propagation_distance_field.h
)

add_library(${EDT_LIBRARY} STATIC ${lib_srcs})
target_link_libraries(${EDT_LIBRARY} ${PCL_LIBRARIES})

set_target_properties(${EDT_LIBRARY}
  PROPERTIES
    POSITION_INDEPENDENT_CODE TRUE
    OUTPUT_NAME "${EDT_LIBRARY}"
    ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
)

install(FILES ${lib_hdrs} DESTINATION "${V4R_3P_INCLUDE_INSTALL_PATH}/EDT")

if(NOT BUILD_SHARED_LIBS)
  v4r_install_target(${EDT_LIBRARY} EXPORT V4RModules ARCHIVE DESTINATION ${V4R_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()
