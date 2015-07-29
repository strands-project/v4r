project(${EDT_LIBRARY})

set(lib_srcs
  propagation_distance_field.cpp
)

#SET(SOURCE_H
  #distance_field.h
  #voxel_grid.h
  #propagation_distance_field.h
#)

#include_directories(${PCL_INCLUDE_DIRS})
add_library(${EDT_LIBRARY} SHARED ${lib_srcs})
target_link_libraries(${EDT_LIBRARY} ${PCL_LIBRARIES})

set_target_properties(${EDT_LIBRARY}
  PROPERTIES
  OUTPUT_NAME "${EDT_LIBRARY}"
  ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
)

if(NOT BUILD_SHARED_LIBS)
  v4r_install_target(${EDT_LIBRARY} EXPORT V4RModules ARCHIVE DESTINATION ${V4R_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()
