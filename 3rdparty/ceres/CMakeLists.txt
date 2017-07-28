project(${CERES_LIBRARY})
include(ExternalProject)

set(_external_target "${CERES_LIBRARY}_external")
ExternalProject_Add(${_external_target}
  PREFIX "${CMAKE_CURRENT_LIST_DIR}"
  URL "http://ceres-solver.org/ceres-solver-1.12.0.tar.gz"
  URL_HASH SHA256=745bfed55111e086954126b748eb9efe20e30be5b825c6dec3c525cf20afc895
  TIMEOUT 30
  PATCH_COMMAND patch -p0  < ../../fix_eigen_vector.patch
  CMAKE_ARGS "-DBUILD_SHARED_LIBS=ON" "-DBUILD_EXAMPLES=OFF" "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
  BUILD_IN_SOURCE 1
  BUILD_COMMAND ${CMAKE_BUILD_TOOL}
)

# SOURCE_DIR is where the source was unpacked and then built.
# We want to make this available outside of the scope of this file.
ExternalProject_Get_Property(${_external_target} SOURCE_DIR)
ExternalProject_Get_Property(${_external_target} INSTALL_DIR)
set(${CERES_LIBRARY}_SOURCE_DIR ${SOURCE_DIR} CACHE INTERNAL "Ceres solver source directory")
set(${CERES_LIBRARY}_INCLUDE_DIR "${INSTALL_DIR}/include" CACHE INTERNAL "Ceres solver include directories")
set(${CERES_LIBRARY}_VERSION "1.12.0" CACHE INTERNAL "Ceres solver version")

# Let CMake know about the shared library built by the external project target.
add_library(${CERES_LIBRARY} SHARED IMPORTED GLOBAL)
add_dependencies(${CERES_LIBRARY} ${_external_target})
set_target_properties(${CERES_LIBRARY}
  PROPERTIES
    IMPORTED_LOCATION "${INSTALL_DIR}/lib/libceres.so"
    OUTPUT_NAME "${CERES_LIBRARY}"
    ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
)

# Setup a rule to install header files
install(DIRECTORY "${INSTALL_DIR}/include/ceres" DESTINATION "${V4R_3P_INCLUDE_INSTALL_PATH}")
install(FILES "${INSTALL_DIR}/lib/libceres.so" DESTINATION ${V4R_3P_LIB_INSTALL_PATH})
install(FILES "${INSTALL_DIR}/lib/libceres.so.1" DESTINATION ${V4R_3P_LIB_INSTALL_PATH})
install(FILES "${INSTALL_DIR}/lib/libceres.so.1.12.0" DESTINATION ${V4R_3P_LIB_INSTALL_PATH})

if(NOT BUILD_SHARED_LIBS)
    message("we need to build with BUILD_SHARED_LIBS set to ON")
endif()
