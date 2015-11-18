project(${GLFW3_LIBRARY})
include(ExternalProject)

# Make a target that downloads and builds GLFW3 from source.
set(_external_target "${GLFW3_LIBRARY}_external")
ExternalProject_Add(${_external_target}
  PREFIX "${CMAKE_CURRENT_LIST_DIR}"
  URL "https://github.com/glfw/glfw/releases/download/3.1.2/glfw-3.1.2.zip"
  URL_HASH SHA1=6285882cea7d562797adaa194f940ba0e938c2b1
  TIMEOUT 30
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
             -DGLFW_BUILD_DOCS=OFF -DGLFW_BUILD_EXAMPLES=OFF -DGLFW_BUILD_TESTS=OFF
)

# INSTALL_DIR is where the build library has been installed.
# We want to make this available outside of the scope of this file.
ExternalProject_Get_Property(${_external_target} INSTALL_DIR)
set(${GLFW3_LIBRARY}_INSTALL_DIR ${INSTALL_DIR} CACHE INTERNAL "GLFW3 install directory")

# Let CMake know about the static library built by the external project target.
add_library(${GLFW3_LIBRARY} STATIC IMPORTED GLOBAL)
add_dependencies(${GLFW3_LIBRARY} ${_external_target})
set_target_properties(${GLFW3_LIBRARY}
  PROPERTIES
    IMPORTED_LOCATION "${INSTALL_DIR}/lib/libglfw3.a"
    OUTPUT_NAME "${GLFW3_LIBRARY}"
    ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
)

# Setup a rule to install header and library files
install(DIRECTORY "${INSTALL_DIR}/include/GLFW" DESTINATION "${V4R_3P_INCLUDE_INSTALL_PATH}")

if(NOT BUILD_SHARED_LIBS)
  # Normally we should use "v4r_install_target", however "libglfw3" is not a target, and it will not work.
  # To make CMake happy, install the library excplicitly.
  install(FILES "${INSTALL_DIR}/lib/libgflw3.a" DESTINATION ${V4R_3P_LIB_INSTALL_PATH})
endif()

