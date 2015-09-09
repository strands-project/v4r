project(${LIBSVM_LIBRARY})
include(ExternalProject)

# Make a target that downloads and builds LIBSVM from source.
# The install command copies the (only) library header to the root libsvm directory. This is needed to allow using V4R
# from without installation (directly from build/ location).
set(_external_target "${LIBSVM_LIBRARY}_external")
ExternalProject_Add(${_external_target}
  PREFIX "${CMAKE_CURRENT_LIST_DIR}"
  URL "https://github.com/cjlin1/libsvm/archive/v320.tar.gz"
  URL_HASH SHA256=e4be7fc8d2e7cb65feae4d6387967484c01c9dc818e7dfff515f663728a6f2ca
  TIMEOUT 30
  CONFIGURE_COMMAND ""
  BUILD_COMMAND make lib
  BUILD_IN_SOURCE 1
  INSTALL_COMMAND cp <SOURCE_DIR>/svm.h ${CMAKE_CURRENT_SOURCE_DIR}
)

# SOURCE_DIR is where the source was unpacked and then built.
# We want to make this available outside of the scope of this file.
ExternalProject_Get_Property(${_external_target} SOURCE_DIR)
set(${LIBSVM_LIBRARY}_SOURCE_DIR ${SOURCE_DIR} CACHE INTERNAL "LIBSVM source directory")

# Let CMake know about the shared library built by the external project target.
add_library(${LIBSVM_LIBRARY} SHARED IMPORTED GLOBAL)
add_dependencies(${LIBSVM_LIBRARY} ${_external_target})
set_target_properties(${LIBSVM_LIBRARY}
  PROPERTIES
    IMPORTED_LOCATION "${SOURCE_DIR}/libsvm.so.2"
    OUTPUT_NAME "${LIBSVM_LIBRARY}"
    ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
)

# Setup a rule to install header files
install(FILES "${SOURCE_DIR}/svm.h" DESTINATION "${V4R_3P_INCLUDE_INSTALL_PATH}/libsvm")

if(NOT BUILD_SHARED_LIBS)
  message(WARNING "LIBSVM support is enabled, but it can only be built as as shared library.")
endif()
