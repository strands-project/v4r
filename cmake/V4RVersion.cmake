set(V4R_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/modules/core/include/v4r/core/version.h")
file(STRINGS "${V4R_VERSION_FILE}" V4R_VERSION_PARTS REGEX "#define V4R_VERSION_[A-Z]+[ ]+" )

string(REGEX REPLACE ".+V4R_VERSION_MAJOR[ ]+([0-9]+).*" "\\1" V4R_VERSION_MAJOR "${V4R_VERSION_PARTS}")
string(REGEX REPLACE ".+V4R_VERSION_MINOR[ ]+([0-9]+).*" "\\1" V4R_VERSION_MINOR "${V4R_VERSION_PARTS}")
string(REGEX REPLACE ".+V4R_VERSION_REVISION[ ]+([0-9]+).*" "\\1" V4R_VERSION_PATCH "${V4R_VERSION_PARTS}")
string(REGEX REPLACE ".+V4R_VERSION_STATUS[ ]+\"([^\"]*)\".*" "\\1" V4R_VERSION_STATUS "${V4R_VERSION_PARTS}")

set(V4R_VERSION_PLAIN "${V4R_VERSION_MAJOR}.${V4R_VERSION_MINOR}.${V4R_VERSION_PATCH}")

set(V4R_VERSION "${V4R_VERSION_PLAIN}${V4R_VERSION_STATUS}")

set(V4R_SOVERSION "${V4R_VERSION_MAJOR}.${V4R_VERSION_MINOR}")
set(V4R_LIBVERSION "${V4R_VERSION_MAJOR}.${V4R_VERSION_MINOR}.${V4R_VERSION_PATCH}")

# create a dependency on version file
# we never use output of the following command but cmake will rerun automatically if the version file changes
configure_file("${V4R_VERSION_FILE}" "${CMAKE_BINARY_DIR}/junk/version.junk" COPYONLY)
