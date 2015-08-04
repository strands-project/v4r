find_path(EIGEN_INCLUDE_PATH "Eigen/Core"
          PATHS /usr/local /opt /usr $ENV{EIGEN_ROOT}/include ENV ProgramFiles ENV ProgramW6432
          PATH_SUFFIXES include/eigen3 Eigen/include/eigen3
          DOC "The path to Eigen3 headers"
          CMAKE_FIND_ROOT_PATH_BOTH)

if(EIGEN_INCLUDE_PATH)
  v4r_include_directories(${EIGEN_INCLUDE_PATH})
  v4r_parse_header("${EIGEN_INCLUDE_PATH}/Eigen/src/Core/util/Macros.h" EIGEN_VERSION_LINES EIGEN_WORLD_VERSION EIGEN_MAJOR_VERSION EIGEN_MINOR_VERSION)
  set(HAVE_EIGEN TRUE)
endif()
