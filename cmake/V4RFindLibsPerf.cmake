# ----------------------------------------------------------------------------
#  Detect other 3rd-party performance and math libraries
# ----------------------------------------------------------------------------

# --- CUDA ---
if(WITH_CUDA)
  include("${V4R_SOURCE_DIR}/cmake/V4RDetectCUDA.cmake")
endif(WITH_CUDA)

# --- Eigen ---
if(WITH_EIGEN)
  find_path(EIGEN_INCLUDE_PATH "Eigen/Core"
            PATHS /usr/local /opt /usr ENV ProgramFiles ENV ProgramW6432
            PATH_SUFFIXES include/eigen3 include/eigen2 Eigen/include/eigen3 Eigen/include/eigen2
            DOC "The path to Eigen3/Eigen2 headers"
            CMAKE_FIND_ROOT_PATH_BOTH)

  if(EIGEN_INCLUDE_PATH)
    v4r_include_directories(${EIGEN_INCLUDE_PATH})
    v4r_parse_header("${EIGEN_INCLUDE_PATH}/Eigen/src/Core/util/Macros.h" EIGEN_VERSION_LINES EIGEN_WORLD_VERSION EIGEN_MAJOR_VERSION EIGEN_MINOR_VERSION)
    set(HAVE_EIGEN 1)
  endif()
endif(WITH_EIGEN)

# --- OpenMP ---
#if(WITH_OPENMP AND NOT HAVE_TBB AND NOT HAVE_CSTRIPES)
if(WITH_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
  set(HAVE_OPENMP "${OPENMP_FOUND}")
endif()

