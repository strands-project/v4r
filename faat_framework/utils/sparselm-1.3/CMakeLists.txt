# sparseLM CMake file; see http://www.cmake.org and 
#                        http://www.insightsoftwareconsortium.org/wiki/index.php/CMake_Tutorial

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(SPARSELM)

######################### START OF CONFIGURATION OPTIONS #########################

# sparse linear solvers available
# unavailable solvers are marked as internal, this should be changed if they become available later
SET(HAVE_MKL 0 CACHE BOOL "Do we have the MKL library?")
SET(HAVE_CHOLMOD 1 CACHE BOOL "Do we have the CHOLMOD library?")
SET(HAVE_CSPARSE 0 CACHE INTERNAL "Do we have the CSparse library?")
SET(HAVE_LDL 0 CACHE INTERNAL "Do we have the LDL library?")
SET(HAVE_UMFPACK 0 CACHE INTERNAL "Do we have the UMFPACK library?")
SET(HAVE_MA57 0 CACHE INTERNAL "Do we have the MA57 library?")
SET(HAVE_MA47 0 CACHE INTERNAL "Do we have the MA47 library?")
SET(HAVE_MA27 0 CACHE INTERNAL "Do we have the MA27 library?")
IF(HAVE_MKL)
  SET(HAVE_PARDISO 1 CACHE BOOL "PARDISO from MKL" FORCE)
ELSE(HAVE_MKL)
  SET(HAVE_PARDISO 0 CACHE INTERNAL "Do we have the PARDISO library?")
ENDIF(HAVE_MKL)
SET(HAVE_SuperLU 0 CACHE INTERNAL "Do we have the SuperLU library?")
SET(HAVE_TAUCS 0 CACHE INTERNAL "Do we have the TAUCS library?")
SET(HAVE_SPOOLES 0 CACHE INTERNAL "Do we have the SPOOLES library?")
SET(HAVE_MUMPS 0 CACHE INTERNAL "Do we have the MUMPS library?")
SET(HAVE_MA77 0 CACHE INTERNAL "Do we have the MA77 library?")

# set SuiteSparse root directory, allowing changes to be reflected in the GUI
IF(0)
IF(HAVE_CHOLMOD OR HAVE_CSPARSE OR HAVE_LDL OR HAVE_UMFPACK)
  IF(NOT SUITESPARSE_PATH)
    SET(SUITESPARSE_PATH "/usr/local/SuiteSparse-3.4.0" CACHE PATH "Path to the SuiteSparse package" FORCE)
  ENDIF(NOT SUITESPARSE_PATH)
ENDIF(HAVE_CHOLMOD OR HAVE_CSPARSE OR HAVE_LDL OR HAVE_UMFPACK)
ENDIF(0)
# until I find a way to set SUITESPARSE_PATH to some value in CMake's GUI
# and then have this value be propagated to all paths depending on it (i.e.,
# CHOLMOD_INCDIR, CHOLMOD_INCDIR, etc), it is defined to be empty.
SET(SUITESPARSE_PATH "")

include(${FAAT_PCL_SOURCE_DIR}/cmake/find_suitesparse.cmake)
#find_package (SUITESPARSE) 
#if (SUITESPARSE_FOUND )
#  MESSAGE(STATUS "SUITESPARSE FOUND")
#endif (SUITESPARSE_FOUND )


SET(USE_ATA_QUADS 1 CACHE BOOL "Use quads to speed up the calculation of A^t*A?")
OPTION(SPLM_BUILD_DEMO "Build demo program?" FALSE)

SET(NEED_UFconfig FALSE)
SET(NEED_AMD FALSE)
SET(NEED_METIS FALSE)

# header file paths
IF(HAVE_MKL)
  SET(MKL_INCDIR "C:/Program Files/Intel/MKL/10.2.2.025/include" CACHE PATH "Path to header files for the MKL library")
ENDIF(HAVE_MKL)

IF(HAVE_CHOLMOD)
  SET(CHOLMOD_INCDIR ${SUITESPARSE_INCLUDE_DIRS} CACHE PATH "Path to header files for the CHOLMOD library")
  SET(NEED_UFconfig TRUE)
ENDIF(HAVE_CHOLMOD)

IF(HAVE_MA57)
  SET(NEED_METIS TRUE)
ENDIF(HAVE_MA57)

IF(HAVE_MA77)
  SET(NEED_METIS TRUE)
ENDIF(HAVE_MA77)

IF(HAVE_CSPARSE)
  SET(CSPARSE_INCDIR ${SUITESPARSE_PATH}/CSparse/Include CACHE PATH "Path to header files for the CSparse library")
ENDIF(HAVE_CSPARSE)

IF(HAVE_LDL)
  SET(LDL_INCDIR ${SUITESPARSE_PATH}/LDL/Include CACHE PATH "Path to header files for the LDL library")
  SET(NEED_AMD TRUE)
  SET(NEED_UFconfig TRUE)
ENDIF(HAVE_LDL)

IF(HAVE_UMFPACK)
  SET(UMFPACK_INCDIR ${SUITESPARSE_PATH}/UMFPACK/Include CACHE PATH "Path to header files for the UMFPACK library")
  SET(NEED_UFconfig TRUE)
ENDIF(HAVE_UMFPACK)

IF(HAVE_UFconfig)
  SET(UFconfig_INCDIR ${SUITESPARSE_PATH}/UFconfig/ CACHE PATH "Path to header files for the UFconfig library")
ENDIF(HAVE_UFconfig)

IF(HAVE_PARDISO)
  IF(NOT HAVE_MKL)
    SET(PARDISO_INCDIR "" CACHE PATH "Path to header files for the PARDISO library")
  ENDIF(NOT HAVE_MKL)
ENDIF(HAVE_PARDISO)

IF(HAVE_SUPERLU)
  SET(SuperLU_INCDIR "/usr/local/include/slu" CACHE PATH "Path to header files for the SuperLU library")
ENDIF(HAVE_SUPERLU)

IF(HAVE_TAUCS)
  SET(TAUCS_INCDIR "" CACHE PATH "Path to header files for the TAUCS library") 
  SET(NEED_METIS TRUE)
ENDIF(HAVE_TAUCS)

IF(HAVE_SPOOLES)
  SET(SPOOLES_INCDIR "C:/spsolvers/spooles.2.2/" CACHE PATH "Path to header files for the SPOOLES library")
ENDIF(HAVE_SPOOLES)

IF(HAVE_MUMPS)
  SET(MUMPS_INCDIR "C:/spsolvers/MUMPS_4.8.3/include/ C:/spsolvers/MUMPS_4.8.3/libseq" CACHE STRING "Paths to header files for the MUMPS library")
ENDIF(HAVE_MUMPS)

# ordering options
IF(NEED_AMD)
  SET(AMD_INCDIR ${SUITESPARSE_PATH}/AMD/Include/ CACHE PATH "Path to header files for the AMD library")
ENDIF(NEED_AMD)

IF(NEED_METIS)
  SET(METIS_INCDIR "" CACHE PATH "Path to header files for the METIS library")
ENDIF(NEED_METIS)


IF(CMAKE_GENERATOR MATCHES "^Visual Studio *")
  # get rid of CRT warnings
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS)
ENDIF(CMAKE_GENERATOR MATCHES "^Visual Studio *")

########################## NO CHANGES BEYOND THIS POINT ##########################

# init variables
SET(INCLUDES "")
SET(SRCS splm.c splm_crsm.c splm_ccsm.c splm_hessian.c splm_time.c splm_stm.c splm_misc.c)

IF(NEED_AMD)
  LIST(APPEND INCLUDES ${AMD_INCDIR})
ENDIF(NEED_AMD)

IF(NEED_METIS)
  LIST(APPEND INCLUDES ${METIS_INCDIR})
ENDIF(NEED_METIS)

IF(NEED_UFconfig)
  LIST(APPEND INCLUDES ${UFconfig_INCDIR})
ENDIF(NEED_UFconfig)

# set sources and includes according to the available solvers
IF(HAVE_MKL)
  ADD_DEFINITIONS(-DHAVE_MKL)
  LIST(APPEND INCLUDES ${MKL_INCDIR})
  LIST(APPEND SRCS splm_dss.c)
ENDIF(HAVE_MKL)

IF(HAVE_CSPARSE)
  ADD_DEFINITIONS(-DHAVE_CSPARSE)
  LIST(APPEND INCLUDES ${CSPARSE_INCDIR})
  LIST(APPEND SRCS splm_csparse.c)
ENDIF(HAVE_CSPARSE)

IF(HAVE_LDL)
  ADD_DEFINITIONS(-DHAVE_LDL)
  LIST(APPEND INCLUDES ${LDL_INCDIR})
  LIST(APPEND SRCS splm_ldlp.c)
ENDIF(HAVE_LDL)

IF(HAVE_CHOLMOD)
  ADD_DEFINITIONS(-DHAVE_CHOLMOD)
  LIST(APPEND INCLUDES ${CHOLMOD_INCDIR})
  LIST(APPEND SRCS splm_cholmod.c)
ENDIF(HAVE_CHOLMOD)

IF(HAVE_MA57)
  ADD_DEFINITIONS(-DHAVE_MA57)
  LIST(APPEND SRCS splm_ma57.c ma57.c)
ENDIF(HAVE_MA57)

IF(HAVE_MA47)
  ADD_DEFINITIONS(-DHAVE_MA47)
  LIST(APPEND SRCS splm_ma47.c ma47.c)
ENDIF(HAVE_MA47)

IF(HAVE_MA27)
  ADD_DEFINITIONS(-DHAVE_MA27)
  LIST(APPEND SRCS splm_ma27.c ma27.c)
ENDIF(HAVE_MA27)

IF(HAVE_PARDISO)
  ADD_DEFINITIONS(-DHAVE_PARDISO)
  LIST(APPEND INCLUDES ${PARDISO_INCDIR})
  LIST(APPEND SRCS splm_pardiso.c)
ENDIF(HAVE_PARDISO)

IF(HAVE_SuperLU)
  ADD_DEFINITIONS(-DHAVE_SuperLU)
  LIST(APPEND INCLUDES ${SuperLU_INCDIR})
  LIST(APPEND SRCS splm_superlu.c)
ENDIF(HAVE_SuperLU)

IF(HAVE_UMFPACK)
  ADD_DEFINITIONS(-DHAVE_UMFPACK)
  LIST(APPEND INCLUDES ${UMFPACK_INCDIR})
  LIST(APPEND SRCS splm_umfpack.c)
ENDIF(HAVE_UMFPACK)

IF(HAVE_TAUCS)
  ADD_DEFINITIONS(-DHAVE_TAUCS)
  LIST(APPEND INCLUDES ${TAUCS_INCDIR})
  LIST(APPEND SRCS splm_taucs.c)
ENDIF(HAVE_TAUCS)

IF(HAVE_SPOOLES)
  ADD_DEFINITIONS(-DHAVE_SPOOLES)
  LIST(APPEND INCLUDES ${SPOOLES_INCDIR})
  LIST(APPEND SRCS splm_spooles.c)
ENDIF(HAVE_SPOOLES)

IF(HAVE_MUMPS)
  ADD_DEFINITIONS(-DHAVE_MUMPS)
  LIST(APPEND INCLUDES ${MUMPS_INCDIR})
  LIST(APPEND SRCS splm_mumps.c)
ENDIF(HAVE_MUMPS)

IF(HAVE_MA77)
  ADD_DEFINITIONS(-DHAVE_MA77)
  LIST(APPEND SRCS splm_ma77.c)
ENDIF(HAVE_MA77)


#CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/splm.h.in ${CMAKE_CURRENT_SOURCE_DIR}/splm.h)


INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}) # splm.h
INCLUDE_DIRECTORIES(${INCLUDES})
MESSAGE(STATUS "splm library include paths: ${INCLUDES}")

# splm library
ADD_LIBRARY(splm SHARED ${SRCS} compiler.h splm.h splm_priv.h)
target_link_libraries(splm cholmod)

# demo program
#IF(BUILD_DEMO)
#  INCLUDE("CMakeLists_demo.txt")
#ENDIF(BUILD_DEMO)

#ADD_SUBDIRECTORY(matlab)
