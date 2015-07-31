if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_COMPILER_IS_GNUCXX 1)
  set(CMAKE_COMPILER_IS_CLANGCXX 1)
endif()
if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_COMPILER_IS_GNUCC 1)
  set(CMAKE_COMPILER_IS_CLANGCC 1)
endif()

# ----------------------------------------------------------------------------
# Detect Intel ICC compiler -- for -fPIC in 3rdparty ( UNIX ONLY ):
#  see  include/v4r/cxtypes.h file for related   ICC & CV_ICC defines.
# NOTE: The system needs to determine if the '-fPIC' option needs to be added
#  for the 3rdparty static libs being compiled.  The CMakeLists.txt files
#  in 3rdparty use the CV_ICC definition being set here to determine if
#  the -fPIC flag should be used.
# ----------------------------------------------------------------------------
if  (__ICL)
  set(CV_ICC   __ICL)
elseif(__ICC)
  set(CV_ICC   __ICC)
elseif(__ECL)
  set(CV_ICC   __ECL)
elseif(__ECC)
  set(CV_ICC   __ECC)
elseif(__INTEL_COMPILER)
  set(CV_ICC   __INTEL_COMPILER)
elseif(CMAKE_C_COMPILER MATCHES "icc")
  set(CV_ICC   icc_matches_c_compiler)
endif()

# ----------------------------------------------------------------------------
# Detect GNU version:
# ----------------------------------------------------------------------------
if(CMAKE_COMPILER_IS_CLANGCXX)
  set(CMAKE_GCC_REGEX_VERSION "4.2.1")
  set(CMAKE_V4R_GCC_VERSION_MAJOR 4)
  set(CMAKE_V4R_GCC_VERSION_MINOR 2)
  set(CMAKE_V4R_GCC_VERSION 42)
  set(CMAKE_V4R_GCC_VERSION_NUM 402)

  execute_process(COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} -v
                  ERROR_VARIABLE CMAKE_V4R_CLANG_VERSION_FULL
                  ERROR_STRIP_TRAILING_WHITESPACE)

  string(REGEX MATCH "version.*$" CMAKE_V4R_CLANG_VERSION_FULL "${CMAKE_V4R_CLANG_VERSION_FULL}")
  string(REGEX MATCH "[0-9]+\\.[0-9]+" CMAKE_CLANG_REGEX_VERSION "${CMAKE_V4R_CLANG_VERSION_FULL}")

elseif(CMAKE_COMPILER_IS_GNUCXX)
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} -dumpversion
                OUTPUT_VARIABLE CMAKE_V4R_GCC_VERSION_FULL
                OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} -v
                ERROR_VARIABLE CMAKE_V4R_GCC_INFO_FULL
                OUTPUT_STRIP_TRAILING_WHITESPACE)

  # Typical output in CMAKE_V4R_GCC_VERSION_FULL: "c+//0 (whatever) 4.2.3 (...)"
  # Look for the version number
  string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" CMAKE_GCC_REGEX_VERSION "${CMAKE_V4R_GCC_VERSION_FULL}")
  if(NOT CMAKE_GCC_REGEX_VERSION)
    string(REGEX MATCH "[0-9]+\\.[0-9]+" CMAKE_GCC_REGEX_VERSION "${CMAKE_V4R_GCC_VERSION_FULL}")
  endif()

  # Split the three parts:
  string(REGEX MATCHALL "[0-9]+" CMAKE_V4R_GCC_VERSIONS "${CMAKE_GCC_REGEX_VERSION}")

  list(GET CMAKE_V4R_GCC_VERSIONS 0 CMAKE_V4R_GCC_VERSION_MAJOR)
  list(GET CMAKE_V4R_GCC_VERSIONS 1 CMAKE_V4R_GCC_VERSION_MINOR)

  set(CMAKE_V4R_GCC_VERSION ${CMAKE_V4R_GCC_VERSION_MAJOR}${CMAKE_V4R_GCC_VERSION_MINOR})
  math(EXPR CMAKE_V4R_GCC_VERSION_NUM "${CMAKE_V4R_GCC_VERSION_MAJOR}*100 + ${CMAKE_V4R_GCC_VERSION_MINOR}")
  message(STATUS "Detected version of GNU GCC: ${CMAKE_V4R_GCC_VERSION} (${CMAKE_V4R_GCC_VERSION_NUM})")
endif()

# Similar code exists in V4RConfig.cmake
if(NOT DEFINED V4R_STATIC)
  # look for global setting
  if(NOT DEFINED BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS)
    set(V4R_STATIC OFF)
  else()
    set(V4R_STATIC ON)
  endif()
endif()
