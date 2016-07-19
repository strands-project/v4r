# ----------------------------------------------------------------------------
#  Root CMake file for V4R
#
#    From the off-tree build directory, invoke:
#      $ cmake <PATH_TO_V4R_ROOT>
#
# ----------------------------------------------------------------------------



include(cmake/V4RMinDepVersions.cmake)

cmake_minimum_required(VERSION "${MIN_VER_CMAKE}" FATAL_ERROR)

# it _must_ go before project(V4R) in order to work
set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Installation Directory")

if(POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif()

if(POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif()

if(POLICY CMP0022)
  cmake_policy(SET CMP0022 OLD)
endif()

if(POLICY CMP0026)
  # silence cmake 3.0+ warnings about reading LOCATION attribute
  cmake_policy(SET CMP0026 OLD)
endif()

# must go before the project command
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
if(DEFINED CMAKE_BUILD_TYPE)
  set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES} )
endif()

project(V4R CXX C)

include(cmake/V4RUtils.cmake)

v4r_clear_vars(V4RModules_TARGETS)

# ----------------------------------------------------------------------------
# Break in case of popular CMake configuration mistakes
# ----------------------------------------------------------------------------
if(NOT CMAKE_SIZEOF_VOID_P GREATER 0)
  message(FATAL_ERROR "CMake fails to deterimine the bitness of target platform.
  Please check your CMake and compiler installation. If you are crosscompiling then ensure that your CMake toolchain file correctly sets the compiler details.")
endif()

# ----------------------------------------------------------------------------
# Detect compiler and target platform architecture
# ----------------------------------------------------------------------------
include(cmake/V4RDetectCXXCompiler.cmake)

# Add these standard paths to the search paths for FIND_LIBRARY
# to find libraries from these locations first
if(EXISTS /lib64)
  list(APPEND CMAKE_LIBRARY_PATH /lib64)
else()
  list(APPEND CMAKE_LIBRARY_PATH /lib)
endif()

if(EXISTS /usr/lib64)
  list(APPEND CMAKE_LIBRARY_PATH /usr/lib64)
else()
  list(APPEND CMAKE_LIBRARY_PATH /usr/lib)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# ----------------------------------------------------------------------------
# V4R cmake options
# ----------------------------------------------------------------------------

# Optional 3rd party components
# ===================================================
V4R_OPTION(WITH_VTK            "Include VTK library support (and build v4r_viz module eiher)"             ON  IF (NOT ANDROID AND NOT IOS AND NOT WINRT) )
V4R_OPTION(WITH_ASSIMP         "Include Assimp (Open Asset Import library) support"                       ON)
V4R_OPTION(WITH_CERES          "Include Ceres Support"                                         ON)
V4R_OPTION(WITH_CUDA           "Include NVidia Cuda Runtime support"                                         ON  IF (NOT IOS AND NOT WINRT) )
V4R_OPTION(WITH_CUFFT          "Include NVidia Cuda Fast Fourier Transform (FFT) library support"            ON  IF (NOT IOS AND NOT WINRT) )
V4R_OPTION(WITH_CUBLAS         "Include NVidia Cuda Basic Linear Algebra Subprograms (BLAS) library support" OFF IF (NOT IOS AND NOT WINRT) )
V4R_OPTION(WITH_OPENGL         "Include OPENGL support"                         ON)
V4R_OPTION(WITH_NVCUVID        "Include NVidia Video Decoding library support"                               OFF IF (NOT IOS AND NOT APPLE) )
V4R_OPTION(WITH_OPENGL         "Include OpenGL support"                      OFF  IF (NOT ANDROID AND NOT WINRT) )
V4R_OPTION(WITH_OPENNI         "Include OpenNI support"                      OFF  IF (NOT ANDROID AND NOT IOS AND NOT WINRT) )
V4R_OPTION(WITH_OPENNI2        "Include OpenNI2 support"                     OFF  IF (NOT ANDROID AND NOT IOS AND NOT WINRT) )
V4R_OPTION(WITH_PNG            "Include PNG support"                         ON)
V4R_OPTION(WITH_QT             "Build with Qt Backend support"               ON  IF (NOT ANDROID AND NOT IOS AND NOT WINRT) )
V4R_OPTION(WITH_OPENMP         "Include OpenMP support"                      ON)
V4R_OPTION(WITH_PTHREADS_PF    "Use pthreads-based parallel_for"             ON   IF (NOT WIN32) )
V4R_OPTION(WITH_EDT            "Include EDT support"                         ON   IF (NOT IOS) )
V4R_OPTION(WITH_OPENCL         "Include OpenCL Runtime support"              NOT ANDROID IF (NOT IOS AND NOT WINRT) )
V4R_OPTION(WITH_OPENCL_SVM     "Include OpenCL Shared Virtual Memory support" OFF ) # experimental
V4R_OPTION(WITH_OPENCLAMDFFT   "Include AMD OpenCL FFT library support"      ON   IF (NOT ANDROID AND NOT IOS AND NOT WINRT) )
V4R_OPTION(WITH_OPENCLAMDBLAS  "Include AMD OpenCL BLAS library support"     ON   IF (NOT ANDROID AND NOT IOS AND NOT WINRT) )
V4R_OPTION(WITH_OPENCV         "Include OpenCV support"                      ON)
V4R_OPTION(WITH_OPENNI2        "Include OpenNI support"                      ON)
V4R_OPTION(WITH_SIFTGPU        "Include SiftGPU support"                     ON)
V4R_OPTION(WITH_METSLIB        "Include METSlib support"                     ON)
V4R_OPTION(WITH_LIBSVM         "Include LIBSVM support"                      ON)
V4R_OPTION(WITH_PCL            "Include PCL support"                         ON)
V4R_OPTION(WITH_QT4            "Include QT support"                          ON)
V4R_OPTION(WITH_BOOST          "Include Boost support"                       ON)
V4R_OPTION(WITH_EIGEN          "Include Eigen3 support"                      ON)
V4R_OPTION(WITH_GLM            "Include GLM support"                         ON)
V4R_OPTION(WITH_GLEW            "Include GLEW support"                       ON)
V4R_OPTION(WITH_GLOG           "Include Google Logging support"              ON)
V4R_OPTION(WITH_CAFFE           "Include Caffe support"              ON)

# V4R build components
# ===================================================
V4R_OPTION(BUILD_SHARED_LIBS        "Build shared libraries (.so) instead of static ones (.a)" NOT (ANDROID OR IOS) )
V4R_OPTION(BUILD_v4r_apps           "Build utility applications (used for example to train classifiers)" (NOT ANDROID AND NOT WINRT) IF (NOT IOS) )
V4R_OPTION(BUILD_DOCS               "Create build rules for V4R Documentation"    ON  IF NOT WINRT)
V4R_OPTION(BUILD_EXAMPLES           "Build all examples"                          ON )
V4R_OPTION(BUILD_EVALUATION_TOOLS   "Build all evaluation tools"                          OFF )
V4R_OPTION(BUILD_UTILITY_TOOLS      "Build all utility tools"                          ON )
#V4R_OPTION(BUILD_ICRA16_TOOLS       "Build evaluations for RA-L Paper 2016, Faeulhammer"    OFF )
V4R_OPTION(BUILD_PACKAGE            "Enables 'make package_source' command"       ON  IF NOT WINRT)
V4R_OPTION(BUILD_PERF_TESTS         "Build performance tests"                     ON  IF (NOT IOS) )
V4R_OPTION(BUILD_TESTS              "Build accuracy & regression tests"           ON  IF (NOT IOS) )
V4R_OPTION(BUILD_WITH_DEBUG_INFO    "Include debug info into debug libs (not MSCV only)" ON )
V4R_OPTION(BUILD_WITH_STATIC_CRT    "Enables use of staticaly linked CRT for staticaly linked V4R" ON IF MSVC )
V4R_OPTION(BUILD_FAT_JAVA_LIB       "Create fat java wrapper containing the whole V4R library" ON IF NOT BUILD_SHARED_LIBS AND CMAKE_COMPILER_IS_GNUCXX )
V4R_OPTION(BUILD_ANDROID_SERVICE    "Build V4R Manager for Google Play" OFF IF ANDROID )
V4R_OPTION(BUILD_CUDA_STUBS         "Build CUDA modules stubs when no CUDA SDK" OFF  IF (NOT IOS) )

# 3rd party libs
V4R_OPTION(BUILD_ZLIB               "Build zlib from source"             WIN32 OR APPLE )
V4R_OPTION(BUILD_EDT                "Build libedt from source"           ON)
V4R_OPTION(BUILD_JASPER             "Build libjasper from source"        WIN32 OR ANDROID OR APPLE )
V4R_OPTION(BUILD_JPEG               "Build libjpeg from source"          WIN32 OR ANDROID OR APPLE )
V4R_OPTION(BUILD_PNG                "Build libpng from source"           WIN32 OR ANDROID OR APPLE )
V4R_OPTION(BUILD_OPENEXR            "Build openexr from source"          (WIN32 OR ANDROID OR APPLE) AND NOT WINRT)
V4R_OPTION(BUILD_TBB                "Download and build TBB from source" ANDROID )
V4R_OPTION(BUILD_SIFTGPU            "Download and build SiftGPU from source" ON)
V4R_OPTION(BUILD_METSLIB            "Download and build METSlib from source" ON)
V4R_OPTION(BUILD_LIBSVM             "Download and build LIBSVM from source" OFF)

# V4R installation options
# ===================================================
V4R_OPTION(INSTALL_CREATE_DISTRIB   "Change install rules to build the distribution package" OFF )
V4R_OPTION(INSTALL_C_EXAMPLES       "Install C examples"        ON )
V4R_OPTION(INSTALL_C_UTILITY_TOOLS        "Install C utility tools"        ON )
V4R_OPTION(INSTALL_C_EVALUATION_TOOLS       "Install C evaluation tools"        OFF )
V4R_OPTION(INSTALL_TO_MANGLED_PATHS "Enables mangled install paths, that help with side by side installs." OFF IF (UNIX AND NOT ANDROID AND NOT IOS AND BUILD_SHARED_LIBS) )
V4R_OPTION(INSTALL_TESTS            "Install accuracy and performance test binaries and test data" OFF)

# V4R build options
# ===================================================
V4R_OPTION(ENABLE_PROFILING           "Enable profiling in the GCC compiler (Add flags: -g -pg)" OFF)
V4R_OPTION(ENABLE_COVERAGE            "Enable coverage collection with  GCov"                    OFF)
V4R_OPTION(ENABLE_OMIT_FRAME_POINTER  "Enable -fomit-frame-pointer for GCC"                      ON   IF (CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_COMPILER_IS_CLANGCXX) )
V4R_OPTION(ENABLE_POWERPC             "Enable PowerPC for GCC"                                   ON   IF (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_SYSTEM_PROCESSOR MATCHES powerpc.*) )
V4R_OPTION(ENABLE_FAST_MATH           "Enable -ffast-math (not recommended for GCC 4.6.x)"       OFF)
V4R_OPTION(ENABLE_SSE                 "Enable SSE instructions"                                  ON)
V4R_OPTION(ENABLE_SSE2                "Enable SSE2 instructions"                                 ON)
V4R_OPTION(ENABLE_SSE3                "Enable SSE3 instructions"                                 ON)
V4R_OPTION(ENABLE_SSSE3               "Enable SSSE3 instructions"                                ON)
V4R_OPTION(ENABLE_SSE41               "Enable SSE4.1 instructions"                               ON)
V4R_OPTION(ENABLE_SSE42               "Enable SSE4.2 instructions"                               ON)
V4R_OPTION(ENABLE_POPCNT              "Enable POPCNT instructions"                               OFF)
V4R_OPTION(ENABLE_AVX                 "Enable AVX instructions"                                  OFF)
V4R_OPTION(ENABLE_AVX2                "Enable AVX2 instructions"                                 OFF)
V4R_OPTION(ENABLE_FMA3                "Enable FMA3 instructions"                                 OFF)
V4R_OPTION(ENABLE_NOISY_WARNINGS      "Show all warnings even if they are too noisy"             OFF)
V4R_OPTION(V4R_WARNINGS_ARE_ERRORS    "Treat warnings as errors"                                 OFF)
V4R_OPTION(ENABLE_IMPL_COLLECTION     "Collect implementation data on function call"             OFF)
V4R_OPTION(GENERATE_ABI_DESCRIPTOR    "Generate XML file for abi_compliance_checker tool"        OFF IF UNIX)

if(ENABLE_IMPL_COLLECTION)
  add_definitions(-DCV_COLLECT_IMPL_DATA)
endif()


# ----------------------------------------------------------------------------
#  Get actual V4R version number from sources
# ----------------------------------------------------------------------------
include(cmake/V4RVersion.cmake)


# ----------------------------------------------------------------------------
#  Build & install layouts
# ----------------------------------------------------------------------------

# Save libs and executables in the same place
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Output directory for applications" )

set(V4R_DOC_INSTALL_PATH share/V4R/doc)
set(V4R_SAMPLES_BIN_INSTALL_PATH "samples")
set(V4R_BIN_INSTALL_PATH "bin")

if(NOT V4R_TEST_INSTALL_PATH)
  set(V4R_TEST_INSTALL_PATH "${V4R_BIN_INSTALL_PATH}")
endif()

if (V4R_TEST_DATA_PATH)
  get_filename_component(V4R_TEST_DATA_PATH ${V4R_TEST_DATA_PATH} ABSOLUTE)
endif()

if(V4R_TEST_DATA_PATH AND NOT V4R_TEST_DATA_INSTALL_PATH)
  set(V4R_TEST_DATA_INSTALL_PATH "share/V4R/testdata")
endif()

set(LIBRARY_OUTPUT_PATH         "${V4R_BINARY_DIR}/lib")
set(3P_LIBRARY_OUTPUT_PATH      "${V4R_BINARY_DIR}/3rdparty/lib${LIB_SUFFIX}")

set(V4R_LIB_INSTALL_PATH         lib${LIB_SUFFIX})
set(V4R_3P_LIB_INSTALL_PATH      share/V4R/3rdparty/${V4R_LIB_INSTALL_PATH})
set(V4R_SAMPLES_SRC_INSTALL_PATH share/V4R/samples)
set(V4R_OTHER_INSTALL_PATH       share/V4R)
set(V4R_INCLUDE_INSTALL_PATH     "include")
set(V4R_3P_INCLUDE_INSTALL_PATH  share/V4R/3rdparty/${V4R_INCLUDE_INSTALL_PATH})

math(EXPR SIZEOF_VOID_P_BITS "8 * ${CMAKE_SIZEOF_VOID_P}")
if(LIB_SUFFIX AND NOT SIZEOF_VOID_P_BITS EQUAL LIB_SUFFIX)
  set(V4R_CONFIG_INSTALL_PATH lib${LIB_SUFFIX}/cmake/v4r)
else()
  set(V4R_CONFIG_INSTALL_PATH share/V4R)
endif()

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${V4R_LIB_INSTALL_PATH}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(INSTALL_TO_MANGLED_PATHS)
  set(V4R_INCLUDE_INSTALL_PATH ${V4R_INCLUDE_INSTALL_PATH}/v4r-${V4R_VERSION})
  string(REPLACE "V4R" "V4R-${V4R_VERSION}" V4R_3P_LIB_INSTALL_PATH "${V4R_3P_LIB_INSTALL_PATH}")
  string(REPLACE "V4R" "V4R-${V4R_VERSION}" V4R_SAMPLES_SRC_INSTALL_PATH "${V4R_SAMPLES_SRC_INSTALL_PATH}")
  string(REPLACE "V4R" "V4R-${V4R_VERSION}" V4R_CONFIG_INSTALL_PATH "${V4R_CONFIG_INSTALL_PATH}")
  string(REPLACE "V4R" "V4R-${V4R_VERSION}" V4R_DOC_INSTALL_PATH "${V4R_DOC_INSTALL_PATH}")
  string(REPLACE "V4R" "V4R-${V4R_VERSION}" V4R_JAR_INSTALL_PATH "${V4R_JAR_INSTALL_PATH}")
  string(REPLACE "V4R" "V4R-${V4R_VERSION}" V4R_TEST_DATA_INSTALL_PATH "${V4R_TEST_DATA_INSTALL_PATH}")
  string(REPLACE "V4R" "V4R-${V4R_VERSION}" V4R_OTHER_INSTALL_PATH "${V4R_OTHER_INSTALL_PATH}")
endif()

# Postfix of so's:
set(V4R_DLLVERSION "")
set(V4R_DEBUG_POSTFIX "")

if(DEFINED CMAKE_DEBUG_POSTFIX)
  set(V4R_DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}")
endif()

# ----------------------------------------------------------------------------
#  Path for build/platform -specific headers
# ----------------------------------------------------------------------------
set(V4R_CONFIG_FILE_INCLUDE_DIR "${CMAKE_BINARY_DIR}/" CACHE PATH "Where to create the platform-dependant v4r_config.h")
v4r_include_directories(${V4R_CONFIG_FILE_INCLUDE_DIR})

# ----------------------------------------------------------------------------
#  Path for additional modules
# ----------------------------------------------------------------------------
set(V4R_EXTRA_MODULES_PATH "" CACHE PATH "Where to look for additional V4R modules")

# ----------------------------------------------------------------------------
#  Autodetect if we are in a GIT repository
# ----------------------------------------------------------------------------
find_host_package(Git QUIET)

if(GIT_FOUND)
  execute_process(COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty --match "[0-9].[0-9].[0-9]*"
    WORKING_DIRECTORY "${V4R_SOURCE_DIR}"
    OUTPUT_VARIABLE V4R_VCSVERSION
    RESULT_VARIABLE GIT_RESULT
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT GIT_RESULT EQUAL 0)
    set(V4R_VCSVERSION "unknown")
  endif()
else()
  # We don't have git:
  set(V4R_VCSVERSION "unknown")
endif()


# ----------------------------------------------------------------------------
# V4R compiler and linker options
# ----------------------------------------------------------------------------
# In case of Makefiles if the user does not setup CMAKE_BUILD_TYPE, assume it's Release:
if(CMAKE_GENERATOR MATCHES "Makefiles|Ninja" AND "${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

include(cmake/V4RCompilerOptions.cmake)


# ----------------------------------------------------------------------------
#       CHECK FOR SYSTEM LIBRARIES, OPTIONS, ETC..
# ----------------------------------------------------------------------------
find_package(PkgConfig QUIET)
include(CheckFunctionExists)
include(CheckIncludeFile)

CHECK_INCLUDE_FILE(pthread.h HAVE_LIBPTHREAD)
set(V4R_LINKER_LIBS ${V4R_LINKER_LIBS} X11 Xrandr Xinerama Xcursor)
set(V4R_LINKER_LIBS ${V4R_LINKER_LIBS} dl m pthread rt)

include(cmake/V4RModule.cmake)

# ----------------------------------------------------------------------------
#  Detect endianness of build platform
# ----------------------------------------------------------------------------

include(TestBigEndian)
test_big_endian(WORDS_BIGENDIAN)

# ----------------------------------------------------------------------------
#  Detect 3rd-party libraries
# ----------------------------------------------------------------------------

#include(cmake/V4RFindLibsGrfmt.cmake)
#include(cmake/V4RFindLibsGUI.cmake)
#include(cmake/V4RFindLibsVideo.cmake)
include(cmake/V4RFindAssimp.cmake)
include(cmake/V4RFindCeres.cmake)
include(cmake/V4RFindLibsPerf.cmake)
include(cmake/V4RFindEigen.cmake)
include(cmake/V4RFindGlew.cmake)
include(cmake/V4RFindGLM.cmake)
include(cmake/V4RFindGlog.cmake)
include(cmake/V4RFindOpenGL.cmake)
include(cmake/V4RFindPCL.cmake)
include(cmake/V4RFindQt.cmake)
include(cmake/V4RFindOpenCV.cmake)
include(cmake/V4RFindOpenNI2.cmake)
include(cmake/V4RFindBoost.cmake)
include(cmake/V4RFindCaffe.cmake)
# ----------------------------------------------------------------------------
#  Detect optional 3rd-party libraries/tools
# ----------------------------------------------------------------------------

include(cmake/V4RFindEDT.cmake)
include(cmake/V4RFindSiftGPU.cmake)
include(cmake/V4RFindMETSlib.cmake)
include(cmake/V4RFindLIBSVM.cmake)

# --- Doxygen for documentation ---
unset(DOXYGEN_FOUND CACHE)
if(BUILD_DOCS)
  find_package(Doxygen)
endif(BUILD_DOCS)

# TODO
# --- OpenCL ---
#if(WITH_OPENCL)
  #include(cmake/V4RDetectOpenCL.cmake)
#endif()

# TODO
#if(WITH_VTK)
  #include(cmake/V4RDetectVTK.cmake)
#endif()

# ----------------------------------------------------------------------------
# Add CUDA libraries (needed for apps/tools, samples)
# ----------------------------------------------------------------------------
if(HAVE_CUDA)
  set(V4R_LINKER_LIBS ${V4R_LINKER_LIBS} ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY})
  if(HAVE_CUBLAS)
    set(V4R_LINKER_LIBS ${V4R_LINKER_LIBS} ${CUDA_cublas_LIBRARY})
  endif()
  if(HAVE_CUFFT)
    set(V4R_LINKER_LIBS ${V4R_LINKER_LIBS} ${CUDA_cufft_LIBRARY})
  endif()
endif()

# Extra V4R targets: uninstall, package_source, perf, etc.
include(cmake/V4RExtraTargets.cmake)


# ----------------------------------------------------------------------------
# Process subdirectories
# ----------------------------------------------------------------------------

# TODO
# v4r.hpp and legacy headers
#add_subdirectory(include)

# V4R modules
add_subdirectory(modules)

# TODO
# Generate targets for documentation
#add_subdirectory(doc)

# TODO
# various data that is used by cv libraries and/or demo applications.
#add_subdirectory(data)

# extra applications
if(BUILD_v4r_apps)
  # TODO
  add_subdirectory(apps)
endif()

# examples
if(BUILD_EXAMPLES)
  add_subdirectory(samples)
endif()

# Figure out the list of dependencies
v4r_get_all_libs(deps_modules deps_extra deps_3rdparty)

# ----------------------------------------------------------------------------
# Finalization: generate configuration-based files
# ----------------------------------------------------------------------------

# Generate platform-dependent and configuration-dependent headers
include(cmake/V4RGenHeaders.cmake)

# TODO
# Generate v4r.pc for pkg-config command
#include(cmake/V4RGenPkgconfig.cmake)

# Generate V4RСonfig.cmake and V4RConfig-version.cmake for cmake projects
include(cmake/V4RGenConfig.cmake)

# TODO
# Generate ABI descriptor
#include(cmake/V4RGenABI.cmake)

# Generate environment setup file
if(INSTALL_TESTS AND V4R_TEST_DATA_PATH)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/v4r_run_all_tests_unix.sh.in"
                 "${CMAKE_BINARY_DIR}/unix-install/v4r_run_all_tests.sh" @ONLY)
  install(PROGRAMS "${CMAKE_BINARY_DIR}/unix-install/v4r_run_all_tests.sh"
          DESTINATION ${V4R_TEST_INSTALL_PATH} COMPONENT tests)
endif()

if(NOT V4R_LICENSE_FILE)
  set(V4R_LICENSE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE)
endif()

# ----------------------------------------------------------------------------
# Summary:
# ----------------------------------------------------------------------------
status("")
status("General configuration for V4R ${V4R_VERSION} =====================================")
if(V4R_VCSVERSION)
  status("  Version control:" ${V4R_VCSVERSION})
endif()

# ========================== build platform ==========================
status("")
status("  Platform:")
status("    Host:"             ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_VERSION} ${CMAKE_HOST_SYSTEM_PROCESSOR})
if(CMAKE_CROSSCOMPILING)
  status("    Target:"         ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
endif()
status("    CMake:"            ${CMAKE_VERSION})
status("    CMake generator:"  ${CMAKE_GENERATOR})
status("    CMake build tool:" ${CMAKE_BUILD_TOOL})
status("    Configuration:"    ${CMAKE_BUILD_TYPE})

# ========================== C/C++ options ==========================
if(CMAKE_CXX_COMPILER_VERSION)
  set(V4R_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} (ver ${CMAKE_CXX_COMPILER_VERSION})")
elseif(CMAKE_COMPILER_IS_CLANGCXX)
  set(V4R_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} (ver ${CMAKE_CLANG_REGEX_VERSION})")
elseif(CMAKE_COMPILER_IS_GNUCXX)
  set(V4R_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} (ver ${CMAKE_GCC_REGEX_VERSION})")
else()
  set(V4R_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1}")
endif()
string(STRIP "${V4R_COMPILER_STR}" V4R_COMPILER_STR)

status("")
status("  C/C++:")
status("    Built as dynamic libs:"  BUILD_SHARED_LIBS THEN YES ELSE NO)
status("    C++ Compiler:"           ${V4R_COMPILER_STR})
status("    C++ flags (Release):"    ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE})
status("    C++ flags (Debug):"      ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG})
status("    C Compiler:"             ${CMAKE_C_COMPILER} ${CMAKE_C_COMPILER_ARG1})
status("    C flags (Release):"      ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE})
status("    C flags (Debug):"        ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG})
status("    Linker flags (Release):" ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_RELEASE})
status("    Linker flags (Debug):"   ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_DEBUG})

# ========================== Dependencies ============================
status("    Extra dependencies:" ${deps_extra})
status("    3rdparty dependencies:" ${deps_3rdparty})

# ========================== V4R modules ==========================
status("")
status("  V4R modules:")
string(REPLACE "v4r_" "" V4R_MODULES_BUILD_ST          "${V4R_MODULES_BUILD}")
string(REPLACE "v4r_" "" V4R_MODULES_DISABLED_USER_ST  "${V4R_MODULES_DISABLED_USER}")
string(REPLACE "v4r_" "" V4R_MODULES_DISABLED_FORCE_ST "${V4R_MODULES_DISABLED_FORCE}")
set(V4R_MODULES_DISABLED_AUTO_ST "")
foreach(m ${V4R_MODULES_DISABLED_AUTO})
  set(__mdeps "")
  foreach(d ${V4R_MODULE_${m}_DEPS})
    if(d MATCHES "^v4r_" AND NOT HAVE_${d})
      list(APPEND __mdeps ${d})
    endif()
  endforeach()
  if(__mdeps)
    list(APPEND V4R_MODULES_DISABLED_AUTO_ST "${m}(deps: ${__mdeps})")
  else()
    list(APPEND V4R_MODULES_DISABLED_AUTO_ST "${m}")
  endif()
endforeach()
string(REPLACE "v4r_" "" V4R_MODULES_DISABLED_AUTO_ST  "${V4R_MODULES_DISABLED_AUTO_ST}")

status("    To be built:"            V4R_MODULES_BUILD          THEN ${V4R_MODULES_BUILD_ST}          ELSE "-")
status("    Disabled:"               V4R_MODULES_DISABLED_USER  THEN ${V4R_MODULES_DISABLED_USER_ST}  ELSE "-")
status("    Disabled by dependency:" V4R_MODULES_DISABLED_AUTO  THEN ${V4R_MODULES_DISABLED_AUTO_ST}  ELSE "-")
status("    Unavailable:"            V4R_MODULES_DISABLED_FORCE THEN ${V4R_MODULES_DISABLED_FORCE_ST} ELSE "-")

# ========================== GUI ==========================
status("")
status("  GUI: ")

if(HAVE_QT5)
  status("    QT 5.x:"            HAVE_QT        THEN "YES (ver ${Qt5Core_VERSION})" ELSE NO)
  status("    QT OpenGL support:" HAVE_QT_OPENGL THEN "YES (${Qt5OpenGL_LIBRARIES} ${Qt5OpenGL_VERSION})" ELSE NO)
elseif(HAVE_QT)
  status("    QT 4.x:"            HAVE_QT        THEN "YES (ver ${QT_VERSION_MAJOR}.${QT_VERSION_MINOR}.${QT_VERSION_PATCH} ${QT_EDITION})" ELSE NO)
  status("    QT OpenGL support:" HAVE_QT_OPENGL THEN "YES (${QT_QTOPENGL_LIBRARY})" ELSE NO)
else()
  if(DEFINED WITH_QT)
    status("    QT:" NO)
  endif()
  status("    GThread :" HAVE_GTHREAD THEN "YES (ver ${ALIASOF_gthread-2.0_VERSION})" ELSE NO)
  status("    GtkGlExt:" HAVE_GTKGLEXT THEN "YES (ver ${ALIASOF_gtkglext-1.0_VERSION})" ELSE NO)
endif()

status("    OpenGL support:" HAVE_OPENGL THEN "YES (${OPENGL_LIBRARIES})" ELSE NO)
status("    VTK support:" HAVE_VTK THEN "YES (ver ${VTK_VERSION})" ELSE NO)

# ========================== MEDIA IO ==========================
status("")
status("  Media I/O: ")
status("    ZLib:"         BUILD_ZLIB    THEN "build (ver ${ZLIB_VERSION})"               ELSE "${ZLIB_LIBRARIES} (ver ${ZLIB_VERSION})")

if(WITH_JPEG)
  status("    JPEG:"       JPEG_FOUND    THEN "${JPEG_LIBRARY} (ver ${JPEG_LIB_VERSION})"        ELSE "build (ver ${JPEG_LIB_VERSION})")
else()
  status("    JPEG:"       "NO")
endif()

if(WITH_WEBP)
  status("    WEBP:"       WEBP_FOUND    THEN "${WEBP_LIBRARY} (ver ${WEBP_VERSION})"        ELSE "build (ver ${WEBP_VERSION})")
else()
  status("    WEBP:"       "NO")
endif()

if(WITH_PNG)
  status("    PNG:"        PNG_FOUND     THEN "${PNG_LIBRARY} (ver ${PNG_VERSION})"              ELSE "build (ver ${PNG_VERSION})")
else()
  status("    PNG:"        "NO")
endif()
if(WITH_TIFF)
  if(TIFF_VERSION_STRING AND TIFF_FOUND)
    status("    TIFF:"     "${TIFF_LIBRARY} (ver ${TIFF_VERSION} - ${TIFF_VERSION})")
  else()
    status("    TIFF:"     TIFF_FOUND    THEN "${TIFF_LIBRARY} (ver ${TIFF_VERSION})"            ELSE "build (ver ${TIFF_VERSION} - ${TIFF_VERSION_STRING})")
  endif()
else()
  status("    TIFF:"       "NO")
endif()
if(WITH_JASPER)
  status("    JPEG 2000:"  JASPER_FOUND  THEN "${JASPER_LIBRARY} (ver ${JASPER_VERSION_STRING})" ELSE "build (ver ${JASPER_VERSION_STRING})")
else()
  status("    JPEG 2000:"  "NO")
endif()
if(WITH_OPENEXR)
  status("    OpenEXR:"  OPENEXR_FOUND  THEN "${OPENEXR_LIBRARIES} (ver ${OPENEXR_VERSION})" ELSE "build (ver ${OPENEXR_VERSION})")
else()
  status("    OpenEXR:"  "NO")
endif()
if(WITH_EDT)
  status("    EDT:"  EDT_FOUND  THEN "${EDT_LIBRARIES} (ver ${EDT_VERSION})" ELSE "build (ver ${EDT_VERSION})")
else()
  status("    EDT:"  "NO")
endif()
if(WITH_SIFTGPU)
  status("    SiftGPU:"  SIFTGPU_FOUND  THEN "${SIFTGPU_LIBRARIES} (ver ${SIFTGPU_VERSION})" ELSE "build (ver ${SIFTGPU_VERSION})")
else()
  status("    SiftGPU:"  "NO")
endif()
if(WITH_METSLIB)
  status("    METSlib:"  METSLIB_FOUND  THEN "${METSLIB_LIBRARIES} (ver ${METSLIB_VERSION})" ELSE "build (ver ${METSLIB_VERSION})")
else()
  status("    METSlib:"  "NO")
endif()
if(WITH_LIBSVM)
  status("    LIBSVM:"  LIBSVM_FOUND  THEN "${LIBSVM_LIBRARIES} (ver ${LIBSVM_VERSION})" ELSE "build (ver ${LIBSVM_VERSION})")
else()
  status("    LIBSVM:"  "NO")
endif()
if(WITH_PCL AND HAVE_PCL)
  status("    PCL:"  "YES (ver ${PCL_VERSION})")
else()
  status("    PCL:"  "NO")
endif()
if(WITH_QT)
    if (HAVE_QT4)
      status("    QT:"  "YES (ver toto ${Qt4_VERSION})")
    elseif(HAVE_QT5)
        status("    QT:"  "YES (ver ${Qt5_VERSION})")
    else()
      status("    QT:"  "NO")
    endif()
else()
  status("    QT:"  "NO")
endif()
if(WITH_OPENCV AND HAVE_OPENCV)
  status("    OpenCV:"  "YES (ver ${OpenCV_VERSION})")
else()
  status("    OpenCV:"  "NO")
endif()
if(WITH_OPENNI2 AND HAVE_OPENNI2)
    status("    OpenNI2:"  "YES (ver ${OPENNI2_VERSION})")
else()
  status("    OpenNI2:"  "NO")
endif()
if(WITH_CERES AND HAVE_CERES)
  status("    Ceres:"  "YES (ver ${CERES_VERSION})")
else()
  status("    Ceres:"  "NO")
endif()

if(WITH_OPENGL AND HAVE_OPENGL)
  status("    OPENGL:"  "YES")
else()
  status("    OPENGL:"  "NO")
endif()

if(WITH_ASSIMP AND HAVE_ASSIMP)
  status("    Assimp:"  "${ASSIMP_LIBRARIES} (ver ${ASSIMP_VERSION})")
else()
  status("    Assimp:"  "NO")
endif()

if( WITH_GDAL )
  status("    GDAL:"   GDAL_FOUND THEN "${GDAL_LIBRARY}" ELSE "NO")
else()
  status("    GDAL:"     "NO")
endif()

if(WITH_GLOG AND HAVE_GLOG)
  status("    GLOG:"  "${GLOG_LIBRARIES}")
else()
  status("    GLOG:"  "NO")
endif()


if(WITH_CAFFE AND HAVE_CAFFE)
  status("    CAFFE:"  "${CAFFE_LIBRARIES}")
else()
  status("    CAFFE:"  "NO")
endif()

# ========================== VIDEO IO ==========================
status("")
status("  Video I/O:")

if(WITH_OPENNI)
  status("    OpenNI:"         HAVE_OPENNI         THEN "YES (ver ${OPENNI_VERSION_STRING}, build ${OPENNI_VERSION_BUILD})"
                                                                                                   ELSE NO)
  status("    OpenNI PrimeSensor Modules:" HAVE_OPENNI_PRIME_SENSOR_MODULE
                                                   THEN "YES (${OPENNI_PRIME_SENSOR_MODULE})"      ELSE NO)
else()
  status("    OpenNI:"  "NO")
endif()

if(WITH_OPENNI2)
  status("    OpenNI2:"		   HAVE_OPENNI2	   THEN "YES (ver ${OPENNI2_VERSION})"
                                                                                                   ELSE NO)
endif()

if(WITH_PVAPI)
  status("    PvAPI:"          HAVE_PVAPI          THEN YES                                        ELSE NO)
else()
  status("    PvAPI:"  "NO")
endif()

if(WITH_GIGEAPI)
  status("    GigEVisionSDK:"  HAVE_GIGE_API       THEN YES                                        ELSE NO)
else()
  status("    GigEVisionSDK:"  "NO")
endif()

# Order is similar to CV_PARALLEL_FRAMEWORK in core/src/parallel.cpp
v4r_clear_vars(CV_PARALLEL_FRAMEWORK)
if(HAVE_TBB)
  set(CV_PARALLEL_FRAMEWORK "TBB (ver ${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR} interface ${TBB_INTERFACE_VERSION})")
elseif(HAVE_CSTRIPES)
  set(CV_PARALLEL_FRAMEWORK "C=")
elseif(HAVE_OPENMP)
  set(CV_PARALLEL_FRAMEWORK "OpenMP")
elseif(HAVE_GCD)
  set(CV_PARALLEL_FRAMEWORK "GCD")
elseif(WINRT OR HAVE_CONCURRENCY)
  set(CV_PARALLEL_FRAMEWORK "Concurrency")
elseif(HAVE_PTHREADS_PF)
  set(CV_PARALLEL_FRAMEWORK "pthreads")
else()
  set(CV_PARALLEL_FRAMEWORK "none")
endif()
status("")
status("  Parallel framework:" TRUE THEN "${CV_PARALLEL_FRAMEWORK}" ELSE NO)


# ========================== Other third-party libraries ==========================
status("")
status("  Other third-party libraries:")

status("    Use Eigen:"      HAVE_EIGEN       THEN "YES (ver ${EIGEN_WORLD_VERSION}.${EIGEN_MAJOR_VERSION}.${EIGEN_MINOR_VERSION})" ELSE NO)
status("    Use GLM:"        HAVE_GLM         THEN "YES (ver ${GLM_VERSION})" ELSE NO)
status("    Use Cuda:"       HAVE_CUDA        THEN "YES (ver ${CUDA_VERSION_STRING})" ELSE NO)
status("    Use OpenCL:"     HAVE_OPENCL      THEN YES ELSE NO)
status("    Use GLEW:"       HAVE_GLEW        THEN YES ELSE NO)

if(HAVE_CUDA)
  status("")
  status("  NVIDIA CUDA")

  status("    Use CUFFT:"            HAVE_CUFFT   THEN YES ELSE NO)
  status("    Use CUBLAS:"           HAVE_CUBLAS  THEN YES ELSE NO)
  status("    USE NVCUVID:"          HAVE_NVCUVID THEN YES ELSE NO)
  status("    NVIDIA GPU arch:"      ${V4R_CUDA_ARCH_BIN})
  status("    NVIDIA PTX archs:"     ${V4R_CUDA_ARCH_PTX})
  status("    Use fast math:"        CUDA_FAST_MATH THEN YES ELSE NO)
endif()

if(HAVE_OPENCL)
  status("")
  status("  OpenCL:")
  if(HAVE_OPENCL_STATIC)
    set(__opencl_ver "static")
  else()
    set(__opencl_ver "dynamic")
  endif()
  status("    Version:"       ${__opencl_ver})
  if(OPENCL_INCLUDE_DIR)
    status("    Include path:"       ${OPENCL_INCLUDE_DIRS})
  endif()
  if(OPENCL_LIBRARIES)
    set(__libs "")
    foreach(l ${OPENCL_LIBRARIES})
      if(TARGET ${l})
        get_target_property(p ${l} LOCATION)
        if(p MATCHES NOTFOUND)
          list(APPEND __libs "${l}")
        else()
          list(APPEND __libs "${p}")
        endif()
      else()
        list(APPEND __libs "${l}")
      endif()
    endforeach()
    status("    libraries:"          ${__libs})
  endif()
  status("    Use AMDFFT:"           HAVE_CLAMDFFT  THEN YES ELSE NO)
  status("    Use AMDBLAS:"          HAVE_CLAMDBLAS THEN YES ELSE NO)
endif()

# ========================== documentation ==========================
if(BUILD_DOCS)
  status("")
  status("  Documentation:")
  status("    Doxygen:"             DOXYGEN_FOUND             THEN "${DOXYGEN_EXECUTABLE} (ver ${DOXYGEN_VERSION})" ELSE NO)
endif()

# ========================== samples and tests ==========================
status("")
status("  Tests and samples:")
status("    Tests:"             BUILD_TESTS AND HAVE_v4r_ts       THEN YES ELSE NO)
status("    Performance tests:" BUILD_PERF_TESTS AND HAVE_v4r_ts  THEN YES ELSE NO)
status("    C/C++ Examples:"    BUILD_EXAMPLES                       THEN YES ELSE NO)

# ========================== auxiliary ==========================
status("")
status("  Install path:" "${CMAKE_INSTALL_PREFIX}")
status("")
status("  v4r_config.h is in:" "${V4R_CONFIG_FILE_INCLUDE_DIR}")
status("-----------------------------------------------------------------")
status("")

v4r_finalize_status()

# ----------------------------------------------------------------------------
# Warn in the case of in-source build
# ----------------------------------------------------------------------------
if("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}")
  message(WARNING "The source directory is the same as binary directory. \"make clean\" may damage the source tree")
endif()

# ----------------------------------------------------------------------------
# CPack stuff
# ----------------------------------------------------------------------------

# TODO
#include(cmake/V4RPackaging.cmake)
