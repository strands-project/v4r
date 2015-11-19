# --------------------------------------------------------------------------------------------
#  Installation for CMake Module:  V4RConfig.cmake
#  Part 1/2: ${BIN_DIR}/V4RConfig.cmake              -> For use *without* "make install"
#  Part 2/2: ${BIN_DIR}/unix-install/V4RConfig.cmake -> For use with "make install"
# -------------------------------------------------------------------------------------------

if(INSTALL_TO_MANGLED_PATHS)
  set(V4R_USE_MANGLED_PATHS_CONFIGCMAKE TRUE)
else()
  set(V4R_USE_MANGLED_PATHS_CONFIGCMAKE FALSE)
endif()

if(NOT V4R_CUDA_CC)
  set(V4R_CUDA_CC_CONFIGCMAKE "\"\"")
  set(V4R_CUDA_VERSION "")
else()
  set(V4R_CUDA_CC_CONFIGCMAKE "${V4R_CUDA_CC}")
  set(V4R_CUDA_VERSION ${CUDA_VERSION_STRING})
endif()

set(V4R_ADD_DEBUG_RELEASE_CONFIGCMAKE FALSE)

#build list of modules available for the V4R user
set(V4R_LIB_COMPONENTS "")
foreach(m ${V4R_MODULES_PUBLIC})
  list(INSERT V4R_LIB_COMPONENTS 0 ${${m}_MODULE_DEPS_OPT} ${m})
endforeach()
v4r_list_unique(V4R_LIB_COMPONENTS)
set(V4R_MODULES_CONFIGCMAKE ${V4R_LIB_COMPONENTS})
v4r_list_filterout(V4R_LIB_COMPONENTS "^v4r_")
if(V4R_LIB_COMPONENTS)
  list(REMOVE_ITEM V4R_MODULES_CONFIGCMAKE ${V4R_LIB_COMPONENTS})
endif()

# -------------------------------------------------------------------------------------------
#  Part 1/2: ${BIN_DIR}/V4RConfig.cmake              -> For use *without* "make install"
# -------------------------------------------------------------------------------------------
set(V4R_INCLUDE_DIRS_CONFIGCMAKE "\"${V4R_CONFIG_FILE_INCLUDE_DIR}\"")

foreach(_dep ${deps_3rdparty})
  string(TOUPPER ${_dep} _DEP)
  set(_dir ${${_DEP}_INCLUDE_DIRS})
  if(_dir)
    list(APPEND V4R_INCLUDE_DIRS_CONFIGCMAKE "\"${_dir}\"")
  endif()
endforeach()

foreach(m ${V4R_MODULES_BUILD})
  if(EXISTS "${V4R_MODULE_${m}_LOCATION}/include")
    list(APPEND V4R_INCLUDE_DIRS_CONFIGCMAKE "\"${V4R_MODULE_${m}_LOCATION}/include\"")
  endif()
endforeach()
string(REPLACE ";" " " V4R_INCLUDE_DIRS_CONFIGCMAKE "${V4R_INCLUDE_DIRS_CONFIGCMAKE}")

export(TARGETS ${V4RModules_TARGETS} FILE "${CMAKE_BINARY_DIR}/V4RModules.cmake")

configure_file("${V4R_SOURCE_DIR}/cmake/templates/V4RConfig.cmake.in" "${CMAKE_BINARY_DIR}/V4RConfig.cmake" @ONLY)
configure_file("${V4R_SOURCE_DIR}/cmake/templates/V4RConfig-version.cmake.in" "${CMAKE_BINARY_DIR}/V4RConfig-version.cmake" @ONLY)

# --------------------------------------------------------------------------------------------
#  Part 2/2: ${BIN_DIR}/unix-install/V4RConfig.cmake -> For use *with* "make install"
# -------------------------------------------------------------------------------------------
set(V4R_INCLUDE_DIRS_CONFIGCMAKE "\"\${V4R_INSTALL_PATH}/${V4R_3P_INCLUDE_INSTALL_PATH}" "\${V4R_INSTALL_PATH}/${V4R_INCLUDE_INSTALL_PATH}\"")

#set(V4R2_INCLUDE_DIRS_CONFIGCMAKE "\"\"")
set(V4R_3RDPARTY_LIB_DIRS_CONFIGCMAKE "\"\${V4R_INSTALL_PATH}/${V4R_3P_LIB_INSTALL_PATH}\"")

if(UNIX)
  #http://www.vtk.org/Wiki/CMake/Tutorials/Packaging reference
  # For a command "find_package(<name> [major[.minor]] [EXACT] [REQUIRED|QUIET])"
  # cmake will look in the following dir on unix:
  #                <prefix>/(share|lib)/cmake/<name>*/                     (U)
  #                <prefix>/(share|lib)/<name>*/                           (U)
  #                <prefix>/(share|lib)/<name>*/(cmake|CMake)/             (U)
  configure_file("${V4R_SOURCE_DIR}/cmake/templates/V4RConfig.cmake.in" "${CMAKE_BINARY_DIR}/unix-install/V4RConfig.cmake" @ONLY)
  configure_file("${V4R_SOURCE_DIR}/cmake/templates/V4RConfig-version.cmake.in" "${CMAKE_BINARY_DIR}/unix-install/V4RConfig-version.cmake" @ONLY)
  install(FILES "${CMAKE_BINARY_DIR}/unix-install/V4RConfig.cmake" DESTINATION ${V4R_CONFIG_INSTALL_PATH}/ COMPONENT dev)
  install(FILES ${CMAKE_BINARY_DIR}/unix-install/V4RConfig-version.cmake DESTINATION ${V4R_CONFIG_INSTALL_PATH}/ COMPONENT dev)
  install(EXPORT V4RModules DESTINATION ${V4R_CONFIG_INSTALL_PATH}/ FILE V4RModules.cmake COMPONENT dev)
endif()
