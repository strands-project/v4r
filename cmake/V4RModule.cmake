# Local variables (set for each module):
#
# name       - short name in lower case i.e. core
# the_module - full name in lower case i.e. v4r_core

# Global variables:
#
# V4R_MODULE_${the_module}_LOCATION
# V4R_MODULE_${the_module}_BINARY_DIR
# V4R_MODULE_${the_module}_DESCRIPTION
# V4R_MODULE_${the_module}_CLASS - PUBLIC|INTERNAL|BINDINGS
# V4R_MODULE_${the_module}_HEADERS
# V4R_MODULE_${the_module}_SOURCES
# V4R_MODULE_${the_module}_DEPS - final flattened set of module dependencies
# V4R_MODULE_${the_module}_DEPS_EXT - non-module dependencies
# V4R_MODULE_${the_module}_REQ_DEPS
# V4R_MODULE_${the_module}_OPT_DEPS
# V4R_MODULE_${the_module}_PRIVATE_REQ_DEPS
# V4R_MODULE_${the_module}_PRIVATE_OPT_DEPS
# V4R_MODULE_${the_module}_CUDA_OBJECTS - compiled CUDA objects list
# V4R_MODULE_${the_module}_CHILDREN - list of submodules for compound modules (cmake >= 2.8.8)
# V4R_MODULE_${the_module}_WRAPPERS - list of wrappers supporting this module
# HAVE_${the_module} - for fast check of module availability

# To control the setup of the module you could also set:
# the_description - text to be used as current module description
# V4R_MODULE_TYPE - STATIC|SHARED - set to force override global settings for current module
# BUILD_${the_module}_INIT - ON|OFF (default ON) - initial value for BUILD_${the_module}
# V4R_MODULE_CHILDREN - list of submodules

# The verbose template for V4R module:
#
#   v4r_add_module(modname <dependencies>)
#   v4r_glob_module_sources(([EXCLUDE_CUDA] <extra sources&headers>)
#                          or glob them manually and v4r_set_module_sources(...)
#   v4r_module_include_directories(<extra include directories>)
#   v4r_create_module()
#   <add extra link dependencies, compiler options, etc>
#   <add extra installation rules>
#   v4r_add_accuracy_tests(<extra dependencies>)
#   v4r_add_perf_tests(<extra dependencies>)
#   v4r_add_samples(<extra dependencies>)
#
#
# If module have no "extra" then you can define it in one line:
#
#   v4r_define_module(modname <dependencies>)

# clean flags for modules enabled on previous cmake run
# this is necessary to correctly handle modules removal
foreach(mod ${V4R_MODULES_BUILD} ${V4R_MODULES_DISABLED_USER} ${V4R_MODULES_DISABLED_AUTO} ${V4R_MODULES_DISABLED_FORCE})
  if(HAVE_${mod})
    unset(HAVE_${mod} CACHE)
  endif()
  unset(V4R_MODULE_${mod}_REQ_DEPS CACHE)
  unset(V4R_MODULE_${mod}_OPT_DEPS CACHE)
  unset(V4R_MODULE_${mod}_PRIVATE_REQ_DEPS CACHE)
  unset(V4R_MODULE_${mod}_PRIVATE_OPT_DEPS CACHE)
  unset(V4R_MODULE_${mod}_LINK_DEPS CACHE)
  unset(V4R_MODULE_${mod}_WRAPPERS CACHE)
endforeach()

# clean modules info which needs to be recalculated
set(V4R_MODULES_PUBLIC         "" CACHE INTERNAL "List of V4R modules marked for export")
set(V4R_MODULES_BUILD          "" CACHE INTERNAL "List of V4R modules included into the build")
set(V4R_MODULES_DISABLED_USER  "" CACHE INTERNAL "List of V4R modules explicitly disabled by user")
set(V4R_MODULES_DISABLED_AUTO  "" CACHE INTERNAL "List of V4R modules implicitly disabled due to dependencies")
set(V4R_MODULES_DISABLED_FORCE "" CACHE INTERNAL "List of V4R modules which can not be build in current configuration")

# adds dependencies to V4R module
# Usage:
#   add_dependencies(v4r_<name> [REQUIRED] [<list of dependencies>] [OPTIONAL <list of modules>] [WRAP <list of wrappers>])
# Notes:
# * <list of dependencies> - can include full names of modules or full pathes to shared/static libraries or cmake targets
macro(v4r_add_dependencies full_modname)
  v4r_debug_message("v4r_add_dependencies(" ${full_modname} ${ARGN} ")")
  #we don't clean the dependencies here to allow this macro several times for every module
  foreach(d "REQUIRED" ${ARGN})
    if(d STREQUAL "REQUIRED")
      set(__depsvar V4R_MODULE_${full_modname}_REQ_DEPS)
    elseif(d STREQUAL "OPTIONAL")
      set(__depsvar V4R_MODULE_${full_modname}_OPT_DEPS)
    elseif(d STREQUAL "PRIVATE_REQUIRED")
      set(__depsvar V4R_MODULE_${full_modname}_PRIVATE_REQ_DEPS)
    elseif(d STREQUAL "PRIVATE_OPTIONAL")
      set(__depsvar V4R_MODULE_${full_modname}_PRIVATE_OPT_DEPS)
    elseif(d STREQUAL "WRAP")
      set(__depsvar V4R_MODULE_${full_modname}_WRAPPERS)
    else()
      list(APPEND ${__depsvar} "${d}")
    endif()
  endforeach()
  unset(__depsvar)

  v4r_list_unique(V4R_MODULE_${full_modname}_REQ_DEPS)
  v4r_list_unique(V4R_MODULE_${full_modname}_OPT_DEPS)
  v4r_list_unique(V4R_MODULE_${full_modname}_PRIVATE_REQ_DEPS)
  v4r_list_unique(V4R_MODULE_${full_modname}_PRIVATE_OPT_DEPS)
  v4r_list_unique(V4R_MODULE_${full_modname}_WRAPPERS)

  set(V4R_MODULE_${full_modname}_REQ_DEPS ${V4R_MODULE_${full_modname}_REQ_DEPS}
    CACHE INTERNAL "Required dependencies of ${full_modname} module")
  set(V4R_MODULE_${full_modname}_OPT_DEPS ${V4R_MODULE_${full_modname}_OPT_DEPS}
    CACHE INTERNAL "Optional dependencies of ${full_modname} module")
  set(V4R_MODULE_${full_modname}_PRIVATE_REQ_DEPS ${V4R_MODULE_${full_modname}_PRIVATE_REQ_DEPS}
    CACHE INTERNAL "Required private dependencies of ${full_modname} module")
  set(V4R_MODULE_${full_modname}_PRIVATE_OPT_DEPS ${V4R_MODULE_${full_modname}_PRIVATE_OPT_DEPS}
    CACHE INTERNAL "Optional private dependencies of ${full_modname} module")
  set(V4R_MODULE_${full_modname}_WRAPPERS ${V4R_MODULE_${full_modname}_WRAPPERS}
    CACHE INTERNAL "List of wrappers supporting module ${full_modname}")
endmacro()

# declare new V4R module in current folder
# Usage:
#   v4r_add_module(<name> [INTERNAL|BINDINGS] [REQUIRED] [<list of dependencies>] [OPTIONAL <list of optional dependencies>] [WRAP <list of wrappers>])
# Example:
#   v4r_add_module(yaom INTERNAL v4r_core v4r_highgui v4r_flann OPTIONAL v4r_cudev)
macro(v4r_add_module _name)
  v4r_debug_message("v4r_add_module(" ${_name} ${ARGN} ")")
  string(TOLOWER "${_name}" name)
  set(the_module v4r_${name})

  # the first pass - collect modules info, the second pass - create targets
  if(V4R_INITIAL_PASS)
    #guard agains redefinition
    if(";${V4R_MODULES_BUILD};${V4R_MODULES_DISABLED_USER};" MATCHES ";${the_module};")
      message(FATAL_ERROR "Redefinition of the ${the_module} module.
  at:                    ${CMAKE_CURRENT_SOURCE_DIR}
  previously defined at: ${V4R_MODULE_${the_module}_LOCATION}
")
    endif()

    if(NOT DEFINED the_description)
      set(the_description "The ${name} V4R module")
    endif()

    if(NOT DEFINED BUILD_${the_module}_INIT)
      set(BUILD_${the_module}_INIT ON)
    endif()

    # create option to enable/disable this module
    option(BUILD_${the_module} "Include ${the_module} module into the V4R build" ${BUILD_${the_module}_INIT})

    # remember the module details
    set(V4R_MODULE_${the_module}_DESCRIPTION "${the_description}" CACHE INTERNAL "Brief description of ${the_module} module")
    set(V4R_MODULE_${the_module}_LOCATION    "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "Location of ${the_module} module sources")

    set(V4R_MODULE_${the_module}_LINK_DEPS "" CACHE INTERNAL "")

    # parse list of dependencies
    if("${ARGV1}" STREQUAL "INTERNAL" OR "${ARGV1}" STREQUAL "BINDINGS")
      set(V4R_MODULE_${the_module}_CLASS "${ARGV1}" CACHE INTERNAL "The category of the module")
      set(__v4r_argn__ ${ARGN})
      list(REMOVE_AT __v4r_argn__ 0)
      v4r_add_dependencies(${the_module} ${__v4r_argn__})
      unset(__v4r_argn__)
    else()
      set(V4R_MODULE_${the_module}_CLASS "PUBLIC" CACHE INTERNAL "The category of the module")
      v4r_add_dependencies(${the_module} ${ARGN})
      if(BUILD_${the_module})
        set(V4R_MODULES_PUBLIC ${V4R_MODULES_PUBLIC} "${the_module}" CACHE INTERNAL "List of V4R modules marked for export")
      endif()
    endif()

    if(BUILD_${the_module})
      set(V4R_MODULES_BUILD ${V4R_MODULES_BUILD} "${the_module}" CACHE INTERNAL "List of V4R modules included into the build")
    else()
      set(V4R_MODULES_DISABLED_USER ${V4R_MODULES_DISABLED_USER} "${the_module}" CACHE INTERNAL "List of V4R modules explicitly disabled by user")
    endif()

    # add submodules if any
    set(V4R_MODULE_${the_module}_CHILDREN "${V4R_MODULE_CHILDREN}" CACHE INTERNAL "List of ${the_module} submodules")

    # add reverse wrapper dependencies
    foreach (wrapper ${V4R_MODULE_${the_module}_WRAPPERS})
      v4r_add_dependencies(v4r_${wrapper} OPTIONAL ${the_module})
    endforeach()

    # stop processing of current file
    return()
  else()
    set(V4R_MODULE_${the_module}_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL "")
    if(NOT BUILD_${the_module})
      return() # extra protection from redefinition
    endif()
    project(${the_module})
  endif()
endmacro()

# excludes module from current configuration
macro(v4r_module_disable module)
  set(__modname ${module})
  if(NOT __modname MATCHES "^v4r_")
    set(__modname v4r_${module})
  endif()
  list(APPEND V4R_MODULES_DISABLED_FORCE "${__modname}")
  set(HAVE_${__modname} OFF CACHE INTERNAL "Module ${__modname} can not be built in current configuration")
  set(V4R_MODULE_${__modname}_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "Location of ${__modname} module sources")
  set(V4R_MODULES_DISABLED_FORCE "${V4R_MODULES_DISABLED_FORCE}" CACHE INTERNAL "List of V4R modules which can not be build in current configuration")
  if(BUILD_${__modname})
    # touch variable controlling build of the module to suppress "unused variable" CMake warning
  endif()
  unset(__modname)
  return() # leave the current folder
endmacro()


# collect modules from specified directories
# NB: must be called only once!
macro(v4r_glob_modules)
  if(DEFINED V4R_INITIAL_PASS)
    message(FATAL_ERROR "V4R has already loaded its modules. Calling v4r_glob_modules second time is not allowed.")
  endif()
  set(__directories_observed "")

  # collect modules
  set(V4R_INITIAL_PASS ON)
  set(V4R_PROCESSING_EXTRA_MODULES 0)
  foreach(__path ${ARGN})
    if("${__path}" STREQUAL "EXTRA")
      set(V4R_PROCESSING_EXTRA_MODULES 1)
    endif()
    get_filename_component(__path "${__path}" ABSOLUTE)

    list(FIND __directories_observed "${__path}" __pathIdx)
    if(__pathIdx GREATER -1)
      message(FATAL_ERROR "The directory ${__path} is observed for V4R modules second time.")
    endif()
    list(APPEND __directories_observed "${__path}")

    file(GLOB __v4rmodules RELATIVE "${__path}" "${__path}/*")
    if(__v4rmodules)
      list(SORT __v4rmodules)
      foreach(mod ${__v4rmodules})
        get_filename_component(__modpath "${__path}/${mod}" ABSOLUTE)
        if(EXISTS "${__modpath}/CMakeLists.txt")

          list(FIND __directories_observed "${__modpath}" __pathIdx)
          if(__pathIdx GREATER -1)
            message(FATAL_ERROR "The module from ${__modpath} is already loaded.")
          endif()
          list(APPEND __directories_observed "${__modpath}")

          add_subdirectory("${__modpath}" "${CMAKE_CURRENT_BINARY_DIR}/${mod}/.${mod}")
        endif()
      endforeach()
    endif()
  endforeach()
  v4r_clear_vars(__v4rmodules __directories_observed __path __modpath __pathIdx)

  # resolve dependencies
  __v4r_resolve_dependencies()

  # create modules
  set(V4R_INITIAL_PASS OFF PARENT_SCOPE)
  set(V4R_INITIAL_PASS OFF)
  foreach(m ${V4R_MODULES_BUILD})
    if(m MATCHES "^v4r_")
      string(REGEX REPLACE "^v4r_" "" __shortname "${m}")
      add_subdirectory("${V4R_MODULE_${m}_LOCATION}" "${CMAKE_CURRENT_BINARY_DIR}/${__shortname}")
    else()
      message(WARNING "Check module name: ${m}")
      add_subdirectory("${V4R_MODULE_${m}_LOCATION}" "${CMAKE_CURRENT_BINARY_DIR}/${m}")
    endif()
  endforeach()
  unset(__shortname)
endmacro()


# disables V4R module with missing dependencies
function(__v4r_module_turn_off the_module)
  list(REMOVE_ITEM V4R_MODULES_DISABLED_AUTO "${the_module}")
  list(APPEND V4R_MODULES_DISABLED_AUTO "${the_module}")
  list(REMOVE_ITEM V4R_MODULES_BUILD "${the_module}")
  list(REMOVE_ITEM V4R_MODULES_PUBLIC "${the_module}")
  set(HAVE_${the_module} OFF CACHE INTERNAL "Module ${the_module} can not be built in current configuration")

  set(V4R_MODULES_DISABLED_AUTO "${V4R_MODULES_DISABLED_AUTO}" CACHE INTERNAL "")
  set(V4R_MODULES_BUILD "${V4R_MODULES_BUILD}" CACHE INTERNAL "")
  set(V4R_MODULES_PUBLIC "${V4R_MODULES_PUBLIC}" CACHE INTERNAL "")
endfunction()

# sort modules by dependencies
function(__v4r_sort_modules_by_deps __lst)
  v4r_list_sort(${__lst})
  set(input ${${__lst}})
  set(result "")
  while(input)
    list(LENGTH input length_before)
    foreach (m ${input})
      # check if module is in the result already
      if (NOT ";${result};" MATCHES ";${m};")
        # scan through module dependencies...
        set(unresolved_deps_found FALSE)
        foreach (d ${V4R_MODULE_${m}_CHILDREN} ${V4R_MODULE_${m}_DEPS})
          # ... which are not already in the result and are enabled
          if ((NOT ";${result};" MATCHES ";${d};") AND HAVE_${d})
            set(unresolved_deps_found TRUE)
            break()
          endif()
        endforeach()
        # chek if all dependencies for this module has been resolved
        if (NOT unresolved_deps_found)
          list(APPEND result ${m})
          list(REMOVE_ITEM input ${m})
        endif()
      endif()
    endforeach()
    list(LENGTH input length_after)
    # check for infinite loop or unresolved dependencies
    if (NOT length_after LESS length_before)
      message(WARNING "Unresolved dependencies or loop in dependency graph (${length_after})\n"
        "Processed ${__lst}: ${${__lst}}\n"
        "Good modules: ${result}\n"
        "Bad modules: ${input}"
      )
      list(APPEND result ${input})
      break()
    endif()
  endwhile()
  set(${__lst} "${result}" PARENT_SCOPE)
endfunction()

# resolve dependensies
function(__v4r_resolve_dependencies)
  foreach(m ${V4R_MODULES_DISABLED_USER})
    set(HAVE_${m} OFF CACHE INTERNAL "Module ${m} will not be built in current configuration")
  endforeach()
  foreach(m ${V4R_MODULES_BUILD})
    set(HAVE_${m} ON CACHE INTERNAL "Module ${m} will be built in current configuration")
  endforeach()

  # disable MODULES with unresolved dependencies
  set(has_changes ON)
  while(has_changes)
    set(has_changes OFF)
    foreach(m ${V4R_MODULES_BUILD})
      set(__deps ${V4R_MODULE_${m}_REQ_DEPS} ${V4R_MODULE_${m}_PRIVATE_REQ_DEPS})
      while(__deps)
        v4r_list_pop_front(__deps d)
        string(TOUPPER "${d}" upper_d)
        if(NOT (HAVE_${d} OR HAVE_${upper_d} OR TARGET ${d} OR EXISTS ${d}))
          #if(d MATCHES "^v4r_") # TODO Remove this condition in the future and use HAVE_ variables only
            message(STATUS "Module ${m} disabled because ${d} dependency can't be resolved!")
            __v4r_module_turn_off(${m})
            set(has_changes ON)
            break()
          #else()
            #message(STATUS "Assume that non-module dependency is available: ${d} (for module ${m})")
          #endif()
        else()
        endif()
      endwhile()
    endforeach()
  endwhile()

#  message(STATUS "List of active modules: ${V4R_MODULES_BUILD}")

  foreach(m ${V4R_MODULES_BUILD})
    set(deps_${m} ${V4R_MODULE_${m}_REQ_DEPS})
    foreach(d ${V4R_MODULE_${m}_OPT_DEPS})
      if(NOT (";${deps_${m}};" MATCHES ";${d};"))
        string(TOUPPER "${d}" upper_d)
        if(HAVE_${upper_d} OR HAVE_${d} OR TARGET ${d})
          list(APPEND deps_${m} ${d})
        endif()
      endif()
    endforeach()
#    message(STATUS "Initial deps of ${m} (w/o private deps): ${deps_${m}}")
  endforeach()

  # propagate dependencies
  set(has_changes ON)
  while(has_changes)
    set(has_changes OFF)
    foreach(m2 ${V4R_MODULES_BUILD}) # transfer deps of m2 to m
      foreach(m ${V4R_MODULES_BUILD})
        if((NOT m STREQUAL m2) AND ";${deps_${m}};" MATCHES ";${m2};")
          foreach(d ${deps_${m2}})
            if(NOT (";${deps_${m}};" MATCHES ";${d};"))
#              message(STATUS "  Transfer dependency ${d} from ${m2} to ${m}")
              list(APPEND deps_${m} ${d})
              set(has_changes ON)
            endif()
          endforeach()
        endif()
      endforeach()
    endforeach()
  endwhile()

  # process private deps
  foreach(m ${V4R_MODULES_BUILD})
    foreach(d ${V4R_MODULE_${m}_PRIVATE_REQ_DEPS})
      if(NOT (";${deps_${m}};" MATCHES ";${d};"))
        list(APPEND deps_${m} ${d})
      endif()
    endforeach()
    foreach(d ${V4R_MODULE_${m}_PRIVATE_OPT_DEPS})
      if(NOT (";${deps_${m}};" MATCHES ";${d};"))
        if(HAVE_${d} OR TARGET ${d})
          list(APPEND deps_${m} ${d})
        endif()
      endif()
    endforeach()
  endforeach()

  v4r_list_sort(V4R_MODULES_BUILD)

  foreach(m ${V4R_MODULES_BUILD})
#    message(STATUS "FULL deps of ${m}: ${deps_${m}}")
    set(V4R_MODULE_${m}_DEPS ${deps_${m}})
    set(V4R_MODULE_${m}_DEPS_EXT ${deps_${m}})
    v4r_list_filterout(V4R_MODULE_${m}_DEPS_EXT "^v4r_[^ ]+$")
    if(V4R_MODULE_${m}_DEPS_EXT AND V4R_MODULE_${m}_DEPS)
      list(REMOVE_ITEM V4R_MODULE_${m}_DEPS ${V4R_MODULE_${m}_DEPS_EXT})
    endif()
  endforeach()

  # reorder dependencies
  foreach(m ${V4R_MODULES_BUILD})
    __v4r_sort_modules_by_deps(V4R_MODULE_${m}_DEPS)
    set(LINK_DEPS ${V4R_MODULE_${m}_DEPS})
    set(LINK_DEPS_EXT)
    v4r_list_sort(V4R_MODULE_${m}_DEPS_EXT)

    foreach(d ${V4R_MODULE_${m}_DEPS_EXT})
      string(TOUPPER "${d}" upper_d)
      if(DEFINED ${upper_d}_LIBRARIES)
        list(APPEND LINK_DEPS_EXT ${${upper_d}_LIBRARIES})
      else()
        list(APPEND LINK_DEPS_EXT ${d})
      endif()
    endforeach()
    v4r_list_unique(LINK_DEPS_EXT)

    set(V4R_MODULE_${m}_DEPS ${V4R_MODULE_${m}_DEPS} CACHE INTERNAL "Flattened dependencies of ${m} module")
    set(V4R_MODULE_${m}_DEPS_EXT ${V4R_MODULE_${m}_DEPS_EXT} CACHE INTERNAL "Extra dependencies of ${m} module")
    set(V4R_MODULE_${m}_DEPS_TO_LINK ${LINK_DEPS} CACHE INTERNAL "Flattened dependencies of ${m} module (for linker)")
    set(V4R_MODULE_${m}_DEPS_EXT_TO_LINK ${LINK_DEPS_EXT} CACHE INTERNAL "Flattened extra dependencies of ${m} module (for linker)")

#    message(STATUS "  module deps of ${m}: ${V4R_MODULE_${m}_DEPS}")
#    message(STATUS "  module link deps of ${m}: ${V4R_MODULE_${m}_DEPS_TO_LINK}")
#    message(STATUS "  extra deps of ${m}: ${V4R_MODULE_${m}_DEPS_EXT}")
#    message(STATUS "")
  endforeach()

  __v4r_sort_modules_by_deps(V4R_MODULES_BUILD)

  set(V4R_MODULES_PUBLIC        ${V4R_MODULES_PUBLIC}        CACHE INTERNAL "List of V4R modules marked for export")
  set(V4R_MODULES_BUILD         ${V4R_MODULES_BUILD}         CACHE INTERNAL "List of V4R modules included into the build")
  set(V4R_MODULES_DISABLED_AUTO ${V4R_MODULES_DISABLED_AUTO} CACHE INTERNAL "List of V4R modules implicitly disabled due to dependencies")
endfunction()


# setup include paths for the list of passed modules
macro(v4r_include_modules)
  foreach(d ${ARGN})
    if(d MATCHES "^v4r_" AND HAVE_${d})
      if (EXISTS "${V4R_MODULE_${d}_LOCATION}/include")
        v4r_include_directories("${V4R_MODULE_${d}_LOCATION}/include")
      endif()
    elseif(EXISTS "${d}")
      v4r_include_directories("${d}")
    endif()
  endforeach()
endmacro()

# same as previous but with dependencies
macro(v4r_include_modules_recurse)
  v4r_include_modules(${ARGN})
  foreach(d ${ARGN})
    if(d MATCHES "^v4r_" AND HAVE_${d} AND DEFINED V4R_MODULE_${d}_DEPS)
      foreach (sub ${V4R_MODULE_${d}_DEPS})
        v4r_include_modules(${sub})
      endforeach()
    endif()
  endforeach()
endmacro()

# setup include paths for the list of passed modules
macro(v4r_target_include_modules target)
  foreach(d ${ARGN})
    string(TOUPPER "${d}" upper_d)
    if(d MATCHES "^v4r_" AND HAVE_${d})
      if (EXISTS "${V4R_MODULE_${d}_LOCATION}/include")
        v4r_target_include_directories(${target} "${V4R_MODULE_${d}_LOCATION}/include")
      endif()
    elseif(EXISTS "${d}")
      v4r_target_include_directories(${target} "${d}")
    elseif(${upper_d}_INCLUDE_DIRS)
      v4r_target_include_directories(${target} "${${upper_d}_INCLUDE_DIRS}")
    else()
    endif()
  endforeach()
endmacro()

# setup include paths for the list of passed modules and recursively add dependent modules
macro(v4r_target_include_modules_recurse target)
  foreach(d ${ARGN})
    if(d MATCHES "^v4r_" AND HAVE_${d})
      if (EXISTS "${V4R_MODULE_${d}_LOCATION}/include")
        v4r_target_include_directories(${target} "${V4R_MODULE_${d}_LOCATION}/include")
      endif()
      if(V4R_MODULE_${d}_DEPS)
        v4r_target_include_modules(${target} ${V4R_MODULE_${d}_DEPS})
      endif()
    elseif(EXISTS "${d}")
      v4r_target_include_directories(${target} "${d}")
    endif()
  endforeach()
endmacro()

# setup include path for V4R headers for specified module
# v4r_module_include_directories(<extra include directories/extra include modules>)
macro(v4r_module_include_directories)
  v4r_target_include_directories(${the_module}
      "${V4R_MODULE_${the_module}_LOCATION}/include"
      "${V4R_MODULE_${the_module}_LOCATION}/src"
      "${CMAKE_CURRENT_BINARY_DIR}" # for precompiled headers
      )
  v4r_target_include_modules(${the_module} ${V4R_MODULE_${the_module}_DEPS} ${V4R_MODULE_${the_module}_DEPS_EXT} ${ARGN})
endmacro()


# sets header and source files for the current module
# NB: all files specified as headers will be installed
# Usage:
# v4r_set_module_sources([HEADERS] <list of files> [SOURCES] <list of files>)
macro(v4r_set_module_sources)
  v4r_debug_message("v4r_set_module_sources(" ${ARGN} ")")

  set(V4R_MODULE_${the_module}_HEADERS "")
  set(V4R_MODULE_${the_module}_SOURCES "")

  foreach(f "HEADERS" ${ARGN})
    if(f STREQUAL "HEADERS" OR f STREQUAL "SOURCES")
      set(__filesvar "V4R_MODULE_${the_module}_${f}")
    else()
      list(APPEND ${__filesvar} "${f}")
    endif()
  endforeach()

  # the hacky way to embeed any files into the V4R without modification of its build system
  if(COMMAND v4r_get_module_external_sources)
    v4r_get_module_external_sources()
  endif()

  # use full paths for module to be independent from the module location
  v4r_convert_to_full_paths(V4R_MODULE_${the_module}_HEADERS)

  set(V4R_MODULE_${the_module}_HEADERS ${V4R_MODULE_${the_module}_HEADERS} CACHE INTERNAL "List of header files for ${the_module}")
  set(V4R_MODULE_${the_module}_SOURCES ${V4R_MODULE_${the_module}_SOURCES} CACHE INTERNAL "List of source files for ${the_module}")
endmacro()

# finds and sets headers and sources for the standard V4R module
# Usage:
# v4r_glob_module_sources([EXCLUDE_CUDA] <extra sources&headers in the same format as used in v4r_set_module_sources>)
macro(v4r_glob_module_sources)
  v4r_debug_message("v4r_glob_module_sources(" ${ARGN} ")")
  set(_argn ${ARGN})
  list(FIND _argn "EXCLUDE_CUDA" exclude_cuda)
  if(NOT exclude_cuda EQUAL -1)
    list(REMOVE_AT _argn ${exclude_cuda})
  endif()

  file(GLOB_RECURSE lib_srcs
       "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp"
  )
  #file(GLOB_RECURSE lib_int_hdrs
       #"${CMAKE_CURRENT_LIST_DIR}/src/*.hpp"
       #"${CMAKE_CURRENT_LIST_DIR}/src/*.h"
  #)
  file(GLOB lib_hdrs
       #"${CMAKE_CURRENT_LIST_DIR}/include/v4r/*.hpp"
       "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/*.h"
       "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/impl/*.hpp"
  )
  file(GLOB lib_hdrs_detail
       "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/detail/*.h"
       "${CMAKE_CURRENT_LIST_DIR}/include/v4r/${name}/detail/*.hpp"
  )

  set(lib_cuda_srcs "")
  set(lib_cuda_hdrs "")
  if(HAVE_CUDA AND exclude_cuda EQUAL -1)
    file(GLOB lib_cuda_srcs
         "${CMAKE_CURRENT_LIST_DIR}/src/cuda/*.cu"
    )
    file(GLOB lib_cuda_hdrs
         "${CMAKE_CURRENT_LIST_DIR}/src/cuda/*.hpp"
    )
    source_group("Src\\Cuda"      FILES ${lib_cuda_srcs} ${lib_cuda_hdrs})
  endif()

  file(GLOB cl_kernels
       "${CMAKE_CURRENT_LIST_DIR}/src/opencl/*.cl"
  )
  if(cl_kernels)
    set(OCL_NAME opencl_kernels_${name})
    v4r_include_directories(${OPENCL_INCLUDE_DIRS})
    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.cpp" "${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.hpp"
      COMMAND ${CMAKE_COMMAND} "-DMODULE_NAME=${name}" "-DCL_DIR=${CMAKE_CURRENT_LIST_DIR}/src/opencl" "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.cpp" -P "${V4R_SOURCE_DIR}/cmake/cl2cpp.cmake"
      DEPENDS ${cl_kernels} "${V4R_SOURCE_DIR}/cmake/cl2cpp.cmake")
    v4r_source_group("Src\\opencl\\kernels" FILES ${cl_kernels})
    v4r_source_group("Src\\opencl\\kernels\\autogenerated" FILES "${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.cpp" "${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.hpp")
    list(APPEND lib_srcs ${cl_kernels} "${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.cpp" "${CMAKE_CURRENT_BINARY_DIR}/${OCL_NAME}.hpp")
  endif()

  v4r_set_module_sources(${_argn} HEADERS ${lib_hdrs} ${lib_hdrs_detail}
                         SOURCES ${lib_srcs} ${lib_int_hdrs} ${lib_cuda_srcs} ${lib_cuda_hdrs})
endmacro()

# creates V4R module in current folder
# creates new target, configures standard dependencies, compilers flags, install rules
# Usage:
#   v4r_create_module(<extra link dependencies>)
#   v4r_create_module()
macro(v4r_create_module)
  v4r_debug_message("v4r_create_module(" ${ARGN} ")")
  set(V4R_MODULE_${the_module}_LINK_DEPS "${V4R_MODULE_${the_module}_LINK_DEPS};${ARGN}" CACHE INTERNAL "")
  _v4r_create_module(${ARGN})
  set(the_module_target ${the_module})
endmacro()

macro(_v4r_create_module)
  set(sub_objs "")
  set(sub_links "")
  set(cuda_objs "")
  if (V4R_MODULE_${the_module}_CHILDREN)
    message(STATUS "Complex module ${the_module}")
    foreach (m ${V4R_MODULE_${the_module}_CHILDREN})
      if (BUILD_${m} AND TARGET ${m}_object)
        get_target_property(_sub_links ${m} LINK_LIBRARIES)
        list(APPEND sub_objs $<TARGET_OBJECTS:${m}_object>)
        list(APPEND sub_links ${_sub_links})
        message(STATUS "    + ${m}")
      else()
        message(STATUS "    - ${m}")
      endif()
      list(APPEND cuda_objs ${V4R_MODULE_${m}_CUDA_OBJECTS})
    endforeach()
  endif()

  v4r_add_library(${the_module} ${V4R_MODULE_TYPE} ${V4R_MODULE_${the_module}_HEADERS} ${V4R_MODULE_${the_module}_SOURCES}
    "${V4R_CONFIG_FILE_INCLUDE_DIR}/v4r_config.h" "${V4R_CONFIG_FILE_INCLUDE_DIR}/v4r_modules.h" ${sub_objs})

  if (cuda_objs)
    target_link_libraries(${the_module} ${cuda_objs})
  endif()

  # TODO: is it needed?
  if (sub_links)
    v4r_list_filterout(sub_links "^v4r_")
    v4r_list_unique(sub_links)
    target_link_libraries(${the_module} ${sub_links})
  endif()

  unset(sub_objs)
  unset(sub_links)
  unset(cuda_objs)

  v4r_target_link_libraries(${the_module} ${V4R_MODULE_${the_module}_DEPS_TO_LINK})
  v4r_target_link_libraries(${the_module} LINK_INTERFACE_LIBRARIES ${V4R_MODULE_${the_module}_DEPS_TO_LINK})
  set(_lil_deps_ext)
  foreach(d ${V4R_MODULE_${the_module}_DEPS_EXT_TO_LINK})
    if(TARGET ${d})
      get_target_property(_target_type ${d} TYPE)
      if(NOT ("${_target_type}" STREQUAL "STATIC_LIBRARY" OR BUILD_SHARED_LIBS))
        list(APPEND _lil_deps_ext ${d})
      endif()
    else()
      list(APPEND _lil_deps_ext ${d})
    endif()
  endforeach()
  v4r_target_link_libraries(${the_module} LINK_INTERFACE_LIBRARIES ${_lil_deps_ext})
  v4r_target_link_libraries(${the_module} ${V4R_MODULE_${the_module}_DEPS_EXT_TO_LINK} ${V4R_LINKER_LIBS} ${ARGN})
  if (HAVE_CUDA)
    v4r_target_link_libraries(${the_module} ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY})
  endif()

  add_dependencies(v4r_modules ${the_module})

  set_target_properties(${the_module} PROPERTIES
    OUTPUT_NAME "${the_module}${V4R_DLLVERSION}"
    DEBUG_POSTFIX "${V4R_DEBUG_POSTFIX}"
    ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
    INSTALL_NAME_DIR lib
  )

  # For dynamic link numbering convenions
  set_target_properties(${the_module} PROPERTIES
    VERSION ${V4R_LIBVERSION}
    SOVERSION ${V4R_SOVERSION}
  )

  if((NOT DEFINED V4R_MODULE_TYPE AND BUILD_SHARED_LIBS)
      OR (DEFINED V4R_MODULE_TYPE AND V4R_MODULE_TYPE STREQUAL SHARED))
    set_target_properties(${the_module} PROPERTIES COMPILE_DEFINITIONS V4RAPI_EXPORTS)
    set_target_properties(${the_module} PROPERTIES DEFINE_SYMBOL V4RAPI_EXPORTS)
  endif()

  v4r_install_target(${the_module} EXPORT V4RModules OPTIONAL
    RUNTIME DESTINATION ${V4R_BIN_INSTALL_PATH} COMPONENT libs
    LIBRARY DESTINATION ${V4R_LIB_INSTALL_PATH} COMPONENT libs NAMELINK_SKIP
    ARCHIVE DESTINATION ${V4R_LIB_INSTALL_PATH} COMPONENT dev
    )
  get_target_property(_target_type ${the_module} TYPE)
  if("${_target_type}" STREQUAL "SHARED_LIBRARY")
    install(TARGETS ${the_module}
      LIBRARY DESTINATION ${V4R_LIB_INSTALL_PATH} COMPONENT dev NAMELINK_ONLY)
  endif()

  foreach(m ${V4R_MODULE_${the_module}_CHILDREN} ${the_module})
    # only "public" headers need to be installed
    if(V4R_MODULE_${m}_HEADERS AND ";${V4R_MODULES_PUBLIC};" MATCHES ";${m};")
      foreach(hdr ${V4R_MODULE_${m}_HEADERS})
        string(REGEX REPLACE "^.*v4r/" "v4r/" hdr2 "${hdr}")
        if(NOT hdr2 MATCHES "v4r/${m}/private.*" AND hdr2 MATCHES "^(v4r/?.*)/[^/]+.h(..)?$" )
          install(FILES ${hdr} OPTIONAL DESTINATION "${V4R_INCLUDE_INSTALL_PATH}/${CMAKE_MATCH_1}" COMPONENT dev)
        endif()
      endforeach()
    endif()
  endforeach()

  if (TARGET ${the_module}_object)
    # copy COMPILE_DEFINITIONS
    get_target_property(main_defs ${the_module} COMPILE_DEFINITIONS)
    if (main_defs)
      set_target_properties(${the_module}_object PROPERTIES COMPILE_DEFINITIONS ${main_defs})
    endif()
  endif()
endmacro()

# short command for adding simple V4R module
# see v4r_add_module for argument details
# Usage:
# v4r_define_module(module_name  [INTERNAL] [EXCLUDE_CUDA] [REQUIRED] [<list of dependencies>] [OPTIONAL <list of optional dependencies>] [WRAP <list of wrappers>])
macro(v4r_define_module module_name)
  v4r_debug_message("v4r_define_module(" ${module_name} ${ARGN} ")")
  set(_argn ${ARGN})
  set(exclude_cuda "")
  foreach(arg ${_argn})
    if("${arg}" STREQUAL "EXCLUDE_CUDA")
      set(exclude_cuda "${arg}")
      list(REMOVE_ITEM _argn ${arg})
    endif()
  endforeach()

  v4r_add_module(${module_name} ${_argn})
  v4r_glob_module_sources(${exclude_cuda})
  v4r_module_include_directories()
  v4r_create_module()

  v4r_add_accuracy_tests()
  v4r_add_perf_tests()
  v4r_add_samples()
endmacro()

# ensures that all passed modules are available
# sets V4R_DEPENDENCIES_FOUND variable to TRUE/FALSE
macro(v4r_check_dependencies)
  set(V4R_DEPENDENCIES_FOUND TRUE)
  foreach(d ${ARGN})
    if(d MATCHES "^v4r_[^ ]+$" AND NOT HAVE_${d})
      set(V4R_DEPENDENCIES_FOUND FALSE)
      break()
    endif()
  endforeach()
endmacro()

# auxiliary macro to parse arguments of v4r_add_accuracy_tests and v4r_add_perf_tests commands
macro(__v4r_parse_test_sources tests_type)
  set(V4R_${tests_type}_${the_module}_SOURCES "")
  set(V4R_${tests_type}_${the_module}_DEPS "")
  set(__file_group_name "")
  set(__file_group_sources "")
  foreach(arg "DEPENDS_ON" ${ARGN} "FILES")
    if(arg STREQUAL "FILES")
      set(__currentvar "__file_group_sources")
      if(__file_group_name AND __file_group_sources)
        source_group("${__file_group_name}" FILES ${__file_group_sources})
        list(APPEND V4R_${tests_type}_${the_module}_SOURCES ${__file_group_sources})
      endif()
      set(__file_group_name "")
      set(__file_group_sources "")
    elseif(arg STREQUAL "DEPENDS_ON")
      set(__currentvar "V4R_${tests_type}_${the_module}_DEPS")
    elseif(" ${__currentvar}" STREQUAL " __file_group_sources" AND NOT __file_group_name) # spaces to avoid CMP0054
      set(__file_group_name "${arg}")
    else()
      list(APPEND ${__currentvar} "${arg}")
    endif()
  endforeach()
  unset(__file_group_name)
  unset(__file_group_sources)
  unset(__currentvar)
endmacro()

# this is a command for adding V4R performance tests to the module
# v4r_add_perf_tests(<extra_dependencies>)
function(v4r_add_perf_tests)
  v4r_debug_message("v4r_add_perf_tests(" ${ARGN} ")")

  set(perf_path "${CMAKE_CURRENT_LIST_DIR}/perf")
  if(BUILD_PERF_TESTS AND EXISTS "${perf_path}")
    __v4r_parse_test_sources(PERF ${ARGN})

    # v4r_imgcodecs is required for imread/imwrite
    set(perf_deps v4r_ts ${the_module} v4r_imgcodecs ${V4R_MODULE_${the_module}_DEPS} ${V4R_MODULE_v4r_ts_DEPS})
    v4r_check_dependencies(${perf_deps})

    if(V4R_DEPENDENCIES_FOUND)
      set(the_target "v4r_perf_${name}")
      # project(${the_target})

      if(NOT V4R_PERF_${the_module}_SOURCES)
        file(GLOB_RECURSE perf_srcs "${perf_path}/*.cpp")
        file(GLOB_RECURSE perf_hdrs "${perf_path}/*.hpp" "${perf_path}/*.h")
        v4r_source_group("Src" DIRBASE "${perf_path}" FILES ${perf_srcs})
        v4r_source_group("Include" DIRBASE "${perf_path}" FILES ${perf_hdrs})
        set(V4R_PERF_${the_module}_SOURCES ${perf_srcs} ${perf_hdrs})
      endif()

      v4r_add_executable(${the_target} ${V4R_PERF_${the_module}_SOURCES})
      v4r_target_include_modules(${the_target} ${perf_deps} "${perf_path}")
      v4r_target_link_libraries(${the_target} ${perf_deps} ${V4R_MODULE_${the_module}_DEPS} ${V4R_LINKER_LIBS})
      add_dependencies(v4r_perf_tests ${the_target})

      # Additional target properties
      set_target_properties(${the_target} PROPERTIES
        DEBUG_POSTFIX "${V4R_DEBUG_POSTFIX}"
        RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}"
      )

    else(V4R_DEPENDENCIES_FOUND)
      # TODO: warn about unsatisfied dependencies
    endif(V4R_DEPENDENCIES_FOUND)
    if(INSTALL_TESTS)
      install(TARGETS ${the_target} RUNTIME DESTINATION ${V4R_TEST_INSTALL_PATH} COMPONENT tests)
    endif()
  endif()
endfunction()

# this is a command for adding V4R accuracy/regression tests to the module
# v4r_add_accuracy_tests([FILES <source group name> <list of sources>] [DEPENDS_ON] <list of extra dependencies>)
function(v4r_add_accuracy_tests)
  v4r_debug_message("v4r_add_accuracy_tests(" ${ARGN} ")")

  set(test_path "${CMAKE_CURRENT_LIST_DIR}/test")
  if(BUILD_TESTS AND EXISTS "${test_path}")
    __v4r_parse_test_sources(TEST ${ARGN})

    # v4r_imgcodecs is required for imread/imwrite
    set(test_deps v4r_ts ${the_module} v4r_imgcodecs v4r_videoio ${V4R_MODULE_${the_module}_DEPS} ${V4R_MODULE_v4r_ts_DEPS})
    v4r_check_dependencies(${test_deps})
    if(V4R_DEPENDENCIES_FOUND)
      set(the_target "v4r_test_${name}")
      # project(${the_target})

      if(NOT V4R_TEST_${the_module}_SOURCES)
        file(GLOB_RECURSE test_srcs "${test_path}/*.cpp")
        file(GLOB_RECURSE test_hdrs "${test_path}/*.hpp" "${test_path}/*.h")
        v4r_source_group("Src" DIRBASE "${test_path}" FILES ${test_srcs})
        v4r_source_group("Include" DIRBASE "${test_path}" FILES ${test_hdrs})
        set(V4R_TEST_${the_module}_SOURCES ${test_srcs} ${test_hdrs})
      endif()

      v4r_add_executable(${the_target} ${V4R_TEST_${the_module}_SOURCES})
      v4r_target_include_modules(${the_target} ${test_deps} "${test_path}")
      v4r_target_link_libraries(${the_target} ${test_deps} ${V4R_MODULE_${the_module}_DEPS} ${V4R_LINKER_LIBS})
      add_dependencies(v4r_tests ${the_target})

      # Additional target properties
      set_target_properties(${the_target} PROPERTIES
        DEBUG_POSTFIX "${V4R_DEBUG_POSTFIX}"
        RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}"
      )

      enable_testing()
      get_target_property(LOC ${the_target} LOCATION)
      add_test(${the_target} "${LOC}")

    else(V4R_DEPENDENCIES_FOUND)
      # TODO: warn about unsatisfied dependencies
    endif(V4R_DEPENDENCIES_FOUND)

    if(INSTALL_TESTS)
      install(TARGETS ${the_target} RUNTIME DESTINATION ${V4R_TEST_INSTALL_PATH} COMPONENT tests)
    endif()
  endif()
endfunction()

function(v4r_add_samples)
  v4r_debug_message("v4r_add_samples(" ${ARGN} ")")

  set(samples_path "${CMAKE_CURRENT_SOURCE_DIR}/samples")
  string(REGEX REPLACE "^v4r_" "" module_id ${the_module})

  if(BUILD_EXAMPLES AND EXISTS "${samples_path}")
    set(samples_deps ${the_module} ${V4R_MODULE_${the_module}_DEPS} v4r_imgcodecs v4r_videoio v4r_highgui ${ARGN})
    v4r_check_dependencies(${samples_deps})

    if(V4R_DEPENDENCIES_FOUND)
      file(GLOB sample_sources "${samples_path}/*.cpp")

      foreach(source ${sample_sources})
        get_filename_component(name "${source}" NAME_WE)
        set(the_target "example_${module_id}_${name}")

        v4r_add_executable(${the_target} "${source}")
        v4r_target_include_modules(${the_target} ${samples_deps})
        v4r_target_link_libraries(${the_target} ${samples_deps})
        set_target_properties(${the_target} PROPERTIES PROJECT_LABEL "(sample) ${name}")

      endforeach()
    endif()
  endif()

  if(INSTALL_C_EXAMPLES AND EXISTS "${samples_path}")
  file(GLOB DEPLOY_FILES_AND_DIRS "${samples_path}/*")
    foreach(ITEM ${DEPLOY_FILES_AND_DIRS})
        IF( IS_DIRECTORY "${ITEM}" )
            LIST( APPEND sample_dirs "${ITEM}" )
        ELSE()
            LIST( APPEND sample_files "${ITEM}" )
        ENDIF()
    endforeach()
    install(FILES ${sample_files}
            DESTINATION ${V4R_SAMPLES_SRC_INSTALL_PATH}/${module_id}
            PERMISSIONS OWNER_READ GROUP_READ WORLD_READ COMPONENT samples)
    install(DIRECTORY ${sample_dirs}
            DESTINATION ${V4R_SAMPLES_SRC_INSTALL_PATH}/${module_id}
            USE_SOURCE_PERMISSIONS COMPONENT samples)
  endif()
endfunction()
