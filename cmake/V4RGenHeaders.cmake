# platform-specific config file
configure_file("${V4R_SOURCE_DIR}/cmake/templates/v4r_config.h.in" "${V4R_CONFIG_FILE_INCLUDE_DIR}/v4r_config.h")
install(FILES "${V4R_CONFIG_FILE_INCLUDE_DIR}/v4r_config.h" DESTINATION ${V4R_INCLUDE_INSTALL_PATH} COMPONENT dev)

# ----------------------------------------------------------------------------
#  v4r_modules.h based on actual modules list
# ----------------------------------------------------------------------------
set(V4R_MODULE_DEFINITIONS_CONFIGMAKE "")

set(V4R_MOD_LIST ${V4R_MODULES_PUBLIC})
v4r_list_sort(V4R_MOD_LIST)
foreach(m ${V4R_MOD_LIST})
  string(TOUPPER "${m}" m)
  set(V4R_MODULE_DEFINITIONS_CONFIGMAKE "${V4R_MODULE_DEFINITIONS_CONFIGMAKE}#define HAVE_${m}\n")
endforeach()

set(V4R_MODULE_DEFINITIONS_CONFIGMAKE "${V4R_MODULE_DEFINITIONS_CONFIGMAKE}\n")

configure_file("${V4R_SOURCE_DIR}/cmake/templates/v4r_modules.h.in" "${V4R_CONFIG_FILE_INCLUDE_DIR}/v4r_modules.h")
install(FILES "${V4R_CONFIG_FILE_INCLUDE_DIR}/v4r_modules.h" DESTINATION ${V4R_INCLUDE_INSTALL_PATH} COMPONENT dev)
