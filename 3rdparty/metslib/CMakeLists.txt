SET(EXT_METSLIB 1 CACHE BOOL "Metaheuristics library" )
IF(EXT_METSLIB)

PROJECT(v4rmetslib)

SET(SOURCE_H
  abstract-search.hh
  local-search.hh
  mets.hh
  metslib_config.hh
  model.hh
  observer.hh
  simulated-annealing.hh
  tabu-search.hh
  termination-criteria.hh
)

#add_library(${PROJECT_NAME} SHARED ${SOURCE_H})
#target_link_libraries(${PROJECT_NAME})
#v4r_add_headers(${PROJECT_NAME} "${SOURCE_H}")
#_install_library(${PREFIX_LIB}metslib ${SOURCE_H} "/3rdparty/metslib/")

install(FILES ${SOURCE_H} DESTINATION "${V4R_3P_INCLUDE_INSTALL_PATH}/metslib")

if(NOT BUILD_SHARED_LIBS)
  v4r_install_target(${METS_LIBRARY} EXPORT V4RModules ARCHIVE DESTINATION ${V4R_3P_LIB_INSTALL_PATH} COMPONENT dev)
endif()

ENDIF(EXT_METSLIB)

