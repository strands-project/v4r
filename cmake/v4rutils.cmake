
set(V4R_LIBRARIES CACHE INTERNAL "v4r library names" FORCE)
set(V4R_LIBRARIES_PC CACHE INTERNAL "v4r libraries names for pc" FORCE)

macro( v4r_add_headers LIBRARY_NAME SOURCE_HEADER )

  string(REGEX REPLACE "v4r" "" V4R_INCLUDE_NAME ${LIBRARY_NAME})

  #if(V4R_LIBRARIES STREQUAL "")
  #  set(V4R_LIBRARIES "${LIBRARY_NAME}" CACHE INTERNAL "v4r library names")
  #else()
  #  set(V4R_LIBRARIES "${V4R_LIBRARIES} ${LIBRARY_NAME}" CACHE INTERNAL "v4r library names")
  #endif()

  #if(V4R_LIBRARIES_PC STREQUAL "")
  #  set(V4R_LIBRARIES_PC "-l${LIBRARY_NAME}" CACHE INTERNAL "v4r library names")
  #else()
  #  set(V4R_LIBRARIES_PC "${V4R_LIBRARIES_PC} -l${LIBRARY_NAME}" CACHE INTERNAL "v4r library names")
  #endif()

  install(DIRECTORY DESTINATION include/v4r/${V4R_INCLUDE_NAME})
  install(FILES ${SOURCE_HEADER} DESTINATION include/v4r/${V4R_INCLUDE_NAME})
  #install(TARGETS ${LIBRARY_NAME} LIBRARY DESTINATION lib)

endmacro( v4r_add_headers )

macro( v4rexternal_add_headers LIBRARY_NAME SOURCE_HEADER )

  string(REGEX REPLACE "v4rexternal" "" V4REXTERNAL_INCLUDE_NAME ${LIBRARY_NAME})

  if(V4REXTERNAL_LIBRARIES STREQUAL "")
    set(V4REXTERNAL_LIBRARIES "${LIBRARY_NAME}" CACHE INTERNAL "v4rexternal library names")
  else()
    set(V4REXTERNAL_LIBRARIES "${V4REXTERNAL_LIBRARIES} ${LIBRARY_NAME}" CACHE INTERNAL "v4rexternal library names")
  endif()

  if(V4REXTERNAL_LIBRARIES_PC STREQUAL "")
    set(V4REXTERNAL_LIBRARIES_PC "-l${LIBRARY_NAME}" CACHE INTERNAL "v4rexternal library names")
  else()
    set(V4REXTERNAL_LIBRARIES_PC "${V4REXTERNAL_LIBRARIES_PC} -l${LIBRARY_NAME}" CACHE INTERNAL "v4rexternal library names")
  endif()

  install(DIRECTORY DESTINATION include/v4rexternal/${V4REXTERNAL_INCLUDE_NAME})
  install(FILES ${SOURCE_HEADER} DESTINATION include/v4rexternal/${V4REXTERNAL_INCLUDE_NAME})
  #install(TARGETS ${LIBRARY_NAME} LIBRARY DESTINATION lib)

endmacro( v4rexternal_add_headers )



macro( v4r_add_library LIBRARY_NAME SOURCE_HEADER ) 

	string(REGEX REPLACE "v4r" "" V4R_INCLUDE_NAME ${LIBRARY_NAME})
	
	set(V4R_LIBRARIES "${V4R_LIBRARIES} ${LIBRARY_NAME}" CACHE INTERNAL "v4r library names") 
	set(V4R_LIBRARIES_PC "${V4R_LIBRARIES_PC} -l${LIBRARY_NAME}" CACHE INTERNAL "v4r library names") 
	
	install(DIRECTORY DESTINATION include/v4r/${V4R_INCLUDE_NAME})
	install(FILES ${SOURCE_HEADER} DESTINATION include/v4r/${V4R_INCLUDE_NAME})
	install(TARGETS ${LIBRARY_NAME} LIBRARY DESTINATION lib)

endmacro( v4r_add_library ) 

macro( v4rexternal_add_library LIBRARY_NAME SOURCE_HEADER )

  string(REGEX REPLACE "3rdparty" "" V4R3rdparty_INCLUDE_NAME ${LIBRARY_NAME})

  if(V4REXTERNAL_LIBRARIES STREQUAL "")
    set(V4REXTERNAL_LIBRARIES "${LIBRARY_NAME}" CACHE INTERNAL "3rdparty library names")
  else()
    set(V4REXTERNAL_LIBRARIES "${V4REXTERNAL_LIBRARIES} ${LIBRARY_NAME}" CACHE INTERNAL "3rdparty library names")
  endif()

  if(V4REXTERNAL_LIBRARIES_PC STREQUAL "")
    set(V4REXTERNAL_LIBRARIES_PC "-l${LIBRARY_NAME}" CACHE INTERNAL "3rdparty library names")
  else()
    set(V4REXTERNAL_LIBRARIES_PC "${V4REXTERNAL_LIBRARIES_PC} -l${LIBRARY_NAME}" CACHE INTERNAL "3rdparty library names")
  endif()

  install(DIRECTORY DESTINATION include/3rdparty/${V4REXTERNAL_INCLUDE_NAME})
  install(FILES ${SOURCE_HEADER} DESTINATION include/3rdparty/${V4REXTERNAL_INCLUDE_NAME} COMPONENT pcl_${_component})
  install(TARGETS ${LIBRARY_NAME} LIBRARY DESTINATION lib)

endmacro( v4rexternal_add_library )


macro( v4r_add_binary BINARY_NAME )

	install(TARGETS ${BINARY_NAME} RUNTIME DESTINATION bin)

endmacro( v4r_add_binary ) 

