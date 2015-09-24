if(WITH_ASSIMP)
  find_path(ASSIMP_INCLUDE_PATH assimp/config.h
            PATHS /usr/local /opt /usr $ENV{ASSIMP_ROOT}
            DOC "The path to Assimp header files"
            CMAKE_FIND_ROOT_PATH_BOTH)
  find_library(ASSIMP_LIBRARIES
               NAMES assimp
               PATHS /usr/local /opt /usr $ENV{ASSIMP_ROOT}
               DOC "The Assimp library")
  if(ASSIMP_INCLUDE_PATH AND ASSIMP_LIBRARIES)
    set(ASSIMP_INCLUDE_DIRS "${ASSIMP_INCLUDE_PATH}")
    set(HAVE_ASSIMP YES)
    # Find version number in pkgconfig file
    find_file(ASSIMP_PKGCONFIG_FILE assimp.pc
              PATHS /usr/local /opt /usr $ENV{ASSIMP_ROOT}
              PATH_SUFFIXES lib/pkgconfig
              DOC "The Assimp pkfconfig file"
              CMAKE_FIND_ROOT_PATH_BOTH)
    if(ASSIMP_PKGCONFIG_FILE)
      file(STRINGS "${ASSIMP_PKGCONFIG_FILE}" _version REGEX "Version: [0-9.]+")
      if(_version)
        string(REGEX REPLACE "Version: " "" ASSIMP_VERSION ${_version})
      endif()
    endif()
  endif()
endif()
