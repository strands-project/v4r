#pragma once

#include <v4r/core/macros.h>

#include <string>
#include <vector>

#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS

namespace bf = boost::filesystem;

namespace v4r
{
namespace io
{

/** Returns folder names in a folder </br>
      * @param dir
      * @return relative_paths
      */
V4R_EXPORTS std::vector<std::string>
getFoldersInDirectory (const boost::filesystem::path &dir);


/** Returns a the name of files in a folder </br>
        * '(.*)bmp'
        * @param dir
        * @param regex_pattern examples "(.*)bmp",  "(.*)$"
        * @param recursive (true if files in subfolders should be returned as well)
        * @return files in folder
        */
V4R_EXPORTS
std::vector<std::string>
getFilesInDirectory (const boost::filesystem::path &dir, const std::string & regex_pattern = std::string(""), bool recursive = true);


/** checks if a file exists
        * @param rFile
        * @return true if file exsits
        */
V4R_EXPORTS bool
existsFile (const boost::filesystem::path &rFile );

/** checks if a folder exists
        * @param rFolder
        * @return true if folder exsits
        */
V4R_EXPORTS bool
existsFolder (const boost::filesystem::path &dir );

/** checks if folder already exists and if not, creates one
          * @param folder_name
          */
V4R_EXPORTS void
createDirIfNotExist(const boost::filesystem::path &dir);

/** checks if the path for the filename already exists,
         * otherwise creates it
         * @param filename
         */
V4R_EXPORTS void
createDirForFileIfNotExist(const boost::filesystem::path &filename);


/** @brief copies a directory from source to destination
          * @param path of source directory
          * @param path of destination directory
          */
V4R_EXPORTS
void
copyDir(const bf::path& sourceDir, const bf::path& destinationDir);


/**
 * @brief removeDir remove a directory with all its contents (including subdirectories) from disk
 * @param path folder path
 */
V4R_EXPORTS
void
removeDir(const bf::path &path);

}

}
