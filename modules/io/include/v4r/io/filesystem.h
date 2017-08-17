/******************************************************************************
 * Copyright (c) 2016 Thomas Faeulhammer
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/


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
