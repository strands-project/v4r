/*
 * filesystem_utils.h
 *
 *  Created on: Mar 15, 2013
 *      Author: aitor
 */

#ifndef FILESYSTEM_UTILS_H_
#define FILESYSTEM_UTILS_H_

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <fstream>
#include <Eigen/Dense>

namespace bf = boost::filesystem;

namespace v4r
{
  class utils
  {

    public:
      /** Returns folder names in a folder </br>
      * @param dir
      * @param rel_path_so_far
      * @param relative_paths
      * @return number of folders in folder
      */
      static int
      getFoldersInDirectory (const std::string & dir,
                             const std::string & rel_path_so_far,
                             std::vector<std::string> & relative_paths);



    /** Returns a the name of files in a folder </br>
    * '(.*)bmp'
    * @param dir
    * @param relative_paths
    * @param rel_path_so_far
    * @param regex_pattern examples "(.*)bmp",  "(.*)$"
    * @param recursive (true if files in subfolders should be returned as well)
    * @return number of files in folder (-1 in case directory name is not valid)
    */
    static int
    getFilesInDirectory (   const std::string & dir,
                            std::vector<std::string> & relative_paths,
                            const std::string & rel_path_so_far = std::string(""),
                            const std::string & regex_pattern = std::string(""),
                            bool recursive = true);

    static bool
    writeMatrixToFile (const std::string &file, const Eigen::Matrix4f & matrix);

    static bool
    readMatrixFromFile (const std::string &file, Eigen::Matrix4f & matrix);

    static bool
    readMatrixFromFile (const std::string &file, Eigen::Matrix4f & matrix, int padding);

    static bool
    writeCentroidToFile (const std::string &file, const Eigen::Vector3f & centroid);

    static bool
    getCentroidFromFile (const std::string &file, Eigen::Vector3f & centroid);

    static bool
    writeFloatToFile (const std::string &file, const float value);

    static bool
    readFloatFromFile (const std::string &file, float& value);


    /** checks if a file exists
    * @param rFile
    * @return true if file exsits
    */
    static bool existsFile ( const std::string &rFile );

  };
}
#endif /* FILESYSTEM_UTILS_H_ */
