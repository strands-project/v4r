/*
 * filesystem_utils.h
 *
 *  Created on: Mar 15, 2013
 *      Author: aitor
 */

#ifndef FILESYSTEM_UTILS_H_
#define FILESYSTEM_UTILS_H_

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <pcl/common/common.h>
#include <fstream>
#include <boost/regex.hpp>

namespace bf = boost::filesystem;

namespace faat_pcl
{
  namespace utils
  {
    void
    getFoldersInDirectory (const bf::path & dir,
                           const std::string & rel_path_so_far,
                           std::vector<std::string> & relative_paths);

    void
    getFilesInDirectory (   const bf::path & dir,
                            std::vector<std::string> & relative_paths,
                            const std::string & rel_path_so_far = std::string(""),
                            const std::string & regex_pattern = std::string(""),
                            bool recursive = true);

    bool
    writeMatrixToFile (const std::string &file, const Eigen::Matrix4f & matrix);

    bool
    readMatrixFromFile (const std::string &file, Eigen::Matrix4f & matrix);

    bool
    readMatrixFromFile (const std::string &file, Eigen::Matrix4f & matrix, int padding);

    bool
    writeCentroidToFile (const std::string &file, const Eigen::Vector3f & centroid);

    bool
    getCentroidFromFile (const std::string &file, Eigen::Vector3f & centroid);

    bool
    writeFloatToFile (const std::string &file, const float value);

    bool
    readFloatFromFile (const std::string &file, float& value);

  }
}
#endif /* FILESYSTEM_UTILS_H_ */
