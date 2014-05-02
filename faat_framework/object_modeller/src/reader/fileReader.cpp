
#include "reader/fileReader.h"

#include <pcl/io/pcd_io.h>

#include <faat_pcl/utils/filesystem_utils.h>

namespace object_modeller
{
namespace reader
{

void FileReader::applyConfig(Config &config)
{
    this->pattern   = config.getString(getConfigName(), "pattern",   ".*cloud_.*.pcd");
    this->inputPath = config.getString(getConfigName(), "inputPath", "./");
    this->step      = config.getInt   (getConfigName(), "step",      1);
}

std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> FileReader::process()
{
    std::vector<std::string> files;

    bf::path input_path = inputPath;

    faat_pcl::utils::getFilesInDirectory(input_path, files, pattern);

    std::cout << "Load pcd files from source dir: " << inputPath << std::endl;

    for (size_t i = 0; i < files.size (); i++)
    {
        std::cout << "Load pcd file " << files[i] << std::endl;

        files[i].insert(0, inputPath);
    }

    // sort files
    std::sort (files.begin (), files.end ());

    // load point clouds
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds;

    for (size_t i = 0; i < files.size (); i+=step)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud (new pcl::PointCloud<pcl::PointXYZRGB>);

        printf("Load PCD file: %s\n", files[i].c_str());

        pcl::io::loadPCDFile (files[i], *pointCloud);

        pointClouds.push_back(pointCloud);
    }

    return pointClouds;
}

}
}
