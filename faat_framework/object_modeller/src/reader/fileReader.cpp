
#include "reader/fileReader.h"

#include <pcl/io/pcd_io.h>

#include <faat_pcl/utils/filesystem_utils.h>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>

namespace object_modeller
{
namespace reader
{

template<class TPointType>
void FileReader<TPointType>::applyConfig(Config::Ptr config)
{
    ConfigItem::applyConfig(config);

    nrSequences = 1;
}


template<class TPointType>
std::vector<typename pcl::PointCloud<TPointType>::Ptr> FileReader<TPointType>::process()
{
    std::string sequenceFolder = getSequenceFolder();

    std::cout << "target folder " << sequenceFolder << std::endl;

    std::vector<std::string> files;

    bf::path input_path = sequenceFolder;

    faat_pcl::utils::getFilesInDirectory(input_path, files, pattern);

    std::cout << "Load pcd files from source dir: " << sequenceFolder << std::endl;

    for (size_t i = 0; i < files.size (); i++)
    {
        std::cout << "Load pcd file " << files[i] << std::endl;

        files[i].insert(0, sequenceFolder);
    }

    // sort files
    std::sort (files.begin (), files.end ());

    // load point clouds
    std::vector<typename pcl::PointCloud<TPointType>::Ptr> pointClouds;

    for (size_t i = 0; i < files.size (); i+=step)
    {
        typename pcl::PointCloud<TPointType>::Ptr pointCloud (new pcl::PointCloud<TPointType>);

        printf("Load PCD file: %s\n", files[i].c_str());

        pcl::io::loadPCDFile (files[i], *pointCloud);

        pointClouds.push_back(pointCloud);
    }

    return pointClouds;
}

template class FileReader<pcl::PointXYZRGB>;
template class FileReader<pcl::PointXYZRGBNormal>;

}
}
