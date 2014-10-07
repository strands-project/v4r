
#include "reader/memory_friendly_fileReader.h"

#include <pcl/io/pcd_io.h>

#include <faat_pcl/utils/filesystem_utils.h>

namespace object_modeller
{
namespace reader
{

    void MemoryFriendlyFileReader::applyConfig(Config::Ptr config)
    {
        ConfigItem::applyConfig(config);

        ConfigItem::registerParameter("pattern", "Pattern", &pattern, std::string(".*cloud_.*.pcd"));
        ConfigItem::registerParameter("inputPath", "Input path", &inputPath, std::string("./"));
        ConfigItem::registerParameter("step", "Step", &step, 1);
        ConfigItem::registerParameter("max_files", "Maximum Files", &max_files_, std::numeric_limits<int>::infinity());
    }

    std::vector<std::string> MemoryFriendlyFileReader::process()
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

        files_.resize(files.size());
        int k=0;
        for (size_t i = 0; i < files.size (); i+=step)
        {
            files_[k] = files[i];
            k++;
        }

        files_.resize(std::min(k, max_files_));
        return files_;

        // load point clouds
        /*std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds;

        for (size_t i = 0; i < files.size (); i+=step)
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud (new pcl::PointCloud<pcl::PointXYZRGB>);

            printf("Load PCD file: %s\n", files[i].c_str());

            pcl::io::loadPCDFile (files[i], *pointCloud);

            pointClouds.push_back(pointCloud);
        }

        return pointClouds;*/
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr MemoryFriendlyFileReader::getCloud(int i)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::io::loadPCDFile (files_[i], *pointCloud);
        return pointCloud;
    }
}
}
