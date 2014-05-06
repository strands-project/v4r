#include "output/pointCloudWriter.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

namespace object_modeller
{
namespace output
{

template<class TPointType>
void PointCloudWriter<TPointType>::applyConfig(Config &config)
{
    this->outputPath = config.getString(Module::getConfigName(), "outputPath",
                                        config.getString("writer", "outputPath", "./out"));
    this->pattern =     config.getString(Module::getConfigName(), "pattern", "cloud_*.pcd");
}

template<class TPointType>
void PointCloudWriter<TPointType>::process(std::vector<typename pcl::PointCloud<TPointType>::Ptr> pointClouds)
{
    boost::filesystem::path dir(this->outputPath);
    boost::filesystem::create_directories(dir);

    for(size_t k=0; k < pointClouds.size(); k++)
    {
        std::stringstream filename;
        filename << outputPath << "/";

        int wildcardIndex = pattern.find_first_of("*");

        if (wildcardIndex != -1)
        {
            filename << pattern.substr(0, wildcardIndex);
            filename << setw( 8 ) << setfill( '0' ) << static_cast<int>(k);
            filename << pattern.substr(wildcardIndex + 1);
        }
        else
        {
            if (pointClouds.size() > 1)
            {
                int wildcardIndex = pattern.find_last_of(".");

                filename << pattern.substr(0, wildcardIndex);
                filename << setw( 8 ) << setfill( '0' ) << static_cast<int>(k);
                filename << pattern.substr(wildcardIndex);
            }
            else
            {
                filename << pattern;
            }
        }

        std::string scene_name;
        filename >> scene_name;
        std::cout << scene_name << std::endl;
        pcl::io::savePCDFileBinary(scene_name, *pointClouds[k]);
    }
}

template class PointCloudWriter<pcl::PointXYZRGB>;
template class PointCloudWriter<pcl::PointXYZRGBNormal>;

}
}
