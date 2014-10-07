#include "output/pointCloudWriter.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

namespace object_modeller
{
namespace output
{

template<class TPointType>
void PointCloudWriter<TPointType>::process(std::vector<typename pcl::PointCloud<TPointType>::Ptr> pointClouds)
{
    std::stringstream _outputPath;
    _outputPath << outputPath << "/";

    int _activeSequence = OutModule<std::vector<typename pcl::PointCloud<TPointType>::Ptr> >::activeSequence;
    int _nrInputSequences = OutModule<std::vector<typename pcl::PointCloud<TPointType>::Ptr> >::nrInputSequences;

    if (_nrInputSequences > 1)
    {
        _outputPath << "seq_" << _activeSequence << "/";
    }

    boost::filesystem::path dir(_outputPath.str());
    boost::filesystem::create_directories(dir);

    for(size_t k=0; k < pointClouds.size(); k++)
    {
        std::stringstream filename;
        filename << _outputPath.str();

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
