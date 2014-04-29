#include "output/pointCloudWriter.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

namespace object_modeller
{
namespace output
{

void PointCloudWriter::applyConfig(Config &config)
{
    this->outputPath = config.getString("pointCloudWriter.outputPath", "./out/");;
}

void PointCloudWriter::process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds)
{
    boost::filesystem::path dir(this->outputPath);
    boost::filesystem::create_directory(dir);

    for(size_t k=0; k < pointClouds.size(); k++)
    {
        std::stringstream temp;
        temp << outputPath << "/cloud_";
        temp << setw( 8 ) << setfill( '0' ) << static_cast<int>(k) << ".pcd";
        std::string scene_name;
        temp >> scene_name;
        std::cout << scene_name << std::endl;
        pcl::io::savePCDFileBinary(scene_name, *pointClouds[k]);
    }
}

}
}
