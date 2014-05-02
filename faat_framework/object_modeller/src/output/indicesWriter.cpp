#include "output/indicesWriter.h"

#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

#include <faat_pcl/utils/registration_utils.h>


struct IndexPoint
{
    int idx;
};

POINT_CLOUD_REGISTER_POINT_STRUCT (IndexPoint,
                                   (int, idx, idx)
                                   )

namespace object_modeller
{
namespace output
{

void IndicesWriter::applyConfig(Config &config)
{
    this->outputPath = config.getString(getConfigName(), "outputPath", "./out");
    this->pattern = config.getString(getConfigName(), "pattern", "object_indices_*.pcd");
}

void IndicesWriter::process(std::vector<std::vector<int> > indices)
{
    boost::filesystem::path dir(this->outputPath);
    boost::filesystem::create_directory(dir);

    for(size_t k=0; k < indices.size(); k++)
    {
        std::vector<int> obj_indices_original = indices[k]; //registration_utils::maskToIndices(indices[k]);

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
            int wildcardIndex = pattern.find_last_of(".");

            filename << pattern.substr(0, wildcardIndex);
            filename << setw( 8 ) << setfill( '0' ) << static_cast<int>(k);
            filename << pattern.substr(wildcardIndex);
        }

        std::string scene_name;
        filename >> scene_name;
        std::cout << scene_name << std::endl;
        pcl::PointCloud<IndexPoint> obj_indices_cloud;
        obj_indices_cloud.width = obj_indices_original.size();
        obj_indices_cloud.height = 1;
        obj_indices_cloud.points.resize(obj_indices_cloud.width);
        for(size_t kk=0; kk < obj_indices_original.size(); kk++)
            obj_indices_cloud.points[kk].idx = obj_indices_original[kk];

        pcl::io::savePCDFileBinary(scene_name, obj_indices_cloud);
    }
}

}
}
