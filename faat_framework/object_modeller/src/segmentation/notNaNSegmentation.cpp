
#include "segmentation/notNaNSegmentation.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

#include <pcl/features/integral_image_normal.h>

#include <faat_pcl/utils/registration_utils.h>

namespace object_modeller
{
namespace segmentation
{

void NotNaNSegmentation::applyConfig(Config &config)
{

}

NotNaNSegmentation::NotNaNSegmentation(std::string config_name) : InOutModule(config_name)
{
}

std::vector<std::vector<int> > NotNaNSegmentation::process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds)
{
    std::vector<std::vector<int> > obj_indices_;

    for (size_t i = 0; i < pointClouds.size (); i++)
    {
        std::vector<int> indices;
        for(size_t k=0; k < pointClouds[i]->points.size(); k++)
        {
            const Eigen::Vector3f & p = pointClouds[i]->points[k].getVector3fMap();
            if(pcl_isnan(p[0]) || pcl_isnan(p[1]) || pcl_isnan(p[2]))
                continue;

            indices.push_back(static_cast<int>(k));
        }

        obj_indices_.push_back (indices);
    }

    return obj_indices_;
}

}
}
