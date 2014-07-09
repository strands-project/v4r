
#include "segmentation/dominantPlaneExtraction.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

#include <pcl/features/integral_image_normal.h>

#include "faat_pcl/object_modelling/seg_do_modelling.h"

#include <faat_pcl/utils/registration_utils.h>

namespace object_modeller
{
namespace segmentation
{

void DominantPlaneExtraction::applyConfig(Config &config)
{

}

DominantPlaneExtraction::DominantPlaneExtraction(std::string config_name) : InOutModule(config_name)
{
    num_plane_inliers = 500;
    seg_type = 1;
    plane_threshold = 0.01f;
}

std::vector<std::vector<int> > DominantPlaneExtraction::process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds)
{
    std::vector<std::vector<int> > obj_indices_;

    std::vector<int> idx;

    for (size_t i = 0; i < pointClouds.size (); i++)
    {
        std::vector<pcl::PointIndices> indices;
        Eigen::Vector4f table_plane;
        doSegmentation<pcl::PointXYZRGB> (pointClouds[i], indices, table_plane, num_plane_inliers, seg_type, plane_threshold);

        std::cout << "Number of clusters found:" << indices.size () << std::endl;

        pcl::PointIndices max;
        for (size_t k = 0; k < indices.size (); k++)
        {
            if (max.indices.size () < indices[k].indices.size ())
                {
                    max = indices[k];
                }
        }

        obj_indices_.push_back (max.indices);
    }

    return obj_indices_;
}

}
}
