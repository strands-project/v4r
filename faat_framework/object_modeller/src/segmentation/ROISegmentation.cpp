
#include "segmentation/ROISegmentation.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

#include <pcl/filters/crop_box.h>
#include <faat_pcl/utils/registration_utils.h>

namespace object_modeller
{
namespace segmentation
{

void ROISegmentation::applyConfig(Config &config)
{

}

ROISegmentation::ROISegmentation(std::string config_name) : InOutModule(config_name)
{
}

std::vector<std::vector<int> > ROISegmentation::process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds)
{
    /*std::vector<std::vector<int> > obj_indices_;

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

    return obj_indices_;*/
}

void ROISegmentation::processSingle(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
                                    std::vector<int> & indices)
{
    pcl::CropBox<pcl::PointXYZRGB> box_filter(true);
    box_filter.setInputCloud(cloud);
    box_filter.setTransform(transformation_);
    box_filter.setMin(min_);
    box_filter.setMax(max_);
    box_filter.setNegative(false);

    box_filter.filter(indices);

    pcl::PointIndices indices_removed;
    box_filter.getRemovedIndices(indices_removed);

    //std::cout << indices_removed.indices.size() << " " << indices.size() << std::endl;

    float bad_val = std::numeric_limits<float>::quiet_NaN();
    Eigen::Vector3f bad_point(bad_val, bad_val, bad_val);
    for(size_t k=0; k < indices_removed.indices.size(); k++)
    {
        cloud->points[indices_removed.indices[k]].getVector3fMap() = bad_point;
    }
}

}
}
