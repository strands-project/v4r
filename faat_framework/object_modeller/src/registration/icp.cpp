#include "registration/icp.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp_nl.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>

#include "v4r/KeypointTools/invPose.hpp"

namespace object_modeller
{
namespace registration
{

void Icp::applyConfig(Config &config)
{
}

std::vector<Eigen::Matrix4f> Icp::process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> inputClouds)
{
    std::vector<Eigen::Matrix4f> poses;

    poses.push_back(Eigen::Matrix4f::Identity());

    Eigen::Matrix4f accum = Eigen::Matrix4f::Identity();

    for (int i=0;i<inputClouds.size() - 1;i++)
    {
        pcl::IterativeClosestPoint<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp;
        std::cout << "icp for " << i << " and " << (i+1) << std::endl;

        std::vector<int> indices;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr source (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr target (new pcl::PointCloud<pcl::PointXYZRGB>);

        pcl::removeNaNFromPointCloud(*inputClouds.at(i), *source, indices);
        pcl::removeNaNFromPointCloud(*inputClouds.at(i+1), *target, indices);

        // downsample
        pcl::VoxelGrid<pcl::PointXYZRGB> grid;
        grid.setLeafSize (0.05, 0.05, 0.05);
        grid.setInputCloud (source);
        grid.filter (*source);
        grid.setInputCloud (target);
        grid.filter (*target);

        // Compute surface normals and curvature
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr points_with_normals_source (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr points_with_normals_target (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

        pcl::NormalEstimation<pcl::PointXYZRGB, pcl::PointXYZRGBNormal> norm_est;
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
        norm_est.setSearchMethod (tree);
        norm_est.setKSearch (30);

        norm_est.setInputCloud (source);
        norm_est.compute (*points_with_normals_source);
        pcl::copyPointCloud(*source, *points_with_normals_source);

        norm_est.setInputCloud (target);
        norm_est.compute (*points_with_normals_target);
        pcl::copyPointCloud (*target, *points_with_normals_target);

        icp.setTransformationEpsilon(1e-6);
        icp.setMaxCorrespondenceDistance(0.1);
        icp.setInputSource(points_with_normals_source);
        icp.setInputTarget(points_with_normals_target);
        pcl::PointCloud<pcl::PointXYZRGBNormal> temp;

        icp.align(temp);

        Eigen::Matrix4f inv;
        kp::invPose(icp.getFinalTransformation(), inv);
        accum = inv * accum;
        std::cout << accum << std::endl;
        poses.push_back(accum);

        std::cout << icp.hasConverged() << " - " << icp.getFitnessScore() << std::endl;
    }

    return poses;
}

}
}
