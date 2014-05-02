#include "modelling/nmBasedCloudIntegration.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d_omp.h>

#include <faat_pcl/utils/noise_models.h>
#include <faat_pcl/utils/noise_model_based_cloud_integration.h>

namespace object_modeller
{
namespace modelling
{

void NmBasedCloudIntegration::applyConfig(Config &config)
{
    resolution = config.getFloat(getConfigName(), "resolution",  0.001f);
    min_points_per_voxel = 0;
    final_resolution = resolution;
    depth_edges = true;
    organized_normals = true;
    max_angle = 60.f;
    lateral_sigma = 0.002f;
    w_t = 0.75f;
}

NmBasedCloudIntegration::NmBasedCloudIntegration(std::string config_name) : InOutModule(config_name)
{
}

std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> NmBasedCloudIntegration::process(boost::tuples::tuple<
             std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>,
             std::vector<Eigen::Matrix4f>,
             std::vector<std::vector<int> >,
             std::vector<pcl::PointCloud<pcl::Normal>::Ptr>,
             std::vector<std::vector<float> > > input)
{
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds = input.get<0>();
    std::vector<Eigen::Matrix4f> transformations = input.get<1>();
    std::vector<std::vector<int> > indices = input.get<2>();
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> normals = input.get<3>();
    std::vector<std::vector<float> > weights = input.get<4>();

    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> result;

    /*
    std::vector<std::vector<float> > weights_;
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> normal_clouds;

    for(size_t i=0; i < pointClouds.size(); i++)
    {
        // calculate normals
        pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
        if(organized_normals)
        {
          std::cout << "Organized normals" << std::endl;
          pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
          ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
          ne.setMaxDepthChangeFactor (0.02f);
          ne.setNormalSmoothingSize (20.0f);
          ne.setBorderPolicy (pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal>::BORDER_POLICY_MIRROR);
          ne.setInputCloud (pointClouds[i]);
          ne.compute (*normal_cloud);
        }
        else
        {
          std::cout << "Not organized normals" << std::endl;
          pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
          ne.setInputCloud (pointClouds[i]);
          pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
          ne.setSearchMethod (tree);
          ne.setRadiusSearch (0.02);
          ne.compute (*normal_cloud);
        }

        normal_clouds.push_back(normal_cloud);

        // calculate weights
        faat_pcl::utils::noise_models::NguyenNoiseModel<pcl::PointXYZRGB> nm;
        nm.setInputCloud(pointClouds[i]);
        nm.setInputNormals(normal_cloud);
        nm.setLateralSigma(lateral_sigma);
        nm.setMaxAngle(max_angle);
        nm.setUseDepthEdges(depth_edges);
        nm.compute();
        std::vector<float> weights;
        nm.getWeights(weights);

        weights_.push_back(weights);
    }
    */

    // do nm based cloud integration
    pcl::PointCloud<pcl::Normal>::Ptr out_normals(new pcl::PointCloud<pcl::Normal>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr octree(new pcl::PointCloud<pcl::PointXYZRGB>());
    faat_pcl::utils::NMBasedCloudIntegration<pcl::PointXYZRGB> nmIntegration;
    nmIntegration.setInputClouds(pointClouds);
    nmIntegration.setResolution(resolution);
    nmIntegration.setWeights(weights);
    nmIntegration.setTransformations(transformations);
    nmIntegration.setMinWeight(w_t);
    nmIntegration.setInputNormals(normals);
    nmIntegration.setMinPointsPerVoxel(min_points_per_voxel);
    nmIntegration.setFinalResolution(final_resolution);
    nmIntegration.setIndices(indices);
    nmIntegration.compute(octree);
    nmIntegration.getOutputNormals(out_normals);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr merged(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    pcl::concatenateFields(*out_normals, *octree, *merged);

    result.push_back(merged);

    return result;
}

}
}
