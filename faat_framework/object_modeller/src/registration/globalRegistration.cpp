#include "registration/globalRegistration.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d_omp.h>

#include <faat_pcl/registration/registration_utils.h>
#include <faat_pcl/registration/mv_lm_icp.h>

#include <faat_pcl/utils/noise_models.h>

namespace object_modeller
{
namespace registration
{

void GlobalRegistration::applyConfig(Config &config)
{

}

GlobalRegistration::GlobalRegistration(std::string config_name) : InOutModule(config_name)
{
    views_overlap_ = 0.3f;
    fast_overlap = false;
    mv_iterations = 5;
    min_dot = 0.98f;
    mv_use_weights_ = true;
    max_angle = 60.f;
    lateral_sigma = 0.002f;
    organized_normals = false;
    w_t = 0.75f;
    canny_low = 100.f;
    canny_high = 150.f;
    sobel = false;
    depth_edges = true;
}

std::vector<Eigen::Matrix4f> GlobalRegistration::process(boost::tuples::tuple<
                                                         std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>,
                                                         std::vector<pcl::PointCloud<pcl::Normal>::Ptr>,
                                                         std::vector<std::vector<float> > > input)
{
    std::vector<Eigen::Matrix4f> poses;

    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds = input.get<0>();
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> normals = input.get<1>();
    std::vector<std::vector<float> > weights = input.get<2>();

    for(size_t i=0; i < pointClouds.size(); i++)
    {
        std::cout << "Multiview point cloud size: " << pointClouds[i]->size() << std::endl;
        std::cout << "Multiview normals size: " << normals[i]->size() << std::endl;
        std::cout << "Multiview weights size: " << weights[i].size() << std::endl;
    }

    /*
    std::vector<std::vector<float> > weights_;
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> normal_clouds;

    for(size_t i=0; i < pointClouds.size(); i++)
    {
        std::cout << "calculate normals" << std::endl;

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

        std::cout << "calculate weights" << std::endl;

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

    //multiview refinement
    std::vector < std::vector<bool> > A;
    A.resize (pointClouds.size ());
    for (size_t i = 0; i < pointClouds.size (); i++)
        A[i].resize (pointClouds.size (), true);

    float ff=views_overlap_;
    faat_pcl::registration_utils::computeOverlapMatrix<pcl::PointXYZRGB> (pointClouds, A, 0.01f, fast_overlap, ff);

    for (size_t i = 0; i < pointClouds.size (); i++)
    {
        for (size_t j = 0; j < pointClouds.size (); j++)
            std::cout << A[i][j] << " ";

        std::cout << std::endl;
    }

    float max_corresp_dist_mv_ = 0.01f;
    float dt_size_mv_ = 0.002f;
    float inlier_threshold_mv = 0.002f;
    faat_pcl::registration::MVNonLinearICP<pcl::PointXYZRGB> icp_nl (dt_size_mv_);
    icp_nl.setInlierThreshold (inlier_threshold_mv);
    icp_nl.setMaxCorrespondenceDistance (max_corresp_dist_mv_);
    icp_nl.setMaxIterations(mv_iterations);
    icp_nl.setInputNormals(normals);
    icp_nl.setClouds (pointClouds);
    icp_nl.setVisIntermediate (false);
    icp_nl.setSparseSolver (true);
    icp_nl.setAdjacencyMatrix (A);
    icp_nl.setMinDot(min_dot);
    //if(mv_use_weights_)
    //    icp_nl.setPointsWeight(weights);
    icp_nl.compute ();

    icp_nl.getTransformation (poses);

    return poses;
}

}
}
