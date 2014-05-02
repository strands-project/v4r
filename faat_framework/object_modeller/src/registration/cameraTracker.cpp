#include "registration/cameraTracker.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/radius_outlier_removal.h>

#include "v4r/KeypointTools/invPose.hpp"

namespace object_modeller
{
namespace registration
{

void CameraTracker::applyConfig(Config &config)
{
    this->keyframesOnly = config.getBool(getConfigName(), "keyframesOnly");


    float rt_inl_dist = 0.005;    // rigid transformation RANSAC inlier dist[m]

    // setup parameters
    kp::KeypointTracker::Parameter kt_param;      //---------- KeypointTracker parameter
    kt_param.tiles = 1;                   // 3
    kt_param.max_features_tile = 1000;    // 100
    kt_param.inl_dist = 7.;               // inlier dist for outlier rejection [px]
    kt_param.pecnt_prefilter = 0.02;
    kt_param.refineLK = true;             // refine keypoint location (LK)
    kt_param.refineMappedLK = false;      // refine keypoint location (map image, then LK)
    kt_param.min_total_matches = 10;      // minimum total number of mathches
    kt_param.min_tiles_used = 3;          // minimum number of tiles with matches
    kt_param.affine_outl_rejection = true;
    kt_param.fast_pyr_levels = 2;
    kp::RigidTransformationRANSAC::Parameter rt_param; //----- RigidTransformationRANSAC parameter
    rt_param.inl_dist = rt_inl_dist;       // inlier dist RT-RANSAC [m]
    rt_param.eta_ransac = 0.01;
    rt_param.max_rand_trials = 10000;
    kp::LoopClosingRT::Parameter lc_param;     //------------- LoopClosingRT parameter
    lc_param.max_angle = 30;
    lc_param.min_total_score = 10.;
    lc_param.min_tiles_used = 3;
    lc_param.rt_param = rt_param;

    kp::CameraTrackerRGBD::Parameter param;
    param.min_total_score = 10;         // minimum score (matches weighted with inv. inl. dist)
    param.min_tiles_used = 3;           // minimum number of tiles with matches
    param.angle_init_keyframe = 7.5;
    param.detect_loops = true;
    param.thr_image_motion = 0.25;      // new keyframe if the image mot. > 0.25*im. width
    param.log_clouds = true;

    param.kt_param = kt_param;
    param.rt_param = rt_param;
    param.lc_param = lc_param;

    camtracker.reset( new kp::CameraTrackerRGBDPCL(param) );
}

std::vector<Eigen::Matrix4f> CameraTracker::process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> inputClouds)
{
    std::cout << "start camera tracker process" << std::endl;

    std::vector<Eigen::Matrix4f> poses;

    std::vector<bool> is_keyframe;

    std::cout << "Start camera tracking" << std::endl;
    std::cout << "input cloud size: " << inputClouds.size() << std::endl;
    camtracker->operate(inputClouds, poses, is_keyframe, false);

    std::cout << "Camera tracking finished" << std::endl;

    poses[0] = Eigen::Matrix4f::Identity();

    // apply poses
    Eigen::Matrix4f inv_pose;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGB>());
    std::vector<int> idx;

    for (unsigned i=0; i<inputClouds.size(); i++)
    {
        kp::invPose(poses[i], inv_pose);
        poses[i] = inv_pose;
    }

    for (unsigned i=0;i<poses.size();i++)
    {
        std::cout << "pose " << i << std::endl;
        std::cout << poses[i] << std::endl;
    }

    std::cout << "Camera tracking results complete" << std::endl;

    /*
    if (keyframesOnly)
    {
        int size = result->size();

        for (int i=0;i<size;i++)
        {
            if (!is_keyframe[i])
            {
                clouds_aligned.erase(clouds_aligned.begin() + i);
                clouds.erase(clouds.begin() + i);
                poses.erase(poses.begin() + i);
                range_images_.erase(range_images_.begin() + i);
            }
        }
    }
    */

    return poses;
}

}
}
