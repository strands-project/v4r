#define KP_NO_CERES_AVAILABLE // Question: What is this?

// standard
#include <cstdlib>
#include <map>
#include <iostream>
#include <fstream>
#include <vector>

// pcl
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/features/organized_edge_detection.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d_omp.h>

// v4r
#include "v4r/KeypointTools/invPose.hpp"
#include "v4r/KeypointCameraTrackerPCL/CameraTrackerRGBDPCL.hh"
#include <v4r/TomGinePCL/tgTomGineThreadPCL.h>

// faat
#include <faat_pcl/utils/filesystem_utils.h>
#include <faat_pcl/utils/registration_utils.h>
#include <faat_pcl/utils/pcl_opencv.h>
#include <faat_pcl/registration/registration_utils.h>
#include "seg_do_modelling.h"
#include <faat_pcl/utils/noise_models.h>
#include <faat_pcl/registration/registration_utils.h>
#include <faat_pcl/registration/mv_lm_icp.h>
#include <faat_pcl/utils/noise_model_based_cloud_integration.h>

// Command line arguments
std::string sourceDir;
std::string outDir;
float zDist = 10.0f;
float x_limits = 0.1f;
int step = 0;
int enableGlobalRegistration = 0;
int keyframesOnly;
int usePclRenderer;

// segmentation
int num_plane_inliers = 500;
int seg_type = 1;
float plane_threshold = 0.01f;

std::vector<std::vector<int> > obj_indices_;
std::vector<std::vector<bool> > obj_masks_;

// global registration

float views_overlap_ = 0.3f;
bool fast_overlap = false;
int mv_iterations = 5;
float min_dot = 0.98f;
bool mv_use_weights_ = true;
float max_angle = 60.f;
float lateral_sigma = 0.002f;
bool organized_normals = true;
float w_t = 0.75f;
float canny_low = 100.f;
float canny_high = 150.f;
bool sobel = false;

std::vector<pcl::PointCloud<pcl::Normal>::Ptr > clouds_normals_;
std::vector<pcl::PointCloud<pcl::Normal>::Ptr > normalclouds_aligned;
std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> filtered_clouds_threshold;
std::vector< std::vector<float> > original_weights;

// nm based cloud integration
float resolution = 0.001f;
int min_points_per_voxel = 0;
float final_resolution = resolution;
bool depth_edges = true;

// general data

boost::shared_ptr<TomGine::tgTomGineThreadPCL> win;
boost::shared_ptr<pcl::visualization::PCLVisualizer> vis;

// original unmodified point clouds from files
std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> range_images_;

// 1. Original unmodified point clouds from files
// 2. filtered far points (-z argument)
std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds;

// aligned point clouds with poses from camera tracker
std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds_aligned;

// aligned point clouds with poses from camera tracker
std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> segmented_aligned;

// poses for each point cloud
std::vector<Eigen::Matrix4f> poses;

// value indicating whether a point cloud is a keyframe (and needed for further processing)
std::vector<bool> is_keyframe;


// helper methods
void parseCommandLineArgs(int argc, char **argv);
std::vector<std::string> loadInputFiles();

void renderPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud, std::string text = "");
void renderPointCloud(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr > pointClouds, std::string text = "");

void renderPointCloudTomGine(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud, std::string text = "");
void renderPointCloudTomGine(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr > pointClouds, std::string text = "");

void renderPointCloudPcl(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud, std::string text = "");
void renderPointCloudPcl(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr > pointClouds, std::string text = "");

void keypointBasedCameraTracking();
void segmentation();
void globalRegistration();
void nmBasedCloudIntegration();


void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event)
{
    if (event.getKeySym () == "space" && event.keyDown ())
    {
        vis->spinOnce();
    }
}


/******************************************************************
 * MAIN
 */
int main(int argc, char *argv[] )
{
    parseCommandLineArgs(argc, argv);

    if (usePclRenderer)
    {
        vis.reset(new pcl::visualization::PCLVisualizer(""));
        vis->registerKeyboardCallback (keyboardEventOccurred);
    }
    else
    {
        win.reset(new TomGine::tgTomGineThreadPCL(800,600));
    }

    std::vector<std::string> model_files = loadInputFiles();

    printf("sourceDir: %s\n", sourceDir.c_str());
    printf("outDir: %s\n", outDir.c_str());


    for (size_t i = 0; i < model_files.size (); i++)
    {
        printf("Input file: %s\n", model_files[i].c_str());
    }


    // load point clouds from files
    for (size_t i = 0; i < model_files.size (); i++)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr rimage (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene (new pcl::PointCloud<pcl::PointXYZRGB>);
        std::stringstream file_to_read;

        file_to_read << sourceDir << "/" << model_files[i];

        printf("Load PCD file: %s\n", file_to_read.str().c_str());

        pcl::io::loadPCDFile (file_to_read.str (), *scene);
        pcl::copyPointCloud(*scene, *rimage);

        range_images_.push_back(rimage);

        // filter far points
        pcl::PassThrough<pcl::PointXYZRGB> pass;
        pass.setFilterLimits (0.f, zDist);
        pass.setFilterFieldName ("z");
        pass.setInputCloud (scene);
        pass.setKeepOrganized (true);
        pass.filter (*scene);

        clouds.push_back(scene);

        renderPointCloud(scene, file_to_read.str());
    }

    keypointBasedCameraTracking();

    segmentation();

    if (enableGlobalRegistration)
    {
        globalRegistration();
    }

    nmBasedCloudIntegration();

    if (!step)
    {
        win->AddLabel2D("Press SPACE to continue", 10, 10, 10);
        win->WaitForEvent(TomGine::TMGL_Press, TomGine::TMGL_Space);
    }

    return 0;
}


void transformNormals(pcl::PointCloud<pcl::Normal>::Ptr & normals_cloud,
                      pcl::PointCloud<pcl::Normal>::Ptr & normals_aligned,
                      Eigen::Matrix4f & transform)
{
    normals_aligned.reset (new pcl::PointCloud<pcl::Normal>);
    normals_aligned->points.resize (normals_cloud->points.size ());
    normals_aligned->width = normals_cloud->width;
    normals_aligned->height = normals_cloud->height;
    for (size_t k = 0; k < normals_cloud->points.size (); k++)
    {
        Eigen::Vector3f nt (normals_cloud->points[k].normal_x, normals_cloud->points[k].normal_y, normals_cloud->points[k].normal_z);
        normals_aligned->points[k].normal_x = static_cast<float> (transform (0, 0) * nt[0] + transform (0, 1) * nt[1]
                                                                  + transform (0, 2) * nt[2]);
        normals_aligned->points[k].normal_y = static_cast<float> (transform (1, 0) * nt[0] + transform (1, 1) * nt[1]
                                                                  + transform (1, 2) * nt[2]);
        normals_aligned->points[k].normal_z = static_cast<float> (transform (2, 0) * nt[0] + transform (2, 1) * nt[1]
                                                                  + transform (2, 2) * nt[2]);
    }
}

template<class PointT>
inline void
computeRGBEdges (typename pcl::PointCloud<PointT>::Ptr & cloud, std::vector<int> & indices,
                 float low = 100.f, float high = 150.f, bool sobel = false)
{
    if(sobel)
    {
        cv::Mat_ < cv::Vec3b > colorImage;
        PCLOpenCV::ConvertPCLCloud2Image<pcl::PointXYZRGB> (cloud, colorImage);

        //cv::namedWindow("test");
        //cv::imshow("test", colorImage);

        cv::Mat grad_x, grad_y;
        cv::Mat abs_grad_x, abs_grad_y;
        int scale = 1;
        int delta = 0;
        int ddepth = CV_16S;
        int kernel_size = 3;

        /// Gradient X
        //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
        cv::Mat src_gray;
        cv::cvtColor( colorImage, src_gray, CV_RGB2GRAY );
        cv::Sobel( src_gray, grad_x, ddepth, 1, 0, kernel_size, scale, delta, cv::BORDER_DEFAULT );
        cv::convertScaleAbs( grad_x, abs_grad_x );

        /// Gradient Y
        //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
        cv::Sobel( src_gray, grad_y, ddepth, 0, 1, kernel_size, scale, delta, cv::BORDER_DEFAULT );
        cv::convertScaleAbs( grad_y, abs_grad_y );

        /// Total Gradient (approximate)
        cv::Mat grad;
        cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
        //    cv::namedWindow("sobel");
        //    cv::imshow("sobel", grad);

        cv::Mat thresholded_gradients = src_gray;
        for ( int j=0; j<abs_grad_x.rows; j++)
        {
            for (int i=0; i<abs_grad_x.cols; i++)
            {
                thresholded_gradients.at<unsigned char>(j,i) = 0;

                /*if(abs_grad_x.at<unsigned char>(j,i) > low || abs_grad_y.at<unsigned char>(j,i) > low)
          {
            thresholded_gradients.at<unsigned char>(j,i) = 255;
          }*/

                if(grad.at<unsigned char>(j,i) > low)
                {
                    thresholded_gradients.at<unsigned char>(j,i) = 255;
                }
            }
        }

        //cv::erode(thresholded_gradients, thresholded_gradients, cv::Mat());
        //cv::dilate(thresholded_gradients, thresholded_gradients, cv::Mat());

        //    cv::namedWindow("sobel_thres");
        //    cv::imshow("sobel_thres", thresholded_gradients);
        //    cv::waitKey(0);

        for ( int j=0; j<abs_grad_x.rows; j++)
        {
            for (int i=0; i<abs_grad_x.cols; i++)
            {
                if(thresholded_gradients.at<unsigned char>(j,i) == 255)
                {
                    indices.push_back(j*cloud->width + i);
                }
            }
        }
    }
    else
    {
        pcl::OrganizedEdgeFromRGB<PointT, pcl::Label> oed;
        oed.setRGBCannyLowThreshold (low);
        oed.setRGBCannyHighThreshold (high);
        oed.setEdgeType (pcl::OrganizedEdgeBase<PointT, pcl::Label>::EDGELABEL_RGB_CANNY);
        oed.setInputCloud (cloud);

        pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
        std::vector<pcl::PointIndices> indices2;
        oed.compute (*labels, indices2);

        for (size_t j = 0; j < indices2.size (); j++)
        {
            if (indices2[j].indices.size () > 0)
            {
                for(size_t k=0; k < indices2[j].indices.size (); k++)
                {
                    if(pcl_isfinite(cloud->points[indices2[j].indices[k]].z))
                        indices.push_back(indices2[j].indices[k]);
                }
            }
        }
    }
}

void nmBasedCloudIntegration()
{
    std::vector<std::vector<float> > weights_;
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> normal_clouds;

    for(size_t i=0; i < range_images_.size(); i++)
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
          ne.setInputCloud (range_images_[i]);
          ne.compute (*normal_cloud);
        }
        else
        {
          std::cout << "Not organized normals" << std::endl;
          pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
          ne.setInputCloud (range_images_[i]);
          pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
          ne.setSearchMethod (tree);
          ne.setRadiusSearch (0.02);
          ne.compute (*normal_cloud);
        }

        normal_clouds.push_back(normal_cloud);

        // calculate weights
        faat_pcl::utils::noise_models::NguyenNoiseModel<pcl::PointXYZRGB> nm;
        nm.setInputCloud(range_images_[i]);
        nm.setInputNormals(normal_cloud);
        nm.setLateralSigma(lateral_sigma);
        nm.setMaxAngle(max_angle);
        nm.setUseDepthEdges(depth_edges);
        nm.compute();
        std::vector<float> weights;
        nm.getWeights(weights);

        weights_.push_back(weights);
    }

    // do nm based cloud integration
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr octree_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    faat_pcl::utils::NMBasedCloudIntegration<pcl::PointXYZRGB> nmIntegration;
    nmIntegration.setInputClouds(range_images_);
    nmIntegration.setResolution(resolution);
    nmIntegration.setWeights(weights_);
    nmIntegration.setTransformations(poses);
    nmIntegration.setMinWeight(w_t);
    nmIntegration.setInputNormals(normal_clouds);
    nmIntegration.setMinPointsPerVoxel(min_points_per_voxel);
    nmIntegration.setFinalResolution(final_resolution);
    nmIntegration.setIndices(obj_indices_);
    nmIntegration.compute(octree_cloud);

    std::cout << "Octree size: " << octree_cloud->points.size() << std::endl;

    //pcl::io::savePCDFileASCII("bigcloud.pcd", *octree_cloud);

    renderPointCloud(octree_cloud, "NMBased Cloud integration");
}

void globalRegistration()
{
    /*
    std::vector<std::vector<float> > weights_;
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> normal_clouds;

    for(size_t i=0; i < segmented_aligned.size(); i++)
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
          ne.setInputCloud (segmented_aligned[i]);
          ne.compute (*normal_cloud);
        }
        else
        {
          std::cout << "Not organized normals" << std::endl;
          pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
          ne.setInputCloud (segmented_aligned[i]);
          pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
          ne.setSearchMethod (tree);
          ne.setRadiusSearch (0.02);
          ne.compute (*normal_cloud);
        }

        normal_clouds.push_back(normal_cloud);

        // calculate weights
        faat_pcl::utils::noise_models::NguyenNoiseModel<pcl::PointXYZRGB> nm;
        nm.setInputCloud(segmented_aligned[i]);
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
    A.resize (segmented_aligned.size ());
    for (size_t i = 0; i < segmented_aligned.size (); i++)
        A[i].resize (segmented_aligned.size (), true);

    float ff=views_overlap_;
    faat_pcl::registration_utils::computeOverlapMatrix<pcl::PointXYZRGB> (segmented_aligned, A, 0.01f, fast_overlap, ff);

    for (size_t i = 0; i < segmented_aligned.size (); i++)
    {
        for (size_t j = 0; j < segmented_aligned.size (); j++)
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
    //icp_nl.setInputNormals(normal_clouds);
    icp_nl.setClouds (segmented_aligned);
    icp_nl.setVisIntermediate (false);
    icp_nl.setSparseSolver (true);
    icp_nl.setAdjacencyMatrix (A);
    icp_nl.setMinDot(min_dot);
    //if(mv_use_weights_)
    //    icp_nl.setPointsWeight(weights_);
    icp_nl.compute ();

    std::vector<Eigen::Matrix4f> transformations;
    icp_nl.getTransformation (transformations);

    for (size_t i = 0; i < segmented_aligned.size (); i++)
    {
        poses[i] = transformations[i] * poses[i];
        pcl::transformPointCloud(*segmented_aligned[i], *segmented_aligned[i], transformations[i]);
    }

    //render
    renderPointCloud(segmented_aligned, "global registration");
}

//https://repo.acin.tuwien.ac.at/tmp/permanent/willow/willow_dataset_training_models_gt.zip
void keypointBasedCameraTracking()
{
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

    printf("Initialize camera tracker \n");
    kp::CameraTrackerRGBDPCL::Ptr camtracker( new kp::CameraTrackerRGBDPCL(param) );

    printf("Run camera tracker \n");
    camtracker->operate(clouds, poses, is_keyframe, false);
    printf("Camera Tracking complete\n");

    poses[0] = Eigen::Matrix4f::Identity();

    std::cout << "clouds size: " << clouds.size() << std::endl;
    std::cout << "poses size: " << poses.size() << std::endl;
    std::cout << "is_keyframe size: " << is_keyframe.size() << std::endl;

    // apply poses
    Eigen::Matrix4f inv_pose;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGB>());
    std::vector<int> idx;

    for (unsigned i=0; i<clouds.size(); i++)
    {
        transformed.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
        kp::invPose(poses[i], inv_pose);
        poses[i] = inv_pose;

        pcl::transformPointCloud(*clouds[i], *transformed, poses[i]);
        pcl::removeNaNFromPointCloud(*transformed,*transformed,idx);

        std::cout << poses[i] << std::endl;
        std::cout << is_keyframe[i] << std::endl;

        clouds_aligned.push_back(transformed);
    }

    if (keyframesOnly)
    {
        int size = clouds_aligned.size();

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

    renderPointCloud(clouds_aligned, "Aligned point clouds");
}

void segmentation()
{
    std::vector<int> idx;

    std::cout << "Nr of clouds for segmentation: " << clouds.size() << std::endl;

    for (size_t i = 0; i < clouds.size (); i++)
    {
        std::vector<pcl::PointIndices> indices;
        Eigen::Vector4f table_plane;
        doSegmentation<pcl::PointXYZRGB> (clouds[i], indices, table_plane, num_plane_inliers, seg_type, plane_threshold);

        std::cout << "Number of clusters found:" << indices.size () << std::endl;
        pcl::PointIndices max;
        for (size_t k = 0; k < indices.size (); k++)
        {
            if (max.indices.size () < indices[k].indices.size ())
                {
                    max = indices[k];
                }
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr obj_interest (new pcl::PointCloud<pcl::PointXYZRGB>(*clouds[i]));
        /*for (int j = 0; j < clouds[i]->points.size (); j++)
        {
            Eigen::Vector3f xyz_p = clouds[i]->points[j].getVector3fMap ();

            if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
                continue;

            float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

            if (val < -0.01)
            {
                obj_interest->points[j].x = std::numeric_limits<float>::quiet_NaN ();
                obj_interest->points[j].y = std::numeric_limits<float>::quiet_NaN ();
                obj_interest->points[j].z = std::numeric_limits<float>::quiet_NaN ();
            }
        }*/

        obj_indices_.push_back (max.indices);
        obj_masks_.push_back(registration_utils::indicesToMask(max.indices, obj_interest->points.size(), false));
        clouds[i] = obj_interest;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr masked_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::copyPointCloud(*clouds[i], max.indices, *masked_cloud);

        /*std::vector<int> obj_indices_original = registration_utils::maskToIndices(obj_masks_[i]);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr masked_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::copyPointCloud(*clouds[i], obj_indices_original, *masked_cloud);*/

        pcl::transformPointCloud(*masked_cloud, *masked_cloud, poses[i]);
        //pcl::removeNaNFromPointCloud(*masked_cloud,*masked_cloud,idx);

        segmented_aligned.push_back(masked_cloud);
    }

    renderPointCloud(segmented_aligned, "Segmented and aligned point clouds");
}

void renderPointCloud(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds, std::string text)
{
    if (usePclRenderer)
    {
        renderPointCloudPcl(pointClouds, text);
    }
    else
    {
        renderPointCloudTomGine(pointClouds, text);
    }
}

void renderPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud, std::string text)
{
    if (usePclRenderer)
    {
        renderPointCloudPcl(pointCloud, text);
    }
    else
    {
        renderPointCloudTomGine(pointCloud, text);
    }
}

void renderPointCloudTomGine(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds, std::string text)
{
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*pointClouds[0], centroid);
    win->SetRotationCenter(centroid[0], centroid[1], centroid[2]);

    win->Clear();

    for (int i=0;i<pointClouds.size();i++)
    {
        win->AddPointCloudPCL(*pointClouds[i]);
    }

    win->AddLabel2D(text, 10, 10, 580);

    win->Update();

    if (step)
    {
        win->AddLabel2D("Press SPACE to continue", 10, 10, 10);
        win->WaitForEvent(TomGine::TMGL_Press, TomGine::TMGL_Space);
    }
}

void renderPointCloudTomGine(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud, std::string text)
{
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*pointCloud, centroid);
    win->SetRotationCenter(centroid[0], centroid[1], centroid[2]);
    TomGine::tgCamera cam = win->GetCamera();
    cam.LookAt(TomGine::vec3(centroid(0),centroid(1),centroid(2)));
    win->SetCamera(cam);

    win->Clear();
    win->AddPointCloudPCL(*pointCloud);
    win->AddLabel2D(text, 10, 10, 580);

    win->Update();

    if (step)
    {
        win->AddLabel2D("Press SPACE to continue", 10, 10, 10);
        win->WaitForEvent(TomGine::TMGL_Press, TomGine::TMGL_Space);
    }
}

void renderPointCloudPcl(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud, std::string text)
{
    int v;
    vis->createViewPort(0,0,1,1,v);
    vis->removeAllPointClouds();

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb (pointCloud);
    vis->addPointCloud<pcl::PointXYZRGB> (pointCloud, handler_rgb, text, v);

    if (step)
    {
        vis->spin();
    }
    else
    {
        vis->spinOnce();
    }
}

void renderPointCloudPcl(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds, std::string text)
{
    int v;
    vis->createViewPort(0,0,1,1,v);
    vis->removeAllPointClouds();

    for (int i=0;i<pointClouds.size();i++)
    {
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler_rgb (pointClouds[i]);
        std::stringstream name;
        name << text << i;
        vis->addPointCloud<pcl::PointXYZRGB> (pointClouds[i], handler_rgb, name.str(), v);
    }

    if (step)
    {
        vis->spin();
    }
    else
    {
        vis->spinOnce();
    }
}

/**
  * Load input files
  */
std::vector<std::string> loadInputFiles()
{
    std::vector<std::string> model_files;
    bf::path input_path = sourceDir;
    std::string pattern_models = ".*cloud_.*.pcd";
    std::string relative_path = "";
    faat_pcl::utils::getFilesInDirectory(input_path, model_files, pattern_models);

    // sort files
        std::sort (model_files.begin (), model_files.end ());

    return model_files;
}


/**
 * setup command line args
 */
void parseCommandLineArgs(int argc, char **argv)
{
    int c;
    while(1)
    {
        c = getopt(argc, argv, "d:o:z:hsgkp");

        if(c == -1)
            break;

        switch(c)
        {
            case 'd':
                sourceDir = optarg;
                break;
            case 'o':
                outDir = optarg;
                break;
            case 'z':
                zDist = std::atof(optarg);
                break;
            case 's':
                step = 1;
                break;
            case 'g':
                enableGlobalRegistration = 1;
                break;
            case 'k':
                keyframesOnly = 1;
                break;
            case 'p':
                usePclRenderer = 1;
                break;
            case 'h':
                printf("%s [-d sourcedir] [-o outdir] [-h]\n"
                "   -s source directory\n"
                "   -o output directory\n"
                "   -h help\n",  argv[0]);
                exit(1);

                break;
        }
    }
}
