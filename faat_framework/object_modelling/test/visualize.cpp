/*
 * do_modelling.cpp
 *
 *  Created on: Mar 15, 2013
 *      Author: aitor
 */

#include <pcl/console/parse.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <faat_pcl/utils/filesystem_utils.h>

#include <pcl/features/organized_edge_detection.h>
#include <pcl/features/integral_image_normal.h>
#include "faat_pcl/object_modelling/seg_do_modelling.h"
#include <faat_pcl/registration/mv_lm_icp.h>
#include <pcl/features/normal_3d_omp.h>
#include <faat_pcl/utils/filesystem_utils.h>
//#include <faat_pcl/registration/feature_agnostic_icp_with_gc.h>
#include <faat_pcl/registration/icp_with_gc.h>
#include <faat_pcl/utils/noise_models.h>
#include <faat_pcl/registration/registration_utils.h>
#include <faat_pcl/utils/registration_utils.h>
#include <faat_pcl/utils/pcl_opencv.h>

struct IndexPoint
{
    int idx;
};

POINT_CLOUD_REGISTER_POINT_STRUCT (IndexPoint,
                                   (int, idx, idx)
                                   )

inline bool
readMatrixFromFile2 (std::string file, Eigen::Matrix4f & matrix, int ignore = 0)
{

    std::ifstream in;
    in.open (file.c_str (), std::ifstream::in);

    char linebuf[1024];
    in.getline (linebuf, 1024);
    std::string line (linebuf);
    std::vector<std::string> strs_2;
    boost::split (strs_2, line, boost::is_any_of (" "));

    int c = 0;
    for (int i = ignore; i < (ignore + 16); i++, c++)
    {
        matrix (c / 4, c % 4) = static_cast<float> (atof (strs_2[i].c_str ()));
    }

    return true;
}

inline bool
writeMatrixToFile (std::string file, Eigen::Matrix4f & matrix)
{
    std::ofstream out (file.c_str ());
    if (!out)
    {
        std::cout << "Cannot open file.\n";
        return false;
    }

    for (size_t i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            out << matrix (i, j);
            if (!(i == 3 && j == 3))
                out << " ";
        }
    }
    out.close ();

    return true;
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

int
main (int argc, char ** argv)
{
    float Z_DIST_ = 1.5f;
    std::string pcd_files_dir_;
    bool sort_pcd_files_ = true;
    bool use_max_cluster_ = true;
    float data_scale = 1.f;
    float x_limits = 0.4f;
    int num_plane_inliers = 500;
    bool single_object = true;
    std::string aligned_output_dir = "";
    float max_corresp_dist_ = 0.1f;
    float voxel_grid_size = 0.003f;
    float dt_size = voxel_grid_size;
    float visualize = false;
    bool fast_overlap = false;
    int ignore = 1;
    int max_vis = std::numeric_limits<int>::max();
    int iterations = 10;

    float canny_low = 100.f;
    float canny_high = 150.f;
    bool sobel = false;

    bool use_cg_ = true;
    bool vis_final_ = false;
    float views_overlap_ = 0.3f;
    bool mv_use_weights_ = true;
    bool vis_pairwise_ = false;

    aligned_output_dir = "test_modelling";

    pcl::console::parse_argument (argc, argv, "-use_cg", use_cg_);
    pcl::console::parse_argument (argc, argv, "-iterations", iterations);
    pcl::console::parse_argument (argc, argv, "-low", canny_low);
    pcl::console::parse_argument (argc, argv, "-high", canny_high);
    pcl::console::parse_argument (argc, argv, "-ignore", ignore);
    pcl::console::parse_argument (argc, argv, "-pcd_files_dir", pcd_files_dir_);
    pcl::console::parse_argument (argc, argv, "-sort_pcd_files", sort_pcd_files_);
    pcl::console::parse_argument (argc, argv, "-use_max_cluster", use_max_cluster_);
    pcl::console::parse_argument (argc, argv, "-data_scale", data_scale);
    pcl::console::parse_argument (argc, argv, "-x_limits", x_limits);
    pcl::console::parse_argument (argc, argv, "-Z_DIST", Z_DIST_);
    pcl::console::parse_argument (argc, argv, "-num_plane_inliers", num_plane_inliers);
    pcl::console::parse_argument (argc, argv, "-single_object", single_object);
    pcl::console::parse_argument (argc, argv, "-vis_final", vis_final_);
    pcl::console::parse_argument (argc, argv, "-aligned_output_dir", aligned_output_dir);

    pcl::console::parse_argument (argc, argv, "-max_corresp_dist", max_corresp_dist_);
    pcl::console::parse_argument (argc, argv, "-vx_size", voxel_grid_size);
    pcl::console::parse_argument (argc, argv, "-dt_size", dt_size);
    pcl::console::parse_argument (argc, argv, "-visualize", visualize);
    pcl::console::parse_argument (argc, argv, "-fast_overlap", fast_overlap);
    pcl::console::parse_argument (argc, argv, "-max_vis", max_vis);
    pcl::console::parse_argument (argc, argv, "-sobel", sobel);
    pcl::console::parse_argument (argc, argv, "-views_overlap", views_overlap_);
    pcl::console::parse_argument (argc, argv, "-mv_use_weights", mv_use_weights_);
    pcl::console::parse_argument (argc, argv, "-vis_pairwise_", vis_pairwise_);

    bf::path aligned_output = aligned_output_dir;
    if(!bf::exists(aligned_output))
    {
        bf::create_directory(aligned_output);
    }

    float w_t = 0.75f;
    float max_angle = 60.f;
    float lateral_sigma = 0.002f;
    bool organized_normals = true;
    int seg_type = 0;
    float min_dot = 0.98f;
    bool do_multiview = true;
    bool use_normals = true;

    pcl::console::parse_argument (argc, argv, "-w_t", w_t);
    pcl::console::parse_argument (argc, argv, "-max_angle", max_angle);
    pcl::console::parse_argument (argc, argv, "-lateral_sigma", lateral_sigma);
    pcl::console::parse_argument (argc, argv, "-organized_normals", organized_normals);
    pcl::console::parse_argument (argc, argv, "-min_dot", min_dot);
    pcl::console::parse_argument (argc, argv, "-seg_type", seg_type);
    pcl::console::parse_argument (argc, argv, "-do_multiview", do_multiview);
    pcl::console::parse_argument (argc, argv, "-use_normals", use_normals);

    int step = 1;
    int mv_iterations = 10;
    std::string out_file;

    pcl::console::parse_argument (argc, argv, "-mv_iterations", mv_iterations);
    pcl::console::parse_argument (argc, argv, "-step", step);
    pcl::console::parse_argument (argc, argv, "-out_file", out_file);

    std::vector<std::string> scene_files;
    std::vector<std::string> indices_files;
    std::vector<std::string> transformation_files;
    std::string pattern_scenes = ".*cloud_.*.pcd";
    std::string pattern_indices = ".*object_indices_.*.pcd";
    std::string transformations_pattern = ".*pose.*.txt";

    bf::path input_dir = pcd_files_dir_;
    faat_pcl::utils::getFilesInDirectory(input_dir, scene_files, pattern_scenes);
    faat_pcl::utils::getFilesInDirectory(input_dir, indices_files, pattern_indices);
    faat_pcl::utils::getFilesInDirectory(input_dir, transformation_files, transformations_pattern);

    std::cout << "Number of clouds:" << scene_files.size() << std::endl;
    std::cout << "Number of models:" << indices_files.size() << std::endl;
    std::cout << "Number of transformations:" << transformation_files.size() << std::endl;

    std::sort(scene_files.begin(), scene_files.end());
    std::sort(indices_files.begin(), indices_files.end());
    std::sort(transformation_files.begin(), transformation_files.end());

    typedef pcl::PointXYZRGB PointType;
    typedef pcl::PointXYZRGBNormal PointTypeNormal;
    typedef pcl::PointXYZRGBNormal PointTInternal;

    if (sort_pcd_files_)
        std::sort (scene_files.begin (), scene_files.end ());

    std::vector<pcl::PointCloud<PointType>::Ptr> range_images_;
    range_images_.resize (scene_files.size ());

    std::vector<pcl::PointCloud<PointType>::Ptr> clouds_aligned;
    clouds_aligned.resize (scene_files.size ());

    std::vector<pcl::PointCloud<pcl::Normal>::Ptr > normalclouds_aligned;
    normalclouds_aligned.resize (scene_files.size ());

    std::vector<pcl::PointIndices> object_indices_;
    object_indices_.resize(scene_files.size());

    std::vector<Eigen::Matrix4f> global_poses;
    global_poses.resize(scene_files.size());

    pcl::PointCloud<PointType>::Ptr filtered_big_cloud2 (new pcl::PointCloud<PointType>);

    for (size_t i = 0; i < scene_files.size (); i++)
    {

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene(new pcl::PointCloud<pcl::PointXYZRGB>);

        {
            std::stringstream load;
            load << pcd_files_dir_ << "/" << scene_files[i];
            pcl::io::loadPCDFile(load.str(), *scene);
            range_images_[i].reset(new pcl::PointCloud<PointType>);
            pcl::copyPointCloud(*scene, *range_images_[i]);
        }

        Eigen::Matrix4f trans;

        if(transformation_files.size() != 0)
        {
            std::stringstream load;
            load << pcd_files_dir_ << "/" << transformation_files[i];
            faat_pcl::utils::readMatrixFromFile(load.str(), trans);
            global_poses[i] = trans;
        }

        pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
        if(organized_normals)
        {
            std::cout << "Organized normals" << std::endl;
            pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
            ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
            ne.setMaxDepthChangeFactor (0.02f);
            ne.setNormalSmoothingSize (20.0f);
            ne.setBorderPolicy (pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal>::BORDER_POLICY_MIRROR);
            ne.setInputCloud (scene);
            ne.compute (*normal_cloud);
        }
        else
        {
            std::cout << "Not organized normals" << std::endl;
            pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
            ne.setInputCloud (scene);
            pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
            ne.setSearchMethod (tree);
            ne.setRadiusSearch (0.02);
            ne.compute (*normal_cloud);
        }

        faat_pcl::utils::noise_models::NguyenNoiseModel<pcl::PointXYZRGB> nm;
        nm.setInputCloud(scene);
        nm.setInputNormals(normal_cloud);
        nm.setLateralSigma(lateral_sigma);
        nm.setMaxAngle(max_angle);
        nm.setUseDepthEdges(true);
        nm.compute();
        std::vector<float> weights;
        nm.getWeights(weights);

        pcl::PointIndices obj_indices;

        int valid = 0;
        for(size_t k=0; k < weights.size(); k++)
        {
            if(weights[k] > w_t)
            {
                obj_indices.indices.push_back(k);
                valid++;
            }
        }

        if(transformation_files.size() != 0)
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_trans2(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_trans(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::copyPointCloud(*scene, obj_indices, *scene_trans2);
            pcl::transformPointCloud(*scene_trans2, *scene_trans, trans);

            *filtered_big_cloud2 += *scene_trans;
        }
        else
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_trans2(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::copyPointCloud(*scene, obj_indices, *scene_trans2);
            *filtered_big_cloud2 += *scene_trans2;
        }
    }

    pcl::visualization::PCLVisualizer vis("checking");

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (filtered_big_cloud2);
    vis.addPointCloud<pcl::PointXYZRGB> (filtered_big_cloud2, handler, "big_filtered_averaged");
    vis.spin();

    pcl::io::savePCDFileBinary(out_file, *filtered_big_cloud2);
}
