
#include <iostream>
#include <string>
#include <sstream>
#include <limits>
#include <stdio.h>
#include <opencv2/opencv.hpp>

#include <v4r/io/filesystem.h>
#include <v4r/common/noise_model_based_cloud_integration.h>
#include <v4r/common/noise_models.h>
#include <v4r/common/miscellaneous.h>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;

int main(int argc, const char * argv[]) {
    std::string test_dir;

    google::InitGoogleLogging(argv[0]);

    v4r::NMBasedCloudIntegration<pcl::PointXYZRGB>::Parameter nm_int_param;
    nm_int_param.final_resolution_ = 0.002f;
    nm_int_param.min_points_per_voxel_ = 1;
    nm_int_param.min_weight_ = 0.5f;
    nm_int_param.octree_resolution_ = 0.002f;
    nm_int_param.threshold_ss_ = 0.01f;

    v4r::noise_models::NguyenNoiseModel<pcl::PointXYZRGB>::Parameter nm_param;

    int normal_method = 2;

    po::options_description desc("Noise model based cloud integration\n======================================\n**Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("test_dir", po::value<std::string>(&test_dir)->required(), "directory containing point clouds")
            ("resolution,r", po::value<float>(&nm_int_param.octree_resolution_)->default_value(nm_int_param.octree_resolution_), "")
            ("min_points_per_voxel,n", po::value<int>(&nm_int_param.min_points_per_voxel_)->default_value(nm_int_param.min_points_per_voxel_), "")
            ("min_weight,w", po::value<float>(&nm_int_param.min_weight_)->default_value(nm_int_param.min_weight_), "")
            ("threshold,t", po::value<float>(&nm_int_param.threshold_ss_)->default_value(nm_int_param.threshold_ss_), "")
            ("final_resolution,f", po::value<float>(&nm_int_param.final_resolution_)->default_value(nm_int_param.final_resolution_), "")
            ("lateral_sigma", po::value<float>(&nm_param.lateral_sigma_)->default_value(nm_param.lateral_sigma_), "")
            ("max_angle", po::value<float>(&nm_param.max_angle_)->default_value(nm_param.max_angle_), "")
            ("use_depth_edges", po::value<bool>(&nm_param.use_depth_edges_)->default_value(nm_param.use_depth_edges_), "")
            ("dilate_iterations,i", po::value<int>(&nm_param.dilate_iterations_)->default_value(nm_param.dilate_iterations_), "")
            ("dilate_width", po::value<int>(&nm_param.dilate_width_)->default_value(nm_param.dilate_width_), "")
            ("normal_method", po::value<int>(&normal_method)->default_value(normal_method), "method used for normal computation")
            ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return false;
    }

    try
    {
        po::notify(vm);
    }
    catch(std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
        return false;
    }


    std::vector< std::string> sub_folder_names;
    if(!v4r::io::getFoldersInDirectory( test_dir, "", sub_folder_names) )
    {
        std::cerr << "No subfolders in directory " << test_dir << ". " << std::endl;
        sub_folder_names.push_back("");
    }

    std::sort(sub_folder_names.begin(), sub_folder_names.end());


    int vp1, vp2;
    pcl::visualization::PCLVisualizer vis("registered cloud");
    vis.createViewPort(0,0,0.5,1,vp1);
    vis.createViewPort(0.5,0,1,1,vp2);

    for (size_t sub_folder_id=0; sub_folder_id < sub_folder_names.size(); sub_folder_id++)
    {
        const std::string sequence_path = test_dir + "/" + sub_folder_names[ sub_folder_id ];

        std::vector< std::string > views;
        v4r::io::getFilesInDirectory(sequence_path, views, "", ".*.pcd", false);
        std::sort(views.begin(), views.end());


        pcl::PointCloud<pcl::PointXYZRGB>::Ptr big_cloud_unfiltered (new pcl::PointCloud<pcl::PointXYZRGB>);
        std::vector< pcl::PointCloud<pcl::PointXYZRGB>::Ptr > clouds (views.size());
        std::vector< pcl::PointCloud<pcl::Normal>::Ptr > normals (views.size());
        std::vector<std::vector<float> > weights (views.size());
        std::vector<std::vector<float> > sigmas (views.size());
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > camera_transforms (views.size());

        for (size_t v_id=0; v_id<views.size(); v_id++)
        {
            clouds[v_id].reset ( new pcl::PointCloud<pcl::PointXYZRGB>);
            normals[v_id].reset ( new pcl::PointCloud<pcl::Normal>);

            const std::string fn = sequence_path + "/" + views[ v_id ];
            pcl::io::loadPCDFile(fn, *clouds[v_id]);

            camera_transforms[v_id] = v4r::RotTrans2Mat4f(clouds[v_id]->sensor_orientation_, clouds[v_id]->sensor_origin_);

            // reset view point otherwise pcl visualization is potentially messed up
            Eigen::Vector4f zero_origin; zero_origin[0] = zero_origin[1] = zero_origin[2] = zero_origin[3] = 0.f;
            clouds[v_id]->sensor_orientation_ = Eigen::Quaternionf::Identity();
            clouds[v_id]->sensor_origin_ = zero_origin;

            v4r::computeNormals<pcl::PointXYZRGB>( clouds[v_id], normals[v_id], normal_method);

            v4r::noise_models::NguyenNoiseModel<pcl::PointXYZRGB> nm (nm_param);
            nm.setInputCloud(clouds[v_id]);
            nm.setInputNormals(normals[v_id]);
            nm.compute();
            nm.getWeights(weights[v_id]);
            sigmas[v_id] = nm.getSigmas();

            pcl::PointCloud<pcl::PointXYZRGB> cloud_aligned;
            pcl::transformPointCloud( *clouds[v_id], cloud_aligned, camera_transforms[v_id]);
            *big_cloud_unfiltered += cloud_aligned;
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr octree_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        v4r::NMBasedCloudIntegration<pcl::PointXYZRGB> nmIntegration (nm_int_param);
        nmIntegration.setInputClouds(clouds);
        nmIntegration.setWeights(weights);
        nmIntegration.setSigmas(sigmas);
        nmIntegration.setTransformations(camera_transforms);
        nmIntegration.setInputNormals(normals);
        nmIntegration.compute(octree_cloud);

        std::cout << "Size cloud unfiltered: " << big_cloud_unfiltered->points.size() << ", filtered: " << octree_cloud->points.size() << std::endl;

        vis.removeAllPointClouds(vp1);
        vis.removeAllPointClouds(vp2);
        vis.addPointCloud(big_cloud_unfiltered, "unfiltered_cloud", vp1);
        vis.addPointCloud(octree_cloud, "filtered_cloud", vp2);
        vis.spin();

    }
}
