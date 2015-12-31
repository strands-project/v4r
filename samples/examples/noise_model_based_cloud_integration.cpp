
#include <iostream>
#include <string>
#include <sstream>
#include <limits>
#include <stdio.h>
#include <opencv2/opencv.hpp>

#include <v4r/io/filesystem.h>
#include <v4r/registration/noise_model_based_cloud_integration.h>
#include <v4r/common/noise_models.h>
#include <v4r/common/miscellaneous.h>

#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;

int main(int argc, const char * argv[]) {
    std::string test_dir, out_dir;
    float chop_z = std::numeric_limits<float>::max();
    bool visualize;

    google::InitGoogleLogging(argv[0]);

    v4r::NMBasedCloudIntegration<pcl::PointXYZRGB>::Parameter nm_int_param;
    nm_int_param.min_points_per_voxel_ = 1;
    nm_int_param.octree_resolution_ = 0.002f;

    v4r::NguyenNoiseModel<pcl::PointXYZRGB>::Parameter nm_param;

    int normal_method = 2;

    po::options_description desc("Noise model based cloud integration\n======================================\n**Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("input_dir,i", po::value<std::string>(&test_dir)->required(), "directory containing point clouds")
            ("out_dir,o", po::value<std::string>(&out_dir), "output directory where the registered cloud will be stored. If not set, nothing will be written to distk")
            ("resolution,r", po::value<float>(&nm_int_param.octree_resolution_)->default_value(nm_int_param.octree_resolution_), "")
            ("min_points_per_voxel", po::value<int>(&nm_int_param.min_points_per_voxel_)->default_value(nm_int_param.min_points_per_voxel_), "")
            ("threshold_explained", po::value<float>(&nm_int_param.threshold_explained_)->default_value(nm_int_param.threshold_explained_), "")
            ("use_depth_edges", po::value<bool>(&nm_param.use_depth_edges_)->default_value(nm_param.use_depth_edges_), "")
            ("edge_radius", po::value<int>(&nm_param.edge_radius_)->default_value(nm_param.edge_radius_), "")
            ("normal_method,n", po::value<int>(&normal_method)->default_value(normal_method), "method used for normal computation")
            ("chop_z,z", po::value<float>(&chop_z)->default_value(chop_z), "cut of distance in m ")
            ("visualize,v", po::bool_switch(&visualize), "turn visualization on")
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


    std::vector< std::string> folder_names  = v4r::io::getFoldersInDirectory( test_dir);
    if( folder_names.empty() )
    {
        std::cerr << "No subfolders in directory " << test_dir << ". " << std::endl;
        folder_names.push_back("");
    }

    std::sort(folder_names.begin(), folder_names.end());


    int vp1, vp2;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> vis;
    if(visualize)
    {
        vis.reset( new pcl::visualization::PCLVisualizer("registered cloud") );
        vis->createViewPort(0,0,0.5,1,vp1);
        vis->createViewPort(0.5,0,1,1,vp2);
    }

    for (size_t sub_folder_id=0; sub_folder_id < folder_names.size(); sub_folder_id++)
    {
        std::vector< std::string > views = v4r::io::getFilesInDirectory(folder_names[ sub_folder_id ], ".*.pcd", false);
        std::sort(views.begin(), views.end());

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr big_cloud_unfiltered (new pcl::PointCloud<pcl::PointXYZRGB>);
        std::vector< pcl::PointCloud<pcl::PointXYZRGB>::Ptr > clouds (views.size());
        std::vector< pcl::PointCloud<pcl::Normal>::Ptr > normals (views.size());
        std::vector<std::vector<std::vector<float> > > pt_properties (views.size());
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > camera_transforms (views.size());

        for (size_t v_id=0; v_id<views.size(); v_id++)
        {
            clouds[v_id].reset ( new pcl::PointCloud<pcl::PointXYZRGB>);
            normals[v_id].reset ( new pcl::PointCloud<pcl::Normal>);

            pcl::io::loadPCDFile(views[ v_id ], *clouds[v_id]);

            camera_transforms[v_id] = v4r::RotTrans2Mat4f(clouds[v_id]->sensor_orientation_, clouds[v_id]->sensor_origin_);

            // reset view point otherwise pcl visualization is potentially messed up
            Eigen::Vector4f zero_origin; zero_origin[0] = zero_origin[1] = zero_origin[2] = zero_origin[3] = 0.f;
            clouds[v_id]->sensor_orientation_ = Eigen::Quaternionf::Identity();
            clouds[v_id]->sensor_origin_ = zero_origin;

            pcl::PassThrough<pcl::PointXYZRGB> pass;
            pass.setInputCloud (clouds[v_id]);
            pass.setFilterFieldName ("z");
            pass.setFilterLimits (0.f, chop_z);
            pass.setKeepOrganized(true);
            pass.filter (*clouds[v_id]);

            v4r::computeNormals<pcl::PointXYZRGB>( clouds[v_id], normals[v_id], normal_method);

            v4r::NguyenNoiseModel<pcl::PointXYZRGB> nm (nm_param);
            nm.setInputCloud(clouds[v_id]);
            nm.setInputNormals(normals[v_id]);
            nm.compute();
            pt_properties[v_id] = nm.getPointProperties();

            pcl::PointCloud<pcl::PointXYZRGB> cloud_aligned;
            pcl::transformPointCloud( *clouds[v_id], cloud_aligned, camera_transforms[v_id]);
            *big_cloud_unfiltered += cloud_aligned;
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr octree_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        v4r::NMBasedCloudIntegration<pcl::PointXYZRGB> nmIntegration (nm_int_param);
        nmIntegration.setInputClouds(clouds);
        nmIntegration.setPointProperties(pt_properties);
        nmIntegration.setTransformations(camera_transforms);
        nmIntegration.setInputNormals(normals);
        nmIntegration.compute(octree_cloud);

        std::cout << "Size cloud unfiltered: " << big_cloud_unfiltered->points.size() << ", filtered: " << octree_cloud->points.size() << std::endl;

        if(visualize)
        {
            vis->removeAllPointClouds(vp1);
            vis->removeAllPointClouds(vp2);
            vis->addPointCloud(big_cloud_unfiltered, "unfiltered_cloud", vp1);
            vis->addPointCloud(octree_cloud, "filtered_cloud", vp2);
            vis->spin();
        }

        if(vm.count("out_dir"))
        {
            const std::string out_path = out_dir + "/" + folder_names[sub_folder_id];
            v4r::io::createDirIfNotExist(out_path);
            pcl::io::savePCDFileBinary(out_path + "/registered_cloud_filtered.pcd", *octree_cloud);
            pcl::io::savePCDFileBinary(out_path + "/registered_cloud_unfiltered.pcd", *big_cloud_unfiltered);
        }
    }
}
