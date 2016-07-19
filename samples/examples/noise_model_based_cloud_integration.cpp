#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <limits>
#include <stdio.h>
#include <opencv2/opencv.hpp>

#include <v4r/io/eigen.h>
#include <v4r/io/filesystem.h>
#include <v4r/registration/noise_model_based_cloud_integration.h>
#include <v4r/common/noise_models.h>
#include <v4r/common/normals.h>
#include <v4r/common/pcl_opencv.h>

#include <pcl/common/transforms.h>
#include <pcl/common/time.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;
using namespace v4r;

int main(int argc, const char * argv[]) {
    typedef pcl::PointXYZRGB PointT;
    std::string test_dir, out_dir, view_prefix = "cloud_", obj_indices_prefix = "object_indices_", pose_prefix = "pose_";
    float chop_z = std::numeric_limits<float>::max();
    bool visualize=false, use_object_mask = false, use_pose_file = false, debug = false;

    google::InitGoogleLogging(argv[0]);

    NMBasedCloudIntegration<PointT>::Parameter nm_int_param;
    nm_int_param.min_points_per_voxel_ = 1;
    nm_int_param.octree_resolution_ = 0.002f;

    NguyenNoiseModel<PointT>::Parameter nm_param;

    int normal_method = 2;

    po::options_description desc("Noise model based cloud integration\n======================================\n**Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("input_dir,i", po::value<std::string>(&test_dir)->required(), "directory containing point clouds")
            ("out_dir,o", po::value<std::string>(&out_dir), "output directory where the registered cloud will be stored. If not set, nothing will be written to distk")
            ("view_prefix", po::value<std::string>(&view_prefix)->default_value(view_prefix), "view filename prefix for each point cloud (used when using object mask)")
            ("obj_indices_prefix", po::value<std::string>(&obj_indices_prefix)->default_value(obj_indices_prefix), "filename prefix for each object mask file(used when using object mask)")
            ("pose_prefix", po::value<std::string>(&pose_prefix)->default_value(pose_prefix), "filename prefix for each camera pose (used when using use_pose_file)")
            ("resolution,r", po::value<float>(&nm_int_param.octree_resolution_)->default_value(nm_int_param.octree_resolution_), "")
            ("min_points_per_voxel", po::value<int>(&nm_int_param.min_points_per_voxel_)->default_value(nm_int_param.min_points_per_voxel_), "")
//            ("threshold_explained", po::value<float>(&nm_int_param.threshold_explained_)->default_value(nm_int_param.threshold_explained_), "")
            ("use_depth_edges", po::value<bool>(&nm_param.use_depth_edges_)->default_value(nm_param.use_depth_edges_), "")
            ("edge_radius", po::value<int>(&nm_param.edge_radius_)->default_value(nm_param.edge_radius_), "")
            ("normal_method,n", po::value<int>(&normal_method)->default_value(normal_method), "method used for normal computation")
            ("chop_z,z", po::value<float>(&chop_z)->default_value(chop_z), "cut of distance in m ")
            ("visualize,v", po::bool_switch(&visualize), "turn visualization on")
            ("use_object_mask,m", po::bool_switch(&use_object_mask), "reads mask file and only extracts those indices (only if file exists)")
            ("use_pose_file,p", po::bool_switch(&use_pose_file), "reads pose from seperate pose file instead of extracting it directly from .pcd file (only if file exists)")
            ("debug,d", po::bool_switch(&debug), "saves debug information (e.g. point properties) if output dir is set")
      ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return false;
    }

    try
    { po::notify(vm); }
    catch(std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
        return false;
    }

    nm_int_param.edge_radius_px_ = nm_param.edge_radius_;
    std::vector< std::string> folder_names  = io::getFoldersInDirectory( test_dir );

    if( folder_names.empty() )
        folder_names.push_back("");

    int vp1, vp2;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> vis;

    for (const std::string &sub_folder : folder_names)
    {
        const std::string test_seq = test_dir + "/" + sub_folder;
        std::vector< std::string > views = io::getFilesInDirectory(test_seq, ".*.pcd", false);

        pcl::PointCloud<PointT>::Ptr big_cloud_unfiltered (new pcl::PointCloud<PointT>);
        std::vector< pcl::PointCloud<PointT>::Ptr > clouds (views.size());
        std::vector< pcl::PointCloud<pcl::Normal>::Ptr > normals (views.size());
        std::vector<std::vector<std::vector<float> > > pt_properties (views.size());
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > camera_transforms (views.size());
        std::vector<std::vector<int> > obj_indices (views.size());

        for (size_t v_id=0; v_id<views.size(); v_id++)
        {
            std::stringstream txt;
            txt << "processing view " << v_id;
            pcl::ScopeTime t(txt.str().c_str());
            clouds[v_id].reset ( new pcl::PointCloud<PointT>);
            normals[v_id].reset ( new pcl::PointCloud<pcl::Normal>);

            pcl::io::loadPCDFile(test_seq + "/" + views[ v_id ], *clouds[v_id]);

            std::string obj_fn (views[v_id]);
            boost::replace_first (obj_fn, view_prefix, obj_indices_prefix);
            boost::replace_last (obj_fn, ".pcd", ".txt");

            if(io::existsFile(test_seq + "/" + obj_fn) && use_object_mask) {
                ifstream f((test_seq+"/"+obj_fn).c_str());
                int idx;
                while (f >> idx)
                    obj_indices[v_id].push_back(idx);
                f.close();
            }

            std::string pose_fn (views[v_id]);
            boost::replace_first (pose_fn, view_prefix, pose_prefix);
            boost::replace_last (pose_fn, ".pcd", ".txt");

            if(io::existsFile(test_seq + "/" + pose_fn) && use_pose_file) {
                camera_transforms[v_id] = io::readMatrixFromFile(test_seq + "/" + pose_fn);
            }
            else
                camera_transforms[v_id] = RotTrans2Mat4f(clouds[v_id]->sensor_orientation_, clouds[v_id]->sensor_origin_);

            // reset view point otherwise pcl visualization is potentially messed up
            clouds[v_id]->sensor_orientation_ = Eigen::Quaternionf::Identity();
            clouds[v_id]->sensor_origin_ = Eigen::Vector4f::Zero();

            pcl::PassThrough<PointT> pass;
            pass.setInputCloud (clouds[v_id]);
            pass.setFilterFieldName ("z");
            pass.setFilterLimits (0.f, chop_z);
            pass.setKeepOrganized(true);
            pass.filter (*clouds[v_id]);

            {
                pcl::ScopeTime tt("Computing normals");
                computeNormals<PointT>( clouds[v_id], normals[v_id], normal_method);
            }

            {
                pcl::ScopeTime tt("Computing noise model parameter for cloud");
                NguyenNoiseModel<PointT> nm (nm_param);
                nm.setInputCloud(clouds[v_id]);
                nm.setInputNormals(normals[v_id]);
                nm.compute();
                pt_properties[v_id] = nm.getPointProperties();

            }



            pcl::PointCloud<PointT> object_cloud, object_aligned;
            pcl::copyPointCloud(*clouds[v_id], obj_indices[v_id], object_cloud);
            pcl::transformPointCloud( object_cloud, object_aligned, camera_transforms[v_id]);
            *big_cloud_unfiltered += object_aligned;
        }

        pcl::PointCloud<PointT>::Ptr octree_cloud(new pcl::PointCloud<PointT>);
        NMBasedCloudIntegration<PointT> nmIntegration (nm_int_param);
        nmIntegration.setInputClouds(clouds);
        nmIntegration.setPointProperties(pt_properties);
        nmIntegration.setTransformations(camera_transforms);
        nmIntegration.setInputNormals(normals);
        nmIntegration.setIndices(obj_indices);
        nmIntegration.compute(octree_cloud);
        std::vector< pcl::PointCloud<PointT>::Ptr > clouds_used;
        nmIntegration.getInputCloudsUsed(clouds_used);

        std::cout << "Size cloud unfiltered: " << big_cloud_unfiltered->points.size() << ", filtered: " << octree_cloud->points.size() << std::endl;


        if(vm.count("out_dir"))
        {
            const std::string out_path = out_dir + "/" + sub_folder;
            io::createDirIfNotExist(out_path);
            pcl::io::savePCDFileBinary(out_path + "/registered_cloud_filtered.pcd", *octree_cloud);
            pcl::io::savePCDFileBinary(out_path + "/registered_cloud_unfiltered.pcd", *big_cloud_unfiltered);

            for(size_t v_id=0; v_id<clouds_used.size(); v_id++)
            {
                if(debug)
                {
                    std::stringstream fn; fn << out_path << "/filtered_input_" << setfill('0') << setw(5) << v_id << ".pcd";
                    pcl::io::savePCDFileBinary(fn.str(), *clouds_used[v_id]);

                    fn.str(""); fn << out_path << "/distance_to_edge_px_" << setfill('0') << setw(5) << v_id << ".txt";
                    std::ofstream f (fn.str());
                    for (size_t pt_id = 0; pt_id < clouds_used[v_id]->points.size (); pt_id++) {
                        f << pt_properties[v_id][pt_id][2] << std::endl;

                        PointT &pt_tmp = clouds_used[v_id]->points[pt_id];
                        if( !pcl::isFinite( pt_tmp ) )  // set background color for nan points
                            pt_tmp.r = pt_tmp.g = pt_tmp.b = 255.f;
                    }
                    f.close();

                    fn.str(""); fn << out_path << "/filter_input_image" << setfill('0') << setw(5) << v_id << ".png";
                    cv::imwrite(fn.str(), ConvertPCLCloud2Image(*clouds_used[v_id]));
                }
            }
        }

        if(visualize)
        {
            if(!vis)
            {
                vis.reset( new pcl::visualization::PCLVisualizer("registered cloud") );
                vis->createViewPort(0,0,0.5,1,vp1);
                    vis->createViewPort(0.5,0,1,1,vp2);
            }
            vis->removeAllPointClouds(vp1);
            vis->removeAllPointClouds(vp2);
            vis->addPointCloud(big_cloud_unfiltered, "unfiltered_cloud", vp1);
            vis->addPointCloud(octree_cloud, "filtered_cloud", vp2);
            vis->spin();
        }
    }
}
