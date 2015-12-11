#include <v4r/common/miscellaneous.h>
#include <v4r/object_modelling/do_learning.h>
#include <v4r/io/filesystem.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <iostream>
#include <fstream>

#include <time.h>

#include <boost/any.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

//--do_erosion 1 --radius 0.005 --dot_product 0.99 --normal_method 0 -z 2 --transfer_latest_only 0 --do_sift_based_camera_pose_estimation 0 -s /media/Data/datasets/TUW_new/test_set_icra16 -m /media/Data/datasets/TUW/TUW_gt_models_first -o /home/thomas/Desktop/out_TUW/

//deprecated
//-do_erosion 1 -radius 0.005 -dot_product 0.99 -normal_method 0 -chop_z 2 -transfer_latest_only 0 -do_sift_based_camera_pose_estimation 0 -scenes_dir /media/Data/datasets/TUW/test_set -input_mask_dir /home/thomas/Desktop/test -output_dir /home/thomas/Desktop/out_test/ -visualize 1

double getTimeDiff(timeval a, timeval b);

double getTimeDiff(timeval a, timeval b)
{
    double first = a.tv_sec + (a.tv_usec/1000000.0);
    double second = b.tv_sec + (b.tv_usec/1000000.0);

    return (first - second)*1000;
}

int
main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PointT;

    std::string scene_dir, input_mask_dir, output_dir;
    bool visualize = false;
    size_t min_mask_points = 50;
    bool first_frame_only=false; // used for evaluation when only using the first view

    v4r::object_modelling::DOL m;

    po::options_description desc("Evaluation Dynamic Object Learning with Ground Truth\n======================================\n **Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("scenes_dir,s", po::value<std::string>(&scene_dir)->required(), "input directory with .pcd files of the scenes. Each folder is considered as seperate sequence. Views are sorted alphabetically and object mask is applied on first view.")
            ("input_mask_dir,m", po::value<std::string>(&input_mask_dir)->required(), "directory containing the object masks used as a seed to learn the object in the first cloud")
            ("output_dir,o", po::value<std::string>(&output_dir)->default_value(output_dir), "Output directory where the model, training data, timing information and parameter values will be stored")

            ("radius,r", po::value<double>(&m.param_.radius_)->default_value(m.param_.radius_), "Radius used for region growing. Neighboring points within this distance are candidates for clustering it to the object model.")
            ("dot_product", po::value<double>(&m.param_.eps_angle_)->default_value(m.param_.eps_angle_), "Threshold for the normals dot product used for region growing. Neighboring points with a surface normal within this threshold are candidates for clustering it to the object model.")
            ("dist_threshold_growing", po::value<double>(&m.param_.dist_threshold_growing_)->default_value(m.param_.dist_threshold_growing_), "")
            ("seed_res", po::value<double>(&m.param_.seed_resolution_)->default_value(m.param_.seed_resolution_), "")
            ("voxel_res", po::value<double>(&m.param_.voxel_resolution_)->default_value(m.param_.voxel_resolution_), "")
            ("ratio", po::value<double>(&m.param_.ratio_supervoxel_)->default_value(m.param_.ratio_supervoxel_), "")
            ("do_erosion", po::value<bool>(&m.param_.do_erosion_)->default_value(m.param_.do_erosion_), "")
            ("do_mst_refinement", po::value<bool>(&m.param_.do_mst_refinement_)->default_value(m.param_.do_mst_refinement_), "")
            ("do_sift_based_camera_pose_estimation", po::value<bool>(&m.param_.do_sift_based_camera_pose_estimation_)->default_value(m.param_.do_sift_based_camera_pose_estimation_), "")
            ("transfer_latest_only", po::value<bool>(&m.param_.transfer_indices_from_latest_frame_only_)->default_value(m.param_.transfer_indices_from_latest_frame_only_), "")
            ("chop_z,z", po::value<double>(&m.param_.chop_z_)->default_value(m.param_.chop_z_), "Cut-off distance of the input clouds with respect to the camera. Points further away than this distance will be ignored.")
            ("normal_method,n", po::value<int>(&m.param_.normal_method_)->default_value(m.param_.normal_method_), "")
            ("ratio_cluster_obj_supported", po::value<double>(&m.param_.ratio_cluster_obj_supported_)->default_value(m.param_.ratio_cluster_obj_supported_), "")
            ("ratio_cluster_occluded", po::value<double>(&m.param_.ratio_cluster_occluded_)->default_value(m.param_.ratio_cluster_occluded_), "")

            ("stat_outlier_removal_meanK", po::value<int>(&m.sor_params_.meanK_)->default_value(m.sor_params_.meanK_), "MeanK used for statistical outlier removal (see PCL documentation)")
            ("stat_outlier_removal_std_mul", po::value<double>(&m.sor_params_.std_mul_)->default_value(m.sor_params_.std_mul_), "Standard Deviation multiplier used for statistical outlier removal (see PCL documentation)")
            ("inlier_threshold_plane_seg", po::value<double>(&m.p_param_.inlDist)->default_value(m.p_param_.inlDist), "")
            ("min_points_smooth_cluster", po::value<int>(&m.p_param_.minPointsSmooth)->default_value(m.p_param_.minPointsSmooth), "Minimum number of points for a cluster")
            ("min_plane_points", po::value<int>(&m.p_param_.minPoints)->default_value(m.p_param_.minPoints), "Minimum number of points for a cluster to be a candidate for a plane")
            ("smooth_clustering", po::value<bool>(&m.p_param_.smooth_clustering)->default_value(m.p_param_.smooth_clustering), "If true, does smooth clustering. Otherwise only plane clustering.")

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

    m.initSIFT();

    v4r::io::createDirIfNotExist(output_dir);

    ofstream param_file;
    param_file.open ((output_dir + "/param.nfo").c_str());
    m.printParams(param_file);
    param_file    <<  "stat_outlier_removal_meanK" << m.sor_params_.meanK_ << std::endl
                  << "stat_outlier_removal_std_mul" << m.sor_params_.std_mul_ << std::endl
                  << "inlier_threshold_plane_seg" << m.p_param_.inlDist << std::endl
                  << "min_points_smooth_cluster" << m.p_param_.minPointsSmooth << std::endl
                  << "min_plane_points" << m.p_param_.minPoints << std::endl;
    param_file.close();

    std::vector< std::string> sub_folder_names;
    if(!v4r::io::getFoldersInDirectory( scene_dir, "", sub_folder_names) )
    {
        std::cerr << "No subfolders in directory " << scene_dir << ". " << std::endl;
        sub_folder_names.push_back("");
    }

    std::sort(sub_folder_names.begin(), sub_folder_names.end());

    v4r::io::createDirIfNotExist(output_dir + "/models");

    for (size_t sub_folder_id=0; sub_folder_id < sub_folder_names.size(); sub_folder_id++)
    {
        const std::string annotations_dir = input_mask_dir + "/" + sub_folder_names[ sub_folder_id ];
        std::vector< std::string > mask_file_v;
        v4r::io::getFilesInDirectory(annotations_dir, mask_file_v, "", ".*.txt", false);

        std::sort(mask_file_v.begin(), mask_file_v.end());

        for (size_t o_id=0; o_id<mask_file_v.size(); o_id++)
        {
            const std::string mask_file = annotations_dir + "/" + mask_file_v[o_id];

            std::ifstream initial_mask_file;
            initial_mask_file.open( mask_file.c_str() );

            size_t idx_tmp;
            std::vector<size_t> mask;
            while (initial_mask_file >> idx_tmp)
            {
                mask.push_back(idx_tmp);
            }
            initial_mask_file.close();

            if ( mask.size() < min_mask_points) // not enough points to grow an object
                continue;


            const std::string scene_path = scene_dir + "/" + sub_folder_names[ sub_folder_id ];
            std::vector< std::string > views;
            v4r::io::getFilesInDirectory(scene_path, views, "", ".*.pcd", false);

            std::sort(views.begin(), views.end());

            std::cout << "Learning object from mask " << mask_file << " for scene " << scene_path << std::endl;

            timeval start, stop;

            gettimeofday(&start, NULL);
            for(size_t v_id=0; v_id<views.size(); v_id++)
            {
                const std::string view_file = scene_path + "/" + views[ v_id ];
                pcl::PointCloud<PointT>::Ptr pCloud(new pcl::PointCloud<PointT>());
                pcl::io::loadPCDFile(view_file, *pCloud);
                const Eigen::Matrix4f trans = v4r::RotTrans2Mat4f(pCloud->sensor_orientation_, pCloud->sensor_origin_);


                Eigen::Vector4f zero_origin;
                zero_origin[0] = zero_origin[1] = zero_origin[2] = zero_origin[3] = 0.f;
                pCloud->sensor_origin_ = zero_origin;   // for correct visualization
                pCloud->sensor_orientation_ = Eigen::Quaternionf::Identity();

                if (v_id==0)
                    m.learn_object(*pCloud, trans, mask);
                else
                {
                    if(!first_frame_only)
                        m.learn_object(*pCloud, trans);
                }
            }
            gettimeofday(&stop, NULL);

            m.save_model(output_dir + "/models", output_dir + "/training_data/", sub_folder_names[ sub_folder_id ] + "_dol.pcd");
            if (visualize)
                m.visualize();
            m.clear();

            // write running time to file
            const std::string timing_fn = output_dir + "/" + sub_folder_names[ sub_folder_id ] + "_timing.nfo";
            double learning_time = getTimeDiff(stop, start);
            ofstream f( timing_fn.c_str() );
            f << learning_time;
            f.close();
        }
    }
    return 0;
}
