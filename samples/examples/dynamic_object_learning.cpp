#include <v4r/common/miscellaneous.h>
#include <v4r/object_modelling/do_learning.h>
#include <v4r/io/filesystem.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <iostream>
#include <fstream>

#include <boost/any.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;

//-do_erosion 1 -radius 0.005 -dot_product 0.99 -normal_method 0 -chop_z 2 -transfer_latest_only 0 -do_sift_based_camera_pose_estimation 0 -scenes_dir /media/Data/datasets/TUW/test_set -input_mask_dir /home/thomas/Desktop/test -output_dir /home/thomas/Desktop/out_test/ -visualize 1

int
main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PointT;

    std::string scene_dir, input_mask_dir, output_dir;
    bool visualize;
    size_t min_mask_points = 50;

    v4r::object_modelling::DOL m;

    po::options_description desc("Dynamic Object Learning\n======================================\n**Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("radius", po::value<double>( &m.param_.radius_ )->default_value( m.param_.radius_ ), "")
            ("dot_product", po::value<double>( &m.param_.eps_angle_ )->default_value( m.param_.eps_angle_ ), "")
            ("dist_threshold_growing", po::value<double>( &m.param_.dist_threshold_growing_ )->default_value( m.param_.dist_threshold_growing_ ), "")
            ("seed_res", po::value<double>( &m.param_.seed_resolution_ )->default_value( m.param_.seed_resolution_ ), "")
            ("voxel_res", po::value<double>( &m.param_.voxel_resolution_ )->default_value( m.param_.voxel_resolution_ ), "")
            ("ratio", po::value<double>( &m.param_.ratio_supervoxel_ )->default_value( m.param_.ratio_supervoxel_ ), "")
            ("do_erosion", po::value<bool>( &m.param_.do_erosion_ )->default_value( m.param_.do_erosion_ ), "")
            ("do_mst_refinement", po::value<bool>( &m.param_.do_mst_refinement_ )->default_value( m.param_.do_mst_refinement_ ), "")
            ("do_sift_based_camera_pose_estimation", po::value<bool>( &m.param_.do_sift_based_camera_pose_estimation_ )->default_value( m.param_.do_sift_based_camera_pose_estimation_ ), "")
            ("transfer_latest_only", po::value<bool>( &m.param_.transfer_indices_from_latest_frame_only_ )->default_value( m.param_.transfer_indices_from_latest_frame_only_ ), "")
            ("chop_z", po::value<double>( &m.param_.chop_z_ )->default_value( m.param_.chop_z_ ), "")
            ("normal_method", po::value<int>( &m.param_.normal_method_ )->default_value( m.param_.normal_method_ ), "")
            ("ratio_cluster_obj_supported", po::value<double>( &m.param_.ratio_cluster_obj_supported_ )->default_value( m.param_.ratio_cluster_obj_supported_ ), "")
            ("ratio_cluster_occluded", po::value<double>( &m.param_.ratio_cluster_occluded_ )->default_value( m.param_.ratio_cluster_occluded_ ), "")
            ("visualize", po::value<bool>( &visualize )->default_value( false ), "")

            ("stat_outlier_removal_meanK", po::value<int>( &m.sor_params_.meanK_ )->default_value( m.sor_params_.meanK_ ), "")
            ("stat_outlier_removal_std_mul", po::value<double>( &m.sor_params_.std_mul_ )->default_value( m.sor_params_.std_mul_ ), "")
            ("inlier_threshold_plane_seg", po::value<double>( &m.p_param_.inlDist )->default_value( m.p_param_.inlDist ), "")
            ("min_points_smooth_cluster", po::value<int>( &m.p_param_.minPointsSmooth )->default_value( m.p_param_.minPointsSmooth ), "")
            ("min_plane_points", po::value<int>( &m.p_param_.minPoints )->default_value( m.p_param_.minPoints ), "")
            ("smooth_clustering", po::value<bool>( &m.p_param_.smooth_clustering )->default_value( m.p_param_.smooth_clustering ), "")

            ("scenes_dir", po::value<std::string>(&scene_dir )->required(), "")
            ("input_mask_dir", po::value<std::string>(&input_mask_dir )->required(), "")
            ("output_dir", po::value<std::string>(&output_dir )->required(), "")
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


    // writing parameters to file
    ofstream param_file ((output_dir + "/param.nfo").c_str());
    for(const auto& it : vm)
    {
      param_file << "--" << it.first << " ";

      auto& value = it.second.value();
      if (auto v = boost::any_cast<double>(&value))
        param_file << std::setprecision(3) << *v;
      else if (auto v = boost::any_cast<std::string>(&value))
        param_file << *v;
      else if (auto v = boost::any_cast<bool>(&value))
        param_file << *v;
      else if (auto v = boost::any_cast<int>(&value))
        param_file << *v;
      else if (auto v = boost::any_cast<size_t>(&value))
        param_file << *v;
      else
        param_file << "error";

      param_file << " ";
    }
    param_file.close();


    std::vector< std::string> sub_folder_names;
    if(!v4r::io::getFoldersInDirectory( scene_dir, "", sub_folder_names) )
    {
        std::cerr << "No subfolders in directory " << scene_dir << ". " << std::endl;
        sub_folder_names.push_back("");
    }

    std::sort(sub_folder_names.begin(), sub_folder_names.end());

    for (size_t sub_folder_id=0; sub_folder_id < sub_folder_names.size(); sub_folder_id++)
    {
        const std::string output_dir_w_sub = output_dir + "/" + sub_folder_names[ sub_folder_id ];
        const std::string output_rec_model = output_dir_w_sub + "/models/";
        const std::string output_rec_structure = output_dir_w_sub + "/recognition_structure/";

        v4r::io::createDirIfNotExist(output_dir_w_sub);

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

            for(size_t v_id=0; v_id<views.size(); v_id++)
            {
                const std::string view_file = scene_path + "/" + views[ v_id ];
                pcl::PointCloud<PointT> cloud;
                pcl::io::loadPCDFile(view_file, cloud);
                const Eigen::Matrix4f trans = v4r::RotTrans2Mat4f(cloud.sensor_orientation_, cloud.sensor_origin_);


                Eigen::Vector4f zero_origin;
                zero_origin[0] = zero_origin[1] = zero_origin[2] = zero_origin[3] = 0.f;
                cloud.sensor_origin_ = zero_origin;   // for correct visualization
                cloud.sensor_orientation_ = Eigen::Quaternionf::Identity();

                if (v_id==0)
                    m.learn_object(cloud, trans, mask);
                else
                    m.learn_object(cloud, trans);
            }

            std::string out_fn = mask_file_v[o_id];
            boost::replace_last (out_fn, "mask.txt", "dol.pcd");
            m.save_model(output_rec_model, output_rec_structure, out_fn);
//            m.write_model_to_disk(output_rec_structure.str(), output_rec_structure.str(), sub_folder_names[ sub_folder_id ]);
            if (visualize)
                m.visualize();
            m.clear();
        }
    }
    return 0;
}
