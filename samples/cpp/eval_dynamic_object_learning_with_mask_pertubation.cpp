#include <v4r/common/miscellaneous.h>
#include <v4r/object_modelling/do_learning.h>
#include <v4r/io/filesystem.h>
#include <pcl/console/parse.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <iostream>
#include <fstream>
#include <time.h>

//-do_erosion 1 -radius 0.005 -dot_product 0.99 -normal_method 0 -chop_z 2 -transfer_latest_only 0 -do_sift_based_camera_pose_estimation 0 -scenes_dir /media/Data/datasets/TUW/test_set -input_mask_dir /home/thomas/Desktop/test -output_dir /home/thomas/Desktop/out_test/ -visualize 1

int
main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PointT;

    std::string scene_dir, input_mask_dir, output_dir, output_rec_dir="/tmp", recognition_structure_dir;
    bool visualize = false;
    size_t min_mask_points = 50;

    v4r::object_modelling::DOL::Parameter param;
    pcl::console::parse_argument (argc, argv,  "-radius", param.radius_);
    pcl::console::parse_argument (argc, argv,  "-dot_product", param.eps_angle_);
    pcl::console::parse_argument (argc, argv,  "-dist_threshold_growing", param.dist_threshold_growing_);
    pcl::console::parse_argument (argc, argv,  "-seed_res", param.seed_resolution_);
    pcl::console::parse_argument (argc, argv,  "-voxel_res", param.voxel_resolution_);
    pcl::console::parse_argument (argc, argv,  "-ratio", param.ratio_supervoxel_);
    pcl::console::parse_argument (argc, argv,  "-do_erosion", param.do_erosion_);
    pcl::console::parse_argument (argc, argv,  "-do_mst_refinement", param.do_mst_refinement_);
    pcl::console::parse_argument (argc, argv,  "-do_sift_based_camera_pose_estimation", param.do_sift_based_camera_pose_estimation_);
    pcl::console::parse_argument (argc, argv,  "-transfer_latest_only", param.transfer_indices_from_latest_frame_only_);
    pcl::console::parse_argument (argc, argv,  "-chop_z", param.chop_z_);
    pcl::console::parse_argument (argc, argv,  "-normal_method", param.normal_method_);
    pcl::console::parse_argument (argc, argv,  "-ratio_cluster_obj_supported", param.ratio_cluster_obj_supported_);
    pcl::console::parse_argument (argc, argv,  "-ratio_cluster_occluded", param.ratio_cluster_occluded_);
    pcl::console::parse_argument (argc, argv,  "-visualize", visualize);

    v4r::object_modelling::DOL m(param);
    pcl::console::parse_argument (argc, argv,  "-stat_outlier_removal_meanK", m.sor_params_.meanK_);
    pcl::console::parse_argument (argc, argv,  "-stat_outlier_removal_std_mul", m.sor_params_.std_mul_);
    pcl::console::parse_argument (argc, argv,  "-inlier_threshold_plane_seg", m.p_param_.inlDist);
    pcl::console::parse_argument (argc, argv,  "-min_points_smooth_cluster", m.p_param_.minPointsSmooth);
    pcl::console::parse_argument (argc, argv,  "-min_plane_points", m.p_param_.minPoints);
    pcl::console::parse_argument (argc, argv,  "-smooth_clustering", m.p_param_.smooth_clustering);

    pcl::console::parse_argument (argc, argv, "-scenes_dir", scene_dir);
    pcl::console::parse_argument (argc, argv, "-input_mask_dir", input_mask_dir);
    pcl::console::parse_argument (argc, argv, "-output_dir", output_dir);
    pcl::console::parse_argument (argc, argv, "-output_rec_dir", output_rec_dir);
    if(pcl::console::parse_argument (argc, argv, "-recognition_structure_dir", recognition_structure_dir) == -1)
    {
        recognition_structure_dir = "/tmp/recognition_structure_dir";
    }

    m.initSIFT();

    if (scene_dir.compare ("") == 0)
    {
        PCL_ERROR("Set the directory containing scenes. Usage -scenes_dir files [path_to_dir].\n");
        return -1;
    }
    if (input_mask_dir.compare ("") == 0)
    {
        PCL_ERROR("Set the directory containing the input mask. Usage -input_mask_dir files [path_to_dir].\n");
        return -1;
    }
    if (output_dir.compare ("") == 0)
    {
        PCL_ERROR("Set the output directory. Usage -output_dir files [path_to_dir].\n");
        return -1;
    }


    std::stringstream output_rec_model, output_rec_structure;
    output_rec_model << output_rec_dir << "/models/";
    output_rec_structure << output_rec_dir << "/recognition_structure/";

    v4r::io::createDirIfNotExist(output_dir);
    v4r::io::createDirIfNotExist(output_rec_model.str());
    v4r::io::createDirIfNotExist(output_rec_structure.str());

    ofstream param_file;
    param_file.open ((output_dir + "/param.nfo").c_str());
    m.printParams(param_file);
    param_file    <<  "stat_outlier_removal_meanK" << m.sor_params_.meanK_ << std::endl
                  << "stat_outlier_removal_std_mul" << m.sor_params_.std_mul_ << std::endl
                  << "inlier_threshold_plane_seg" << m.p_param_.inlDist << std::endl
                  << "min_points_smooth_cluster" << m.p_param_.minPointsSmooth << std::endl
                  << "min_plane_points" << m.p_param_.minPoints << std::endl;
    param_file.close();

    srand(time(NULL));

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

            cv::Mat mask_img = cv::Mat(480, 640, CV_8UC1);
            for(size_t r=0; r < mask_img.rows; r++)
                for(size_t c=0; c < mask_img.cols; c++)
                    mask_img.at<unsigned char>(r,c) = 0;

            for(size_t i=0; i < mask.size(); i++)
            {
                int r,c;
                r = mask[i] / mask_img.cols;
                c = mask[i] % mask_img.cols;

                mask_img.at<unsigned char>(r,c) = 255;
            }
            cv::Mat const structure_elem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
            cv::Mat close_result;
            cv::morphologyEx(mask_img, close_result, cv::MORPH_CLOSE, structure_elem);

            cv::Mat mask_img_out = cv::Mat(480, 640, CV_8UC1);

            for (size_t noise_width = 5; noise_width < 81; noise_width=noise_width+10)
            {
//                cv::Mat mask_dst;/*
//                cv::erode(close_result, mask_dst, cv::Mat(), cv::Point(-1,-1), 1+std::floor(noise_width/4));
//                cv::blur( close_result, mask_img_out, cv::Size(noise_width, noise_width));
                cv::GaussianBlur(close_result, mask_img_out, cv::Size(noise_width, noise_width), 0, 0, cv::BORDER_DEFAULT );
//                cv::medianBlur( close_result, mask_img_out, noise_width);
//                cv::Mat mask_img = cv::Mat(480, 640, CV_8UC1);
//                for(size_t r=0; r < mask_img_out.rows; r++)
//                {
//                    for(size_t c=0; c < mask_img_out.cols; c++)
//                    {
//                        int tmp = std::abs( mask_img_out.at<unsigned char>(r,c) - (int)128);
//                        mask_img_out.at<unsigned char>(r,c) = tmp;
//                    }
//                 }
//                cv::namedWindow("mask");
//                cv::imshow("mask",mask_img);
//                cv::namedWindow("close");
//                cv::imshow("close",close_result);
//                cv::namedWindow("mask_img_out");
//                cv::imshow("mask_img_out",mask_img_out);
//                cv::namedWindow("diff");
                cv::Mat diff = cv::abs(mask_img_out-close_result);
//                std::cout << diff << std::endl;
//                cv::imshow("diff", diff);

                cv::Mat noise = cv::Mat(480, 640, CV_8UC1);
                for(size_t r=0; r < mask_img.rows; r++)
                {
                    for(size_t c=0; c < mask_img.cols; c++)
                    {
                        if (diff.at<unsigned char>(r,c) > 5)
                        {
                            if (rand()%255 > 128)
                                noise.at<unsigned char>(r,c)  = 255;
                            else
                                noise.at<unsigned char>(r,c)  = 0;
                        }
                        else
                            noise.at<unsigned char>(r,c) = mask_img.at<unsigned char>(r,c);
                    }
                }

             std::ofstream f;
             std::string out_mask_fn = output_dir_w_sub + "/" + mask_file_v[o_id];
             std::stringstream suffix_mask;
             suffix_mask << "mask_" << noise_width << ".txt";
             boost::replace_last (out_mask_fn, "mask.txt", suffix_mask.str());
             f.open(out_mask_fn.c_str());
             std::vector<size_t> mask_pertubated;
             for(size_t r=0; r < mask_img.rows; r++)
             {
                 for(size_t c=0; c < mask_img.cols; c++)
                 {
                     if (noise.at<unsigned char>(r,c)  > 128)
                     {
                         const int idx = r*640+c;
                         mask_pertubated.push_back(idx);
                         f << idx << std::endl;
                     }
                 }
             }
             f.close();
//                cv::namedWindow("noise");
//                cv::imshow("noise", cv::abs(noise));
//                cv::waitKey(0);
//                continue;

            const std::string scene_path = scene_dir + "/" + sub_folder_names[ sub_folder_id ];
            std::vector< std::string > views;
            v4r::io::getFilesInDirectory(scene_path, views, "", ".*.pcd", false);

            std::sort(views.begin(), views.end());

            std::cout << "Learning object from mask " << mask_file << " for scene " << scene_path << std::endl;

            for(size_t v_id=0; v_id<views.size(); v_id++)
            {
                const std::string view_file = scene_path + "/" + views[ v_id ];
                pcl::PointCloud<PointT>::Ptr pCloud(new pcl::PointCloud<PointT>());
                pcl::io::loadPCDFile(view_file, *pCloud);
                const Eigen::Matrix4f trans = v4r::common::RotTrans2Mat4f(pCloud->sensor_orientation_, pCloud->sensor_origin_);


                Eigen::Vector4f zero_origin;
                zero_origin[0] = zero_origin[1] = zero_origin[2] = zero_origin[3] = 0.f;
                pCloud->sensor_origin_ = zero_origin;   // for correct visualization
                pCloud->sensor_orientation_ = Eigen::Quaternionf::Identity();

                if (v_id==0)
                    m.learn_object(*pCloud, trans, mask_pertubated);
                else
                    m.learn_object(*pCloud, trans);
            }

            std::string out_fn = mask_file_v[o_id];
            std::stringstream suffix;
            suffix << "dol_" << noise_width << ".pcd";
            boost::replace_last (out_fn, "mask.txt", suffix.str());
            m.save_model(output_dir_w_sub, recognition_structure_dir, out_fn);
            m.write_model_to_disk(output_rec_model.str(), output_rec_structure.str(), sub_folder_names[ sub_folder_id ]+"_object.pcd");
            if (visualize)
                m.visualize();
            m.clear();
            }
        }
    }
    return 0;
}
