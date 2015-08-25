#include <v4r/recognition/singleview_object_recognizer.h>
#include <pcl/console/parse.h>
#include <v4r/io/filesystem.h>

#include <iostream>

int
main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PointT;
    bool visualize = false;

    std::string models_dir, training_dir_sift, training_dir_shot, training_dir_ourcvfh, sift_structure, test_dir;

    v4r::SingleViewRecognizer r;
    pcl::console::parse_argument (argc, argv,  "-test_dir", test_dir);
    pcl::console::parse_argument (argc, argv,  "-models_dir", models_dir);
    pcl::console::parse_argument (argc, argv,  "-training_dir_sift", training_dir_sift);
    pcl::console::parse_argument (argc, argv,  "-training_dir_shot", training_dir_shot);
    pcl::console::parse_argument (argc, argv,  "-recognizer_structure_sift", sift_structure);
    pcl::console::parse_argument (argc, argv,  "-training_dir_ourcvfh", training_dir_ourcvfh);

    pcl::console::parse_argument (argc, argv,  "-chop_z", r.sv_params_.chop_at_z_ );
    pcl::console::parse_argument (argc, argv,  "-icp_iterations", r.sv_params_.icp_iterations_);
    pcl::console::parse_argument (argc, argv,  "-do_sift", r.sv_params_.do_sift_);
    pcl::console::parse_argument (argc, argv,  "-do_shot", r.sv_params_.do_shot_);
    pcl::console::parse_argument (argc, argv,  "-do_ourcvfh", r.sv_params_.do_ourcvfh_);
    pcl::console::parse_argument (argc, argv,  "-knn_sift", r.sv_params_.knn_sift_);

    pcl::console::parse_argument (argc, argv,  "-cg_size_thresh", r.cg_params_.cg_size_threshold_);
    pcl::console::parse_argument (argc, argv,  "-cg_size", r.cg_params_.cg_size_);
    pcl::console::parse_argument (argc, argv,  "-cg_ransac_threshold", r.cg_params_.ransac_threshold_);
    pcl::console::parse_argument (argc, argv,  "-cg_dist_for_clutter_factor", r.cg_params_.dist_for_clutter_factor_);
    pcl::console::parse_argument (argc, argv,  "-cg_max_taken", r.cg_params_.max_taken_);
    pcl::console::parse_argument (argc, argv,  "-cg_max_time_for_cliques_computation", r.cg_params_.max_time_for_cliques_computation_);
    pcl::console::parse_argument (argc, argv,  "-cg_dot_distance", r.cg_params_.dot_distance_);
    pcl::console::parse_argument (argc, argv,  "-use_cg_graph", r.cg_params_.use_cg_graph_);

    pcl::console::parse_argument (argc, argv,  "-hv_params_clutter_regularizer", r.hv_params_.clutter_regularizer_);
    pcl::console::parse_argument (argc, argv,  "-hv_params_color_sigma_ab", r.hv_params_.color_sigma_ab_);
    pcl::console::parse_argument (argc, argv,  "-hv_params_color_sigma_al", r.hv_params_.color_sigma_l_);
    pcl::console::parse_argument (argc, argv,  "-hv_params_detect_clutter", r.hv_params_.detect_clutter_);
    pcl::console::parse_argument (argc, argv,  "-hv_params_duplicity_cm_weight", r.hv_params_.duplicity_cm_weight_);
    pcl::console::parse_argument (argc, argv,  "-hv_params_histogram_specification", r.hv_params_.histogram_specification_);
    pcl::console::parse_argument (argc, argv,  "-hv_params_hyp_penalty", r.hv_params_.hyp_penalty_);
    pcl::console::parse_argument (argc, argv,  "-hv_params_ignore_color", r.hv_params_.ignore_color_);
    pcl::console::parse_argument (argc, argv,  "-hv_params_initial_status", r.hv_params_.initial_status_);
    pcl::console::parse_argument (argc, argv,  "-hv_params_inlier_threshold", r.hv_params_.inlier_threshold_);
    pcl::console::parse_argument (argc, argv,  "-hv_params_occlusion_threshold", r.hv_params_.occlusion_threshold_);
    pcl::console::parse_argument (argc, argv,  "-hv_params_optimizer_type", r.hv_params_.optimizer_type_);
    pcl::console::parse_argument (argc, argv,  "-hv_params_radius_clutter", r.hv_params_.radius_clutter_);
    pcl::console::parse_argument (argc, argv,  "-hv_params_radius_normals", r.hv_params_.radius_normals_);
    pcl::console::parse_argument (argc, argv,  "-hv_params_regularizer", r.hv_params_.regularizer_);
    pcl::console::parse_argument (argc, argv,  "-hv_params_requires_normals", r.hv_params_.requires_normals_);
    // continue....

    r.setTraining_dir_sift(training_dir_sift);
    r.setTraining_dir_shot(training_dir_shot);
    r.setTraining_dir_ourcvfh(training_dir_ourcvfh);
    r.setModels_dir       (models_dir);
    r.setSift_structure(sift_structure);
    r.initialize();
    r.printParams();

    std::vector< std::string> sub_folder_names;
    if(!v4r::io::getFoldersInDirectory( test_dir, "", sub_folder_names) )
    {
        std::cerr << "No subfolders in directory " << test_dir << ". " << std::endl;
        sub_folder_names.push_back("");
    }

    std::sort(sub_folder_names.begin(), sub_folder_names.end());
    for (size_t sub_folder_id=0; sub_folder_id < sub_folder_names.size(); sub_folder_id++)
    {
        const std::string sequence_path = test_dir + "/" + sub_folder_names[ sub_folder_id ];
        std::vector< std::string > views;
        v4r::io::getFilesInDirectory(sequence_path, views, "", ".*.pcd", false);

        for (size_t v_id=0; v_id<views.size(); v_id++)
        {
            const std::string fn = test_dir + "/" + sub_folder_names[sub_folder_id] + "/" + views[ v_id ];

            std::cout << "Recognizing file " << fn << std::endl;
            pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
            pcl::io::loadPCDFile(fn, *cloud);
            r.setInputCloud(cloud);
            r.recognize();
        }
    }
  return 0;
}
