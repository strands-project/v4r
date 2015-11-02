#include <v4r/common/miscellaneous.h>
#include <v4r/recognition/singleview_object_recognizer.h>
#include <v4r/io/filesystem.h>

#include <pcl/common/centroid.h>
#include <pcl/console/parse.h>
#include <pcl/visualization/cloud_viewer.h>

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdlib.h>


class Recognizer
{
private:
    typedef pcl::PointXYZRGB PointT;
    typedef v4r::Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;

    v4r::SingleViewRecognizer r_;
    std::string test_dir_, out_dir_;
    bool visualize_;
    pcl::visualization::PCLVisualizer::Ptr vis_;
    std::map<std::string, size_t> rec_models_per_id_;

public:
    Recognizer()
    {
        out_dir_ = "/tmp/sv_recognition_out/";
        visualize_ = true;
    }

    void visualize_result(const pcl::PointCloud<PointT>::ConstPtr &cloud, const std::vector<ModelTPtr> &models, const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transforms)
    {
        r_.visualize();

        if(!vis_)
            vis_.reset ( new pcl::visualization::PCLVisualizer("Recognition Results") );
        vis_->removeAllPointClouds();
        vis_->removeAllShapes();
        vis_->addPointCloud(cloud, "cloud");

        for(size_t m_id=0; m_id<models.size(); m_id++)
        {
            const std::string model_id = models[m_id]->id_.substr(0, models[m_id]->id_.length() - 4);
            std::stringstream model_text;
            model_text << model_id << "_" << m_id;
            pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT>() );
            pcl::PointCloud<PointT>::ConstPtr model_cloud = models[m_id]->getAssembled( 0.003f );
            pcl::transformPointCloud( *model_cloud, *model_aligned, transforms[m_id]);

            PointT centroid;
            pcl::computeCentroid(*model_aligned, centroid);
            centroid.x += cloud->sensor_origin_[0];
            centroid.y += cloud->sensor_origin_[1];
            centroid.z += cloud->sensor_origin_[2];
            const float r=50+rand()%205;
            const float g=50+rand()%205;
            const float b=50+rand()%205;
            vis_->addText3D(model_text.str(), centroid, 0.01, r/255, g/255, b/255);

            model_aligned->sensor_orientation_ = cloud->sensor_orientation_;
            model_aligned->sensor_origin_ = cloud->sensor_origin_;
            vis_->addPointCloud(model_aligned, model_text.str());
        }
        vis_->spin();
    }

    bool initialize(int argc, char ** argv)
    {
        pcl::console::parse_argument (argc, argv,  "-visualize", visualize_);
        pcl::console::parse_argument (argc, argv,  "-out_dir", out_dir_);
        pcl::console::parse_argument (argc, argv,  "-test_dir", test_dir_);

        pcl::console::parse_argument (argc, argv,  "-models_dir", r_.models_dir_);
        pcl::console::parse_argument (argc, argv,  "-training_dir", r_.training_dir_);

        pcl::console::parse_argument (argc, argv,  "-idx_flann_fn_sift", r_.idx_flann_fn_sift_);
        pcl::console::parse_argument (argc, argv,  "-idx_flann_fn_shot", r_.idx_flann_fn_shot_);

        pcl::console::parse_argument (argc, argv,  "-chop_z", r_.sv_params_.chop_at_z_ );
        pcl::console::parse_argument (argc, argv,  "-icp_iterations", r_.sv_params_.icp_iterations_);
        pcl::console::parse_argument (argc, argv,  "-do_sift", r_.sv_params_.do_sift_);
        pcl::console::parse_argument (argc, argv,  "-do_shot", r_.sv_params_.do_shot_);
        pcl::console::parse_argument (argc, argv,  "-do_ourcvfh", r_.sv_params_.do_ourcvfh_);
        pcl::console::parse_argument (argc, argv,  "-knn_sift", r_.sv_params_.knn_sift_);

        pcl::console::parse_argument (argc, argv,  "-cg_size_thresh", r_.cg_params_.cg_size_threshold_);
        pcl::console::parse_argument (argc, argv,  "-cg_size", r_.cg_params_.cg_size_);
        pcl::console::parse_argument (argc, argv,  "-cg_ransac_threshold", r_.cg_params_.ransac_threshold_);
        pcl::console::parse_argument (argc, argv,  "-cg_dist_for_clutter_factor", r_.cg_params_.dist_for_clutter_factor_);
        pcl::console::parse_argument (argc, argv,  "-cg_max_taken", r_.cg_params_.max_taken_);
        pcl::console::parse_argument (argc, argv,  "-cg_max_time_for_cliques_computation", r_.cg_params_.max_time_for_cliques_computation_);
        pcl::console::parse_argument (argc, argv,  "-cg_dot_distance", r_.cg_params_.dot_distance_);
        pcl::console::parse_argument (argc, argv,  "-use_cg_graph", r_.cg_params_.use_cg_graph_);

        pcl::console::parse_argument (argc, argv,  "-hv_clutter_regularizer", r_.hv_params_.clutter_regularizer_);
        pcl::console::parse_argument (argc, argv,  "-hv_color_sigma_ab", r_.hv_params_.color_sigma_ab_);
        pcl::console::parse_argument (argc, argv,  "-hv_color_sigma_l", r_.hv_params_.color_sigma_l_);
        pcl::console::parse_argument (argc, argv,  "-hv_detect_clutter", r_.hv_params_.detect_clutter_);
        pcl::console::parse_argument (argc, argv,  "-hv_duplicity_cm_weight", r_.hv_params_.duplicity_cm_weight_);
        pcl::console::parse_argument (argc, argv,  "-hv_histogram_specification", r_.hv_params_.histogram_specification_);
        pcl::console::parse_argument (argc, argv,  "-hv_hyp_penalty", r_.hv_params_.hyp_penalty_);
        pcl::console::parse_argument (argc, argv,  "-hv_ignore_color", r_.hv_params_.ignore_color_);
        pcl::console::parse_argument (argc, argv,  "-hv_initial_status", r_.hv_params_.initial_status_);
        pcl::console::parse_argument (argc, argv,  "-hv_inlier_threshold", r_.hv_params_.inlier_threshold_);
        pcl::console::parse_argument (argc, argv,  "-hv_occlusion_threshold", r_.hv_params_.occlusion_threshold_);
        pcl::console::parse_argument (argc, argv,  "-hv_optimizer_type", r_.hv_params_.optimizer_type_);
        pcl::console::parse_argument (argc, argv,  "-hv_radius_clutter", r_.hv_params_.radius_clutter_);
        pcl::console::parse_argument (argc, argv,  "-hv_radius_normals", r_.hv_params_.radius_normals_);
        pcl::console::parse_argument (argc, argv,  "-hv_regularizer", r_.hv_params_.regularizer_);
        pcl::console::parse_argument (argc, argv,  "-hv_requires_normals", r_.hv_params_.requires_normals_);


        v4r::io::createDirIfNotExist(out_dir_);
        r_.initialize();
        ofstream param_file;
        param_file.open ((out_dir_ + "/param.nfo").c_str());
        r_.printParams(param_file);
        param_file.close();
        r_.printParams();
        return true;
    }

    bool eval()
    {
        std::vector< std::string> sub_folder_names;
        if(!v4r::io::getFoldersInDirectory( test_dir_, "", sub_folder_names) )
        {
            std::cerr << "No subfolders in directory " << test_dir_ << ". " << std::endl;
            sub_folder_names.push_back("");
        }

        std::sort(sub_folder_names.begin(), sub_folder_names.end());
        for (size_t sub_folder_id=0; sub_folder_id < sub_folder_names.size(); sub_folder_id++)
        {
            const std::string sequence_path = test_dir_ + "/" + sub_folder_names[ sub_folder_id ];
            const std::string out_path = out_dir_ + "/" + sub_folder_names[ sub_folder_id ];
            v4r::io::createDirIfNotExist(out_path);

            rec_models_per_id_.clear();

            std::vector< std::string > views;
            v4r::io::getFilesInDirectory(sequence_path, views, "", ".*.pcd", false);
            std::sort(views.begin(), views.end());
            for (size_t v_id=0; v_id<views.size(); v_id++)
            {
                const std::string fn = test_dir_ + "/" + sub_folder_names[sub_folder_id] + "/" + views[ v_id ];

                std::cout << "Recognizing file " << fn << std::endl;
                pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
                pcl::io::loadPCDFile(fn, *cloud);
                r_.setInputCloud(cloud);
                r_.recognize();

                std::vector<ModelTPtr> verified_models;
                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_verified;
                r_.getModelsAndTransforms(verified_models, transforms_verified);
                if (visualize_)
                    visualize_result(cloud, verified_models, transforms_verified);

                for(size_t m_id=0; m_id<verified_models.size(); m_id++)
                {
                    std::cout << "********************" << verified_models[m_id]->id_ << std::endl;

                    const std::string model_id = verified_models[m_id]->id_;
                    const Eigen::Matrix4f tf = transforms_verified[m_id];

                    size_t num_models_per_model_id;

                    std::map<std::string, size_t>::iterator it_rec_mod;
                    it_rec_mod = rec_models_per_id_.find(model_id);
                    if(it_rec_mod == rec_models_per_id_.end())
                    {
                        rec_models_per_id_.insert(std::pair<std::string, size_t>(model_id, 1));
                        num_models_per_model_id = 0;
                    }
                    else
                    {
                        num_models_per_model_id = it_rec_mod->second;
                        it_rec_mod->second++;
                    }

                    std::stringstream out_fn;
                    out_fn << out_path << "/" << views[v_id].substr(0, views[v_id].length()-4) << "_"
                           << model_id.substr(0, model_id.length() - 4) << "_" << num_models_per_model_id << ".txt";

                    ofstream or_file;
                    or_file.open (out_fn.str().c_str());
                    for (size_t row=0; row <4; row++)
                    {
                        for(size_t col=0; col<4; col++)
                        {
                            or_file << tf(row, col) << " ";
                        }
                    }
                    or_file.close();
                }
            }
        }
        return true;
    }
};

int
main (int argc, char ** argv)
{
    srand (time(NULL));
    Recognizer r_eval;
    r_eval.initialize(argc,argv);
    r_eval.eval();
    return 0;
}
