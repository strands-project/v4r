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

bool copyDir(const filesystem::path &source, const filesystem::path &destination);

bool copyDir(const boost::filesystem::path &source, const boost::filesystem::path &destination)
{
    namespace fs = boost::filesystem;
    try
    {
        // Check whether the function call is valid
        if(
            !fs::exists(source) ||
            !fs::is_directory(source)
        )
        {
            std::cerr << "Source directory " << source.string()
                << " does not exist or is not a directory." << '\n'
            ;
            return false;
        }
        if(fs::exists(destination))
        {
            std::cerr << "Destination directory " << destination.string()
                << " already exists." << '\n'
            ;
            return false;
        }
        // Create the destination directory
        if(!fs::create_directory(destination))
        {
            std::cerr << "Unable to create destination directory"
                << destination.string() << '\n'
            ;
            return false;
        }
    }
    catch(fs::filesystem_error const & e)
    {
        std::cerr << e.what() << '\n';
        return false;
    }
    // Iterate through the source directory
    for(
        fs::directory_iterator file(source);
        file != fs::directory_iterator(); ++file
    )
    {
        try
        {
            fs::path current(file->path());
            if(fs::is_directory(current))
            {
                // Found directory: Recursion
                if(
                    !copyDir(
                        current,
                        destination / current.filename()
                    )
                )
                {
                    return false;
                }
            }
            else
            {
                // Found file: Copy
                fs::copy_file(
                    current,
                    destination / current.filename()
                );
            }
        }
        catch(fs::filesystem_error const & e)
        {
            std:: cerr << e.what() << '\n';
        }
    }
    return true;
}

class EvalSvRecognizer
{
private:
    typedef pcl::PointXYZRGB PointT;
    typedef v4r::Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;

    v4r::SingleViewRecognizer r_;
    std::string test_dir_, out_dir_;
    std::string training_dir_tmp_;
    std::string dol_rec_dir_;
    std::string offline_rec_dir_;
    bool visualize_;
    pcl::visualization::PCLVisualizer::Ptr vis_;
    std::map<std::string, size_t> rec_models_per_id_;

    std::map<std::string, std::vector<std::string> > prun2ob;

public:
    EvalSvRecognizer()
    {
        out_dir_ = "/tmp/sv_recognition_out/";
        training_dir_tmp_ = "/tmp/recognition_dir/";
        visualize_ = true;
    }

    void visualize_result(const pcl::PointCloud<PointT>::ConstPtr &cloud, const std::vector<ModelTPtr> &models, const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transforms)
    {
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
        bool retrain;

        std::string map_file;

        pcl::console::parse_argument (argc, argv,  "-retrain", retrain);
        pcl::console::parse_argument (argc, argv,  "-visualize", visualize_);
        pcl::console::parse_argument (argc, argv,  "-out_dir", out_dir_);
        pcl::console::parse_argument (argc, argv,  "-test_dir", test_dir_);
        pcl::console::parse_argument (argc, argv,  "-dol_rec_dir", dol_rec_dir_);
        pcl::console::parse_argument (argc, argv,  "-offline_rec_dir", offline_rec_dir_);
        pcl::console::parse_argument (argc, argv,  "-training_dir_tmp", training_dir_tmp_);
        pcl::console::parse_argument (argc, argv,  "-map_file", map_file);

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

        v4r::io::createDirIfNotExist(training_dir_tmp_ + "/models");
        v4r::io::createDirIfNotExist(training_dir_tmp_ + "/recognition_structure");

        std::ifstream in;
        in.open (map_file.c_str (), std::ifstream::in);
        char linebuf[1024];
        while(in.getline (linebuf, 1024))
        {
            std::string line (linebuf);
            std::vector < std::string > strs_2;
            boost::split (strs_2, line, boost::is_any_of (","));
            if (strs_2.size() > 2 && strs_2[2].length())
                continue;

            const std::string patrol_run = strs_2[0];
            const std::string obj = strs_2[1];
            std::map<std::string, std::vector<std::string> >::iterator it = prun2ob.find(obj);
            if (it != prun2ob.end() )
            {
                it->second.push_back(patrol_run);
            }
            else
            {
                std::vector<std::string> pr_tmp;
                pr_tmp.push_back(patrol_run);
                prun2ob[obj] = pr_tmp;
            }
        }

//        r_.initialize();
        r_.idx_flann_fn_shot_ = "/tmp/shot_flann.idx";
        r_.idx_flann_fn_sift_ = "/tmp/sift_flann.idx";
        ofstream param_file;
        param_file.open ((out_dir_ + "/param.nfo").c_str());
        r_.printParams(param_file);
        param_file.close();
        r_.printParams();
        return true;
    }

    bool eval()
    {
        std::vector< std::string> dol_rec_folders, dol_model_files, offline_rec_folders, offline_model_files;
        v4r::io::getFoldersInDirectory( dol_rec_dir_ + "/recognition_structure", "", dol_rec_folders);
        v4r::io::getFilesInDirectory( dol_rec_dir_ + "/models", dol_model_files, "", ".*.pcd", false);
        v4r::io::getFoldersInDirectory( offline_rec_dir_ + "/recognition_structure", "", offline_rec_folders);
        v4r::io::getFilesInDirectory( offline_rec_dir_ + "/models", offline_model_files, "", ".*.pcd", false);

        std::sort(dol_rec_folders.begin(), dol_rec_folders.end());
        for (size_t dol_rec_id=0; dol_rec_id < dol_rec_folders.size(); dol_rec_id++)
        {
            std::string patrol_run = dol_rec_folders[ dol_rec_id ];
            boost::replace_all (patrol_run, "_object.pcd", "");

            std::string obj;

            // find object belonging to patrol_run
            bool have_found = false;
            std::map<std::string, std::vector<std::string> >::iterator it;
            for ( it = prun2ob.begin(); it != prun2ob.end(); it++)
            {
                for (size_t pr=0; pr<it->second.size();pr++)
                {
                    if ( it->second[pr].compare( patrol_run ) == 0 )
                    {
                         obj = it->first;
                         have_found = true;
                         break;
                    }
                }
                if(have_found)
                    break;
            }


            boost::filesystem::remove_all(boost::filesystem::path(training_dir_tmp_ + "/models/"));
            boost::filesystem::remove_all(boost::filesystem::path(training_dir_tmp_ + "/recognition_structure/"));
            v4r::io::createDirIfNotExist(training_dir_tmp_ + "/models");
            v4r::io::createDirIfNotExist(training_dir_tmp_ + "/recognition_structure");
            std::string src = dol_rec_dir_ + "/recognition_structure/" + patrol_run + "_object.pcd";
            std::string dst = training_dir_tmp_ + "/recognition_structure/" + patrol_run + ".pcd";
            copyDir( boost::filesystem::path(src), boost::filesystem::path(dst));
            boost::filesystem::copy_file( boost::filesystem::path(dol_rec_dir_ + "/models/" + patrol_run + "_object.pcd"),
                                          boost::filesystem::path(training_dir_tmp_ + "/models/" + patrol_run + ".pcd"));

            for (size_t i=0; i<offline_model_files.size(); i++)
            {
                const std::string ofm = offline_model_files[i];
                if ( ofm.compare(obj) ) // copy only other objects
                {
                    src = offline_rec_dir_ + "/recognition_structure/" + ofm;
                    dst = training_dir_tmp_ + "/recognition_structure/" + ofm;
                    copyDir( boost::filesystem::path(src), boost::filesystem::path(dst));
                    boost::filesystem::copy_file( boost::filesystem::path(offline_rec_dir_ + "/models/" + ofm),
                                                  boost::filesystem::path(training_dir_tmp_ + "/models/" + ofm));

                }
            }


            r_.models_dir_ = training_dir_tmp_ + "/models";
            r_.sift_structure_ = training_dir_tmp_ + "/recognition_structure";
            r_.training_dir_sift_ = "/tmp/sift_trained";
            r_.training_dir_shot_ = "/tmp/shot_trained";
            r_.training_dir_ourcvfh_ = "/tmp/ourcvfh_trained";
            boost::filesystem::remove_all(boost::filesystem::path(r_.training_dir_sift_));
            boost::filesystem::remove_all(boost::filesystem::path(r_.training_dir_shot_));
            boost::filesystem::remove_all(boost::filesystem::path(r_.training_dir_ourcvfh_));
            boost::filesystem::remove(boost::filesystem::path(r_.idx_flann_fn_sift_));
            boost::filesystem::remove(boost::filesystem::path(r_.idx_flann_fn_shot_));

            r_.initialize();

            it = prun2ob.find(obj);
            for (size_t j=0; j<it->second.size(); j++)
            {
                if (it->second[j].compare(patrol_run) == 0 ) // training = test
                    continue;

                const std::string test_sequence = test_dir_ + "/" + it->second[j];

                std::vector<std::string> views;
                v4r::io::getFilesInDirectory( test_sequence, views, "", ".*.pcd", false);
                std::sort(views.begin(), views.end());
                for(size_t v_id=0; v_id < views.size(); v_id++)
                {
                    const std::string fn = test_sequence + "/" + views[ v_id ];

                    std::cout << "Recognizing file " << fn << std::endl;
                    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
                    pcl::io::loadPCDFile(fn, *cloud);
                    r_.setInputCloud(cloud);
                    r_.recognize();

                    std::vector<ModelTPtr> verified_models;
                    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_verified;
                    r_.getModelsAndTransforms(verified_models, transforms_verified);
                    rec_models_per_id_.clear();

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
                        out_fn << out_dir_ << "/" << patrol_run << "/" << it->second[j];
                        v4r::io::createDirIfNotExist(out_fn.str());
                        out_fn << "/" << views[v_id].substr(0, views[v_id].length()-4) << "_"
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


//            const std::string sequence_path = test_dir_ + "/" + sub_folder_names[ sub_folder_id ];
//            const std::string out_path = out_dir_ + "/" + sub_folder_names[ sub_folder_id ];
//            v4r::io::createDirIfNotExist(out_path);

//            rec_models_per_id_.clear();

//            std::vector< std::string > views;
//            v4r::io::getFilesInDirectory(sequence_path, views, "", ".*.pcd", false);
//            std::sort(views.begin(), views.end());
//            for (size_t v_id=0; v_id<views.size(); v_id++)
//            {
//                const std::string fn = test_dir_ + "/" + sub_folder_names[sub_folder_id] + "/" + views[ v_id ];

//                std::cout << "Recognizing file " << fn << std::endl;
//                pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
//                pcl::io::loadPCDFile(fn, *cloud);
//                r_.setInputCloud(cloud);
//                r_.recognize();

//                std::vector<ModelTPtr> verified_models;
//                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_verified;
//                r_.getModelsAndTransforms(verified_models, transforms_verified);
//                if (visualize_)
//                    visualize_result(cloud, verified_models, transforms_verified);

//                for(size_t m_id=0; m_id<verified_models.size(); m_id++)
//                {
//                    std::cout << "********************" << verified_models[m_id]->id_ << std::endl;

//                    const std::string model_id = verified_models[m_id]->id_;
//                    const Eigen::Matrix4f tf = transforms_verified[m_id];

//                    size_t num_models_per_model_id;

//                    std::map<std::string, size_t>::iterator it_rec_mod;
//                    it_rec_mod = rec_models_per_id_.find(model_id);
//                    if(it_rec_mod == rec_models_per_id_.end())
//                    {
//                        rec_models_per_id_.insert(std::pair<std::string, size_t>(model_id, 1));
//                        num_models_per_model_id = 0;
//                    }
//                    else
//                    {
//                        num_models_per_model_id = it_rec_mod->second;
//                        it_rec_mod->second++;
//                    }

//                    std::stringstream out_fn;
//                    out_fn << out_path << "/" << views[v_id].substr(0, views[v_id].length()-4) << "_"
//                           << model_id.substr(0, model_id.length() - 4) << "_" << num_models_per_model_id << ".txt";

//                    ofstream or_file;
//                    or_file.open (out_fn.str().c_str());
//                    for (size_t row=0; row <4; row++)
//                    {
//                        for(size_t col=0; col<4; col++)
//                        {
//                            or_file << tf(row, col) << " ";
//                        }
//                    }
//                    or_file.close();
//                }
//            }
        }
        return true;
    }
};

int
main (int argc, char ** argv)
{
    srand (time(NULL));
    EvalSvRecognizer r_eval;
    r_eval.initialize(argc,argv);
    r_eval.eval();
    return 0;
}

