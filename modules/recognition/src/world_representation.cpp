#include "include/world_representation.h"
#include "include/boost_graph_extension.h"
#include <iostream>
#include <fstream>

namespace v4r
{
//multiviewGraph& worldRepresentation::get_current_graph(const std::string &scene_name)
//{
//    std::map<std::string, v4r::multiviewGraph>::iterator it = mv_environment_.find(scene_name);
//    if ( it != mv_environment_.end() )
//         return it->second;
//    else
//    {
//        multiviewGraph newGraph;
//        newGraph.setVisualize_output(visualize_output_);
//    //    newGraph.setIcp_iter(icp_iter_);
////        newGraph.setChop_at_z(chop_at_z_);
////        newGraph.set_scene_name(scene_name);
////        newGraph.setPSingleview_recognizer(pSingleview_recognizer_);
////        newGraph.setSift(sift_);
////        newGraph.set_scene_to_scene(scene_to_scene_);
////        newGraph.set_extension_mode(extension_mode_);
////        newGraph.set_max_vertices_in_graph(max_vertices_in_graph_);
////        newGraph.set_distance_keypoints_get_discarded(distance_keypoints_get_discarded_);
////        graph_v_.push_back(newGraph);

//        mv_environment_[scene_name] = newGraph;
//        return mv_environment_[scene_name];
//    }

////    for (size_t scene_id = 0; scene_id < graph_v_.size(); scene_id++)
////    {
////        if( graph_v_[scene_id].get_scene_name().compare ( scene_name) == 0 )	//--show-hypotheses-from-single-view
////        {
////            return graph_v_[scene_id];
////        }
////    }

////    multiviewGraph newGraph;
////    newGraph.setVisualize_output(visualize_output_);
//////    newGraph.setIcp_iter(icp_iter_);
////    newGraph.setChop_at_z(chop_at_z_);
////    newGraph.set_scene_name(scene_name);
////    newGraph.setPSingleview_recognizer(pSingleview_recognizer_);
////    newGraph.setSift(sift_);
////    newGraph.set_scene_to_scene(scene_to_scene_);
////    newGraph.set_extension_mode(extension_mode_);
////    newGraph.set_max_vertices_in_graph(max_vertices_in_graph_);
////    newGraph.set_distance_keypoints_get_discarded(distance_keypoints_get_discarded_);
////    newGraph.set_visualize_output(visualize_output_);
////    graph_v_.push_back(newGraph);
////    return graph_v_.back();
//}

//void worldRepresentation::setPSingleview_recognizer(const boost::shared_ptr<Recognizer> &value)
//{
//    pSingleview_recognizer_ = value;
//}


bool worldRepresentation::recognize (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &pInput,
                                     const std::string &scene_name,
                                     const std::string &view_name,
                                     const size_t &timestamp,
                                     const std::vector<double> &global_trans_v,
                                     std::vector<ModelTPtr> &models_mv,
                                     std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transforms_mv,
                                     const std::string &filepath_or_results_mv,
                                     const std::string &filepath_or_results_sv)
{
    bool mv_recognize_error;

    MultiviewRecognizer &currentMvGraph = get_current_graph(scene_name);

    Eigen::Matrix4f global_trans;
    if(global_trans_v.size() == 16 && use_robot_pose_)
    {
        for (size_t row=0; row <4; row++)
        {
            for(size_t col=0; col<4; col++)
            {
                global_trans(row, col) = global_trans_v[4*row + col];
            }
        }
        mv_recognize_error = currentMvGraph.recognize(pInput, view_name, global_trans);//req, response);
    }
    else
    {
        std::cout << "No transform (16x1 float vector) provided. " << std::endl;
        mv_recognize_error = currentMvGraph.recognize(pInput, view_name);
    }


    std::vector<ModelTPtr> models_sv;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_sv;
    std::vector<double> execution_times;
    currentMvGraph.getVerifiedHypothesesSingleView(models_sv, transforms_sv);

    currentMvGraph.getVerifiedHypotheses(models_mv, transforms_mv);
    currentMvGraph.get_times(execution_times);

    std::cout << "In the most current vertex I detected " << models_mv.size() << " verified models. " << std::endl;

    if(filepath_or_results_mv.length())
    {
        ofstream annotation_file;
        annotation_file.open("results_3d.txt"); // save a file for ground-truth annotation structure
        std::map<std::string, size_t> rec_models_per_id;

        for(size_t i = 0; i < models_mv.size(); i++)
        {
            std::string model_id = models_mv.at(i)->id_;
            Eigen::Matrix4f tf = transforms_mv[i];

            size_t num_models_per_model_id;

            std::map<std::string, size_t>::iterator it_rec_mod;
            it_rec_mod = rec_models_per_id.find(model_id);
            if(it_rec_mod == rec_models_per_id.end())
            {
                rec_models_per_id.insert(std::pair<std::string, size_t>(model_id, 1));
                num_models_per_model_id = 0;
            }
            else
            {
                num_models_per_model_id = it_rec_mod->second;
                it_rec_mod->second++;
            }

            // Save multiview object recogniton result in file

            std::stringstream or_filepath_ss_mv;
            or_filepath_ss_mv << filepath_or_results_mv << "/" << view_name << "_" << model_id.substr(0, model_id.length() - 4) << "_" << num_models_per_model_id <<".txt";

            ofstream or_file;
            or_file.open (or_filepath_ss_mv.str().c_str());
            for (size_t row=0; row <4; row++)
            {
                for(size_t col=0; col<4; col++)
                {
                    or_file << tf(row, col) << " ";
                }
            }
            or_file.close();


            annotation_file << model_id << " ";
            for (size_t row=0; row <4; row++)
            {
                for(size_t col=0; col<4; col++)
                {
                    annotation_file << tf(row, col) << " ";
                }
            }
            annotation_file << std::endl;
        }
        annotation_file.close();

        // save measured execution times
        std::stringstream or_filepath_times;
        or_filepath_times << filepath_or_results_mv << "/" << view_name << "_times.txt";

        ofstream time_file;
        time_file.open (or_filepath_times.str().c_str());
        for (size_t time_id=0; time_id < execution_times.size(); time_id++)
        {
            time_file << execution_times[time_id] << std::endl;
        }
        time_file.close();
     }


    if (filepath_or_results_sv.length())
    {
        std::map<std::string, size_t> rec_models_per_id_sv;
        for(size_t i = 0; i < models_sv.size(); i++)
        {
            std::string model_id = models_sv.at(i)->id_;
            Eigen::Matrix4f tf = transforms_sv[i];

            size_t num_models_per_model_id;

            std::map<std::string, size_t>::iterator it_rec_mod;
            it_rec_mod = rec_models_per_id_sv.find(model_id);
            if(it_rec_mod == rec_models_per_id_sv.end())
            {
                rec_models_per_id_sv.insert(std::pair<std::string, size_t>(model_id, 1));
                num_models_per_model_id = 0;
            }
            else
            {
                num_models_per_model_id = it_rec_mod->second;
                it_rec_mod->second++;
            }

            // Save single view object recogniton result in file
            // Save multiview object recogniton result in file

            std::stringstream or_filepath_ss_sv;
            or_filepath_ss_sv << filepath_or_results_sv << "/" << view_name << "_" << model_id.substr(0, model_id.length() - 4) << "_" << num_models_per_model_id <<".txt";

            ofstream or_file;
            or_file.open (or_filepath_ss_sv.str().c_str());
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
    return mv_recognize_error;
}



std::string worldRepresentation::models_dir() const
{
    return models_dir_;
}

void worldRepresentation::setModels_dir(const std::string &models_dir)
{
    models_dir_ = models_dir;
}
}
