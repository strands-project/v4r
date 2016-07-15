/*
 *  Created on: Aug, 08, 2014
 *      Author: Thomas Faeulhammer
 */
#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <v4r/recognition/model_only_source.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/common/zbuffering.h>
#include <v4r/common/pcl_visualization_utils.h>
#include <v4r/io/filesystem.h>
#include <v4r/io/eigen.h>
#include <time.h>
#include <stdlib.h>

namespace po = boost::program_options;

//-g /media/Data/datasets/TUW/annotations -o /tmp/bla -s /media/Data/datasets/TUW/test_set_static/ -m /media/Data/datasets/TUW/models -t 0.01 -v
//-g /media/Data/datasets/willow/annotations -o /tmp/bla -s /media/Data/datasets/willow_dataset_new/willow_large_dataset -m /media/Data/datasets/willow_dataset/models -t 0.01 -v


namespace v4r
{

template<typename PointT>
class PcdGtAnnotator
{
public:
    class View{
    public:
        typename pcl::PointCloud<PointT>::Ptr cloud_;
        std::vector< std::string > model_id_;
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transform_to_scene_;
        View()
        {
            cloud_.reset(new pcl::PointCloud<PointT>());
        }
    };

private:
    typedef Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;
    typename pcl::PointCloud<PointT>::Ptr reconstructed_scene_;
    boost::shared_ptr < v4r::ModelOnlySource<pcl::PointXYZRGBNormal, PointT> > source_;
    pcl::visualization::PCLVisualizer::Ptr vis_;
    std::vector<int> viewports_;

    std::vector< std::vector<bool> > visible_model_points_;
    std::vector< std::string > model_id_;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transform_to_scene_;
    std::vector<std::vector<bool> > pixel_annotated_obj_in_first_view_;

    std::vector<std::vector< std::vector<bool> > > is_pixel_annotated_for_model_in_view_; /// @brief outer loop: model id, middle loop: view id, inner loop: pixel id
    std::vector<View> views_;

public:
    std::string gt_dir_;
    std::string models_dir_;
    std::string out_dir_;
    float threshold_;
    float f_;
    bool first_view_only_;  // has been used for RA-L 16 paper, where we were interested in the visible object in the first viewpoint

    size_t unique_id_;

    PcdGtAnnotator()
    {
        f_ = 525.f;
        source_.reset(new v4r::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>);
        models_dir_ = "";
        gt_dir_ = "";
        threshold_ = 0.01f;
        first_view_only_ = false;
        reconstructed_scene_.reset(new pcl::PointCloud<PointT>);
        out_dir_ = "/tmp/annotated_pcds";

        unique_id_ = 0;
    }

    void init_source()
    {
        source_->setPath (models_dir_);
        source_->setLoadViews (false);
        source_->setLoadIntoMemory(false);
        source_->generate ();
    }

    void annotate(const std::string &scenes_dir, const std::string & scene_id = "");
    void visualize();
    void clear();
    void save_to_disk(const std::string &path);
    static void printUsage(int argc, char ** argv);
};

template<typename PointT>
void PcdGtAnnotator<PointT>::annotate (const std::string &scenes_dir, const std::string &scene_id)
{
    std::string scene_full_path = scenes_dir + "/" + scene_id;

    std::vector < std::string > scene_view = v4r::io::getFilesInDirectory( scene_full_path, ".*.pcd", true);
    std::sort(scene_view.begin(), scene_view.end());

    if ( !scene_view.empty() )
    {
        std::cout << "Number of viewpoints in directory is:" << scene_view.size () << std::endl;

        std::string annotations_dir = gt_dir_ + "/" + scene_id;
        std::vector < std::string > gt_file = v4r::io::getFilesInDirectory( annotations_dir, ".*.txt", true);

        if( gt_file.empty())
        {
            std::cerr << "Could not find any annotations in " << annotations_dir << ". " << std::endl;
            return;
        }

        for (size_t s_id = 0; s_id < scene_view.size (); s_id++)
        {
            if (first_view_only_ && s_id > 0)
                    continue;

            std::vector<std::string> strs;
            boost::split (strs, scene_view[s_id], boost::is_any_of ("."));
            std::string scene_file_wo_ext = strs[0] + "_";
            std::string gt_occ_check = scene_file_wo_ext + "occlusion_";

            std::string scene_full_file_path = scene_full_path + "/" + scene_view[s_id];
            pcl::io::loadPCDFile(scene_full_file_path, *reconstructed_scene_);
            View view;
            pcl::copyPointCloud(*reconstructed_scene_, *view.cloud_);
            view.cloud_->sensor_origin_ = reconstructed_scene_->sensor_origin_;
            view.cloud_->sensor_orientation_ = reconstructed_scene_->sensor_orientation_;

            for(size_t gt_id=0; gt_id < gt_file.size(); gt_id++)
            {
                const std::string gt_fn = gt_file[gt_id];
                if( gt_fn.compare(0, scene_file_wo_ext.size(), scene_file_wo_ext) == 0 &&
                        gt_fn.compare(0, gt_occ_check.size(), gt_occ_check))
                {
                    std::cout << gt_fn << std::endl;
                    std::string model_instance = gt_fn.substr(scene_file_wo_ext.size());
                    std::vector<std::string> str_split;
                    boost::split (str_split, model_instance, boost::is_any_of ("."));
                    model_instance = str_split[0];

                    size_t found = model_instance.find_last_of("_");
                    std::string times_text ("times.txt");
                    if ( model_instance.compare(times_text) == 0 )
                    {
                        std::cout << "skipping this one" << std::endl;
                        continue;
                    }
                    std::string model_name = model_instance.substr(0,found);
                    std::cout << "Model: " << model_name << std::endl;
                    ModelTPtr pModel;
                    source_->getModelById(model_name, pModel);

                    std::string gt_full_file_path = annotations_dir + "/" + gt_fn;
                    Eigen::Matrix4f transform = v4r::io::readMatrixFromFile(gt_full_file_path);

                    typename pcl::PointCloud<PointT>::ConstPtr model_cloud = pModel->getAssembled(3);
                    typename pcl::PointCloud<PointT>::Ptr model_aligned(new pcl::PointCloud<PointT>());
                    pcl::transformPointCloud(*model_cloud, *model_aligned, transform);

                    bool model_exists = false;
                    size_t current_model_id = 0;
                    for (size_t m_id=0; m_id < model_id_.size(); m_id++)
                    {
                        if ( model_id_[m_id].compare( model_instance ) == 0)
                        {
                            model_exists = true;
                            current_model_id = m_id;
                            transform_to_scene_[m_id] = transform;
                            break;
                        }
                    }
                    if ( !model_exists )
                    {
                        std::vector<bool> visible_model_pts_tmp (model_aligned->points.size(), false);
                        visible_model_points_.push_back(visible_model_pts_tmp);
                        model_id_.push_back( model_instance );
                        transform_to_scene_.push_back(transform);
                        std::vector<bool> obj_mask (reconstructed_scene_->points.size(), false);
                        pixel_annotated_obj_in_first_view_.push_back ( obj_mask );
                        current_model_id = model_id_.size()-1;
                    }

                    const float cx = (static_cast<float> (reconstructed_scene_->width) / 2.f - 0.5f);
                    const float cy = (static_cast<float> (reconstructed_scene_->height) / 2.f - 0.5f);

                    std::vector<bool> obj_mask_in_view (reconstructed_scene_->points.size(), false);

                    for(size_t m_pt_id=0; m_pt_id < model_aligned->points.size(); m_pt_id++)
                    {
                        PointT mp = model_aligned->points[m_pt_id];
                        int u = static_cast<int> (f_ * mp.x / mp.z + cx);
                        int v = static_cast<int> (f_ * mp.y / mp.z + cy);

                        if(u<0 || u >= (int)reconstructed_scene_->width || v<0 || v >= (int)reconstructed_scene_->height) // model point outside field of view
                            continue;

                        int px_id = v*reconstructed_scene_->width + u;

                        PointT sp = reconstructed_scene_->at(u,v);

                        if ( !pcl::isFinite(sp) )   // Model point is not visible from the view point (shiny spot, noise,...)
                            continue;

                        if( std::abs(mp.z - sp.z) < threshold_ )
                        {
                            visible_model_points_[current_model_id][m_pt_id] = true;

                            obj_mask_in_view[px_id] = true;

                            if( s_id == 0 )
                                pixel_annotated_obj_in_first_view_[current_model_id][px_id] = true;
                        }
                    }

                    std::stringstream out_fn; out_fn << out_dir_ << "/" << setfill('0') << setw(10) << unique_id_ << ".txt";
                    v4r::io::createDirForFileIfNotExist(out_fn.str());
                    std::ofstream obj_indices_per_view_f ( out_fn.str().c_str() );
                    for(size_t i=0; i< obj_mask_in_view.size(); i++)
                    {
                        if(obj_mask_in_view[i])
                            obj_indices_per_view_f << i << std::endl;
                    }
                    obj_indices_per_view_f.close();

                    std::ofstream label_file;
                    label_file.open( out_dir_ + "/label.anno", std::ofstream::out | std::ofstream::app);
                    label_file << scenes_dir << " " << scene_id << " " << scene_view[s_id] << " " << setfill('0') << setw(10) << unique_id_ << " " << model_name << std::endl;
                    label_file.close();
                    unique_id_ ++;
                }
            }
            std::cout << std::endl;

            view.transform_to_scene_ = transform_to_scene_;
            views_.push_back(view);
        }
    }
    else
    {
        PCL_ERROR("You should pass a directory\n");
        return;
    }
}

template<typename PointT>
void PcdGtAnnotator<PointT>::visualize()
{
    if(!vis_)
        vis_.reset ( new pcl::visualization::PCLVisualizer ("ground truth model", true));

    std::vector<std::string> subwindow_titles;
    subwindow_titles.push_back( "scene" );

    for (size_t m_id=0; m_id < model_id_.size(); m_id++)
        subwindow_titles.push_back ( model_id_[m_id] );

    std::vector<int> viewports = v4r::pcl_visualizer::visualization_framework(*vis_, 1, model_id_.size() + 1 , subwindow_titles);

    size_t vp_id = 0;

    Eigen::Vector4f zero_origin;
    zero_origin[0] = zero_origin[1] = zero_origin[2] = zero_origin[3] = 0.f;

    reconstructed_scene_->sensor_origin_ = zero_origin;   // for correct visualization
    reconstructed_scene_->sensor_orientation_ = Eigen::Quaternionf::Identity();
    vis_->addPointCloud(reconstructed_scene_, "scene", viewports[vp_id++]);
    for (size_t m_id=0; m_id < model_id_.size(); m_id++)
    {
        size_t found = model_id_[m_id].find_last_of("_");
        std::string model_name = model_id_[m_id].substr(0,found);
        ModelTPtr pModel;
        source_->getModelById( model_name, pModel );
        typename pcl::PointCloud<PointT>::Ptr visible_model ( new pcl::PointCloud<PointT>() );
        typename pcl::PointCloud<PointT>::Ptr visible_model_aligned ( new pcl::PointCloud<PointT>() );
        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = pModel->getAssembled( 3 );
        size_t num_vis_pts = 0;
        for (size_t v_id=0; v_id<visible_model_points_[m_id].size(); v_id++)
        {
            num_vis_pts++;
        }
        std::cout << num_vis_pts << " visible points of total " << visible_model_points_[m_id].size() << std::endl;
        pcl::copyPointCloud( *model_cloud, visible_model_points_[ m_id ], *visible_model);
        pcl::transformPointCloud( *visible_model, *visible_model_aligned, transform_to_scene_ [m_id]);
        vis_->addPointCloud( visible_model_aligned, model_id_[ m_id ], viewports[vp_id++]);
    }
    vis_->spin();
}


template<typename PointT>
void PcdGtAnnotator<PointT>::clear ()
{
    visible_model_points_.clear();
    model_id_.clear();
    transform_to_scene_.clear();
    views_.clear();
    reconstructed_scene_.reset(new pcl::PointCloud<PointT>);
    pixel_annotated_obj_in_first_view_.clear();
}


template<typename PointT>
void PcdGtAnnotator<PointT>::save_to_disk(const std::string &path)
{
    bf::path path_bf = path;

    if(!bf::exists(path_bf))
        bf::create_directories(path);

//    pcl::visualization::PCLVisualizer vis("obj_mask");
//    vis.addPointCloud(views_[0].cloud_, "cloud");

    for (size_t m_id=0; m_id < model_id_.size(); m_id++)
    {
        size_t found = model_id_[m_id].find_last_of("_");
        std::string model_name = model_id_[m_id].substr(0,found);
        ModelTPtr pModel;
        source_->getModelById( model_name, pModel );
        typename pcl::PointCloud<PointT>::Ptr visible_model ( new pcl::PointCloud<PointT>() );
        typename pcl::PointCloud<PointT>::Ptr visible_model_aligned ( new pcl::PointCloud<PointT>() );
        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = pModel->getAssembled( 3 );
        size_t num_vis_pts = 0;
        for (size_t v_id=0; v_id<visible_model_points_[m_id].size(); v_id++)
        {
            num_vis_pts++;
        }
        std::cout << num_vis_pts << " visible points of total " << visible_model_points_[m_id].size() << std::endl;
        pcl::copyPointCloud( *model_cloud, visible_model_points_[ m_id ], *visible_model);
        pcl::transformPointCloud( *visible_model, *visible_model_aligned, transform_to_scene_ [m_id]);
        visible_model_aligned->sensor_orientation_ = views_.back().cloud_->sensor_orientation_;
        visible_model_aligned->sensor_origin_ = views_.back().cloud_->sensor_origin_;

        if(visible_model_aligned->points.empty())
            continue;

        pcl::io::savePCDFileBinary( path + "/" + model_id_[m_id] + ".pcd", *visible_model_aligned);

        if( pixel_annotated_obj_in_first_view_.size() > m_id )
        {
            std::ofstream obj_indices_f ( (path + "/" + model_id_[m_id] + "_mask.txt").c_str() );
            for(size_t i=0; i< pixel_annotated_obj_in_first_view_[m_id].size(); i++)
            {
                if(pixel_annotated_obj_in_first_view_[m_id][i])
                    obj_indices_f << i << std::endl;
            }
            obj_indices_f.close();
        }
    }
//    vis.spin();
}
}


int
main (int argc, char ** argv)
{
    srand (time(NULL));
    typedef pcl::PointXYZRGB PointT;
    v4r::PcdGtAnnotator<PointT> annotator;
    std::string scene_dir;
    bool visualize = false;

    po::options_description desc("Pixel-wise annotation of point clouds using recognition results represented as ground-truth object pose\n======================================\n **Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("scenes_dir,s", po::value<std::string>(&scene_dir)->required(), "directory containing the scene .pcd files")
            ("models_dir,m", po::value<std::string>(&annotator.models_dir_)->required(), "directory containing the model .pcd files")
            ("gt_dir,g", po::value<std::string>(&annotator.gt_dir_)->required(), "directory containing recognition results")
            ("output_dir,o", po::value<std::string>(&annotator.out_dir_)->default_value(annotator.out_dir_), "output directory")
            ("threshold,t", po::value<float>(&annotator.threshold_)->default_value(annotator.threshold_), "Threshold in m for a point to be counted as inlier")
            ("focal_length,f", po::value<float>(&annotator.f_)->default_value(annotator.f_), "Threshold in m for a point to be counted as inlier")
            ("visualize,v", po::bool_switch(&visualize), "turn visualization on")
            ("first_view_only", po::bool_switch(&annotator.first_view_only_), "if true, outputs the visible object model in the first view only")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return false;
    }

    try { po::notify(vm); }
    catch(std::exception& e) { std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl; return false;  }

    std::vector< std::string> sub_folder_names = v4r::io::getFoldersInDirectory( scene_dir );
    if( sub_folder_names.empty() )
    {
        std::cerr << "No subfolders in directory " << scene_dir << ". " << std::endl;
        sub_folder_names.push_back("");
    }

    std::sort(sub_folder_names.begin(), sub_folder_names.end());
    annotator.init_source();
    for (size_t sub_folder_id=0; sub_folder_id < sub_folder_names.size(); sub_folder_id++)
    {
        annotator.annotate (scene_dir, sub_folder_names[ sub_folder_id ]);

        if(visualize)
            annotator.visualize();

        annotator.save_to_disk(annotator.out_dir_ + "/" + sub_folder_names[ sub_folder_id ]);
        annotator.clear();
    }
}

