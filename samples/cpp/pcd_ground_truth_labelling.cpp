/*
 *  Created on: Aug, 08, 2014
 *      Author: Thomas Faeulhammer
 */
#include <iostream>
#include <fstream>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <v4r/recognition/model_only_source.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/common/occlusion_reasoning.h>
#include <v4r/common/pcl_visualization_utils.h>
#include <v4r/io/filesystem.h>
#include <v4r/io/eigen.h>

//-gt_dir /media/Data/datasets/TUW/annotations -output_dir /home/thomas/Desktop/test -scenes_dir /media/Data/datasets/TUW/test_set_static/ -models_dir /media/Data/datasets/TUW/models -threshold 0.01 -visualize 0

namespace v4r
{

template<typename PointT>
class PcdGtAnnotator
{
private:
    typedef v4r::Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;
    typename pcl::PointCloud<PointT>::Ptr pScenePCl_;
    boost::shared_ptr < v4r::ModelOnlySource<pcl::PointXYZRGBNormal, PointT> > source_;
    float f_;
    pcl::visualization::PCLVisualizer::Ptr vis_;
    std::vector<int> viewports_;

    std::vector< std::vector<bool> > visible_model_points_;
    std::vector< std::string > model_id_;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transform_to_scene_;
    std::vector<std::vector<bool> > pixel_annotated_obj_in_first_view_;

public:
    std::string gt_dir_;
    std::string models_dir_;
    float threshold_;

    PcdGtAnnotator()
    {
        f_ = 525.f;
        source_.reset(new v4r::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>);
        models_dir_ = "";
        gt_dir_ = "";
        threshold_ = 0.01f;
        pScenePCl_.reset(new pcl::PointCloud<PointT>);
    }

    void init_source()
    {
        source_->setPath (models_dir_);
        source_->setLoadViews (false);
        source_->setLoadIntoMemory(false);
        std::string test = "irrelevant";
        source_->generate (test);
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

    std::vector < std::string > scene_view;
    if (v4r::io::getFilesInDirectory( scene_full_path, scene_view, "", ".*.pcd", true) != -1)
    {
        std::cout << "Number of viewpoints in directory is:" << scene_view.size () << std::endl;

        std::vector < std::string > gt_file;
        std::string annotations_dir = gt_dir_ + "/" + scene_id;

        if( v4r::io::getFilesInDirectory( annotations_dir, gt_file, "", ".*.txt", true) == -1)
        {
            std::cerr << "Could not find any annotations in " << annotations_dir << ". " << std::endl;
        }
        for (size_t s_id = 0; s_id < scene_view.size (); s_id++)
        {
            std::vector<std::string> strs;
            boost::split (strs, scene_view[s_id], boost::is_any_of ("."));
            std::string scene_file_wo_ext = strs[0];
            std::string gt_occ_check = scene_file_wo_ext + "_occlusion_";

            std::string scene_full_file_path = scene_full_path + "/" + scene_view[s_id];
            pcl::io::loadPCDFile(scene_full_file_path, *pScenePCl_);

            for(size_t gt_id=0; gt_id < gt_file.size(); gt_id++)
            {
                const std::string gt_fn = gt_file[gt_id];
                if( gt_fn.compare(0, scene_file_wo_ext.size(), scene_file_wo_ext) == 0 &&
                        gt_fn.compare(0, gt_occ_check.size(), gt_occ_check))
                {
                    std::cout << gt_fn << std::endl;
                    std::string model_instance = gt_fn.substr(scene_file_wo_ext.size() + 1);
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
                    std::string model_name = model_instance.substr(0,found) + ".pcd";
                    std::cout << "Model: " << model_name << std::endl;
                    ModelTPtr pModel;
                    source_->getModelById(model_name, pModel);

                    std::string gt_full_file_path = annotations_dir + "/" + gt_fn;
                    Eigen::Matrix4f transform;
                    v4r::io::readMatrixFromFile(gt_full_file_path, transform);

                    typename pcl::PointCloud<PointT>::ConstPtr model_cloud = pModel->getAssembled(0.003f);
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
                        std::vector<bool> obj_mask (pScenePCl_->points.size(), false);
                        pixel_annotated_obj_in_first_view_.push_back ( obj_mask );
                        current_model_id = model_id_.size()-1;
                    }


                    const float cx = (static_cast<float> (pScenePCl_->width) / 2.f - 0.5f);
                    const float cy = (static_cast<float> (pScenePCl_->height) / 2.f - 0.5f);

                    for(size_t m_pt_id=0; m_pt_id < model_aligned->points.size(); m_pt_id++)
                    {
                        PointT mp = model_aligned->points[m_pt_id];
                        const int u = static_cast<int> (f_ * mp.x / mp.z + cx);
                        const int v = static_cast<int> (f_ * mp.y / mp.z + cy);

                        if(u<0 || u >= pScenePCl_->width || v<0 || v >= pScenePCl_->height) // model point outside field of view
                            continue;

                        PointT sp = pScenePCl_->at(u,v);

                        if ( !pcl::isFinite(sp) )   // Model point is not visible from the view point (shiny spot, noise,...)
                            continue;

                        if( (mp.z - threshold_ - sp.z) < 0 )
                        {
                            visible_model_points_[current_model_id][m_pt_id] = true;
                            if( s_id == 0 )
                                pixel_annotated_obj_in_first_view_[current_model_id][v*pScenePCl_->width + u] = true;
                        }
                    }
                }
            }
            std::cout << std::endl;
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
    {
        vis_.reset ( new pcl::visualization::PCLVisualizer ("ground truth model", true));
    }

    std::vector<std::string> subwindow_titles;
    subwindow_titles.push_back( "scene" );

    for (size_t m_id=0; m_id < model_id_.size(); m_id++)
    {
        subwindow_titles.push_back ( model_id_[m_id] );
    }
    std::vector<int> viewports = v4r::common::pcl_visualizer::visualization_framework(vis_, 1, model_id_.size() + 1 , subwindow_titles);

    size_t vp_id = 0;

    Eigen::Vector4f zero_origin;
    zero_origin[0] = zero_origin[1] = zero_origin[2] = zero_origin[3] = 0.f;

    pScenePCl_->sensor_origin_ = zero_origin;   // for correct visualization
    pScenePCl_->sensor_orientation_ = Eigen::Quaternionf::Identity();
    vis_->addPointCloud(pScenePCl_, "scene", viewports[vp_id++]);
    for (size_t m_id=0; m_id < model_id_.size(); m_id++)
    {
        size_t found = model_id_[m_id].find_last_of("_");
        std::string model_name = model_id_[m_id].substr(0,found) + ".pcd";
        ModelTPtr pModel;
        source_->getModelById( model_name, pModel );
        typename pcl::PointCloud<PointT>::Ptr visible_model ( new pcl::PointCloud<PointT>() );
        typename pcl::PointCloud<PointT>::Ptr visible_model_aligned ( new pcl::PointCloud<PointT>() );
        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = pModel->getAssembled( 0.003f );
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
}


template<typename PointT>
void PcdGtAnnotator<PointT>::save_to_disk(const std::string &path)
{
    bf::path path_bf = path;

    if(!bf::exists(path_bf))
        bf::create_directory(path_bf);

    for (size_t m_id=0; m_id < model_id_.size(); m_id++)
    {
        size_t found = model_id_[m_id].find_last_of("_");
        std::string model_name = model_id_[m_id].substr(0,found) + ".pcd";
        ModelTPtr pModel;
        source_->getModelById( model_name, pModel );
        typename pcl::PointCloud<PointT>::Ptr visible_model ( new pcl::PointCloud<PointT>() );
        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = pModel->getAssembled( 0.003f );
        size_t num_vis_pts = 0;
        for (size_t v_id=0; v_id<visible_model_points_[m_id].size(); v_id++)
        {
            num_vis_pts++;
        }
        std::cout << num_vis_pts << " visible points of total " << visible_model_points_[m_id].size() << std::endl;
        pcl::copyPointCloud( *model_cloud, visible_model_points_[ m_id ], *visible_model);
        v4r::common::setCloudPose(transform_to_scene_[m_id], *visible_model);
        pcl::io::savePCDFileBinary( path + "/" + model_id_[m_id] + ".pcd", *visible_model);

        if( pixel_annotated_obj_in_first_view_.size() > m_id )
        {
            std::vector<size_t> obj_mask_in_first_frame = v4r::common::createIndicesFromMask( pixel_annotated_obj_in_first_view_ [m_id] );
            std::ofstream mask;
            mask.open( (path + "/" + model_name + "_mask.txt").c_str() );
            for(size_t i=0; i < obj_mask_in_first_frame.size(); i++)
            {
                mask << obj_mask_in_first_frame[i] << std::endl;
            }
            mask.close();
        }
    }
}


template<typename PointT>
void PcdGtAnnotator<PointT>::printUsage(int argc, char ** argv)
{
    (void)argc;
    std::cout << std::endl << std::endl
              << "Usage " << argv[0]
              << "-models_dir /path/to/models/ "
              << "-gt_dir /path/to/annotations/ "
              << "-scenes_dir /path/to/input_PCDs/ "
              << "-output_dir /path/to/output/ "
              << "[-visualize 1] "
              << std::endl << std::endl;
}
}



int
main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PointT;
    v4r::PcdGtAnnotator<PointT> annotator;
    std::string scene_dir, output_dir;
    bool visualize = false;

    pcl::console::parse_argument (argc, argv, "-scenes_dir", scene_dir);
    pcl::console::parse_argument (argc, argv, "-output_dir", output_dir);
    pcl::console::parse_argument (argc, argv, "-models_dir", annotator.models_dir_);
    pcl::console::parse_argument (argc, argv, "-gt_dir", annotator.gt_dir_);
    pcl::console::parse_argument (argc, argv, "-visualize", visualize);
    pcl::console::parse_argument (argc, argv, "-threshold", annotator.threshold_);


    if (scene_dir.compare ("") == 0)
    {
        PCL_ERROR("Set the directory containing scenes. Usage -pcd_file files [dir].\n");
        v4r::PcdGtAnnotator<PointT>::printUsage(argc, argv);
        return -1;
    }

    if (output_dir.compare ("") == 0)
    {
        PCL_ERROR("Set the directory for saving the models using the -output_dir [dir] option\n");
        v4r::PcdGtAnnotator<PointT>::printUsage(argc, argv);
        return -1;
    }

    bf::path models_dir_path = annotator.models_dir_;
    if (!bf::exists (models_dir_path))
    {
        PCL_ERROR("Models dir path %s does not exist, use -models_dir [dir] option\n", annotator.models_dir_.c_str());
        v4r::PcdGtAnnotator<PointT>::printUsage(argc, argv);
        return -1;
    }

    std::vector< std::string> sub_folder_names;
    if(!v4r::io::getFoldersInDirectory( scene_dir, "", sub_folder_names) )
    {
        std::cerr << "No subfolders in directory " << scene_dir << ". " << std::endl;
        sub_folder_names.push_back("");
    }

    annotator.init_source();
    for (size_t sub_folder_id=0; sub_folder_id < sub_folder_names.size(); sub_folder_id++)
    {
        annotator.annotate (scene_dir, sub_folder_names[ sub_folder_id ]);
        if(visualize)
            annotator.visualize();
        annotator.save_to_disk(output_dir + "/" + sub_folder_names[ sub_folder_id ]);
        annotator.clear();
    }
}

