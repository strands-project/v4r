/*
 * local_recognition_mian_dataset.cpp
 *
 *  Created on: Mar 24, 2012
 *      Author: aitor
 */

#include <pcl/console/parse.h>
#include <faat_pcl/3d_rec_framework/pc_source/mesh_source.h>
#include <faat_pcl/3d_rec_framework/pipeline/local_recognizer.h>
#include <faat_pcl/3d_rec_framework/pipeline/hough_grouping_local_recognizer.h>
#include <faat_pcl/3d_rec_framework/pipeline/recognizer.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/shot_local_estimator_omp.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/recognition/cg/correspondence_grouping.h>
#include <faat_pcl/recognition/cg/graph_geometric_consistency.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/filters/passthrough.h>
#include <faat_pcl/3d_rec_framework/tools/or_evaluator.h>

float VX_SIZE_ICP_ = 0.005f;
bool PLAY_ = false;
std::string go_log_file_ = "test.txt";
float Z_DIST_ = 1.5f;
std::string GT_DIR_;
std::string MODELS_DIR_;
std::string MODELS_DIR_FOR_VIS_;
float model_scale = 1.f;
bool SHOW_GT_ = true;

inline void
getScenesInDirectory (bf::path & dir, std::string & rel_path_so_far, std::vector<std::string> & relative_paths)
{
    //list models in MODEL_FILES_DIR_ and return list
    bf::directory_iterator end_itr;
    for (bf::directory_iterator itr (dir); itr != end_itr; ++itr)
    {
        //check if its a directory, then get models in it
        if (bf::is_directory (*itr))
        {

#if BOOST_FILESYSTEM_VERSION == 3
            std::string so_far = rel_path_so_far + (itr->path ().filename ()).string () + "/";
#else
            std::string so_far = rel_path_so_far + (itr->path ()).filename () + "/";
#endif
            bf::path curr_path = itr->path ();
            getScenesInDirectory (curr_path, so_far, relative_paths);
        }
        else
        {
            //check that it is a ply file and then add, otherwise ignore..
            std::vector<std::string> strs;
#if BOOST_FILESYSTEM_VERSION == 3
            std::string file = (itr->path ().filename ()).string ();
#else
            std::string file = (itr->path ().filename ());
#endif

            boost::split (strs, file, boost::is_any_of ("."));
            std::string extension = strs[strs.size () - 1];

            if (extension == "pcd")
            {

#if BOOST_FILESYSTEM_VERSION == 3
                std::string path = rel_path_so_far + (itr->path ().filename ()).string ();
#else
                std::string path = rel_path_so_far + (itr->path ()).filename ();
#endif

                relative_paths.push_back (path);
            }
        }
    }
}

void
getModelsInDirectory (bf::path & dir, std::string & rel_path_so_far, std::vector<std::string> & relative_paths, std::string & ext)
{
    bf::directory_iterator end_itr;
    for (bf::directory_iterator itr (dir); itr != end_itr; ++itr)
    {
        //check if its a directory, then get models in it
        if (bf::is_directory (*itr))
        {
#if BOOST_FILESYSTEM_VERSION == 3
            std::string so_far = rel_path_so_far + (itr->path ().filename ()).string () + "/";
#else
            std::string so_far = rel_path_so_far + (itr->path ()).filename () + "/";
#endif

            bf::path curr_path = itr->path ();
            getModelsInDirectory (curr_path, so_far, relative_paths, ext);
        }
        else
        {
            //check that it is a ply file and then add, otherwise ignore..
            std::vector<std::string> strs;
#if BOOST_FILESYSTEM_VERSION == 3
            std::string file = (itr->path ().filename ()).string ();
#else
            std::string file = (itr->path ()).filename ();
#endif

            boost::split (strs, file, boost::is_any_of ("."));
            std::string extension = strs[strs.size () - 1];

            if (extension.compare (ext) == 0)
            {
#if BOOST_FILESYSTEM_VERSION == 3
                std::string path = rel_path_so_far + (itr->path ().filename ()).string ();
#else
                std::string path = rel_path_so_far + (itr->path ()).filename ();
#endif

                relative_paths.push_back (path);
            }
        }
    }
}

inline bool
sortFiles (const std::string & file1, const std::string & file2)
{
    std::vector<std::string> strs1;
    boost::split (strs1, file1, boost::is_any_of ("/"));

    std::vector<std::string> strs2;
    boost::split (strs2, file2, boost::is_any_of ("/"));

    std::string id_1 = strs1[strs1.size () - 1];
    std::string id_2 = strs2[strs2.size () - 1];

    size_t pos1 = id_1.find (".ply.pcd");
    size_t pos2 = id_2.find (".ply.pcd");

    id_1 = id_1.substr (0, pos1);
    id_2 = id_2.substr (0, pos2);

    id_1 = id_1.substr (2);
    id_2 = id_2.substr (2);

    return atoi (id_1.c_str ()) < atoi (id_2.c_str ());
}

std::string STATISTIC_OUTPUT_FILE_ = "stats.txt";

template<typename PointT>
void
recognizeAndVisualize (typename boost::shared_ptr<faat_pcl::rec_3d_framework::Recognizer<PointT> > & local,
                       std::string & scene_file)
{

    faat_pcl::rec_3d_framework::or_evaluator::OREvaluator<PointT> or_eval;
    or_eval.setGTDir(GT_DIR_);
    or_eval.setModelsDir(MODELS_DIR_);
    or_eval.setCheckPose(true);
    or_eval.setScenesDir(scene_file);

    typename boost::shared_ptr<faat_pcl::rec_3d_framework::Source<PointT> > model_source_ = local->getDataSource ();
    typedef typename pcl::PointCloud<PointT>::ConstPtr ConstPointInTPtr;
    typedef faat_pcl::rec_3d_framework::Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;

    local->setVoxelSizeICP (VX_SIZE_ICP_);

    pcl::visualization::PCLVisualizer vis ("Recognition results");
    int v1, v2, v3, v4, v5;

    if(SHOW_GT_)
    {
        /*vis.createViewPort (0.0, 0.0, 0.5, 0.5, v1);
      vis.createViewPort (0.5, 0, 1, 0.5, v4);
      vis.createViewPort (0, 0.5, 1, 1.0, v2);
      vis.createViewPort (0.5, 0.5, 1.0, 1.0, v3);*/

        vis.createViewPort (0.0, 0.0, 0.33, 1.0, v1);
        vis.createViewPort (0.33, 0.0, 0.66, 1.0, v2);
        vis.createViewPort (0.66, 0.0, 1, 1.0, v3);

        vis.addText ("Ground truth", 1, 30, 18, 1, 0, 0, "gt_text", v3);
        vis.addText ("Recognition hypotheses", 1, 30, 18, 1, 0, 0, "recog_hyp_text", v2);
    }
    else
    {
        vis.createViewPort (0.0, 0.0, 0.5, 1.0, v1);
        vis.createViewPort (0.5, 0, 1.0, 1.0, v2);
        vis.addText ("Recognition results", 1, 30, 18, 1, 0, 0, "recog_res_text", v2);
    }

    bf::path input = scene_file;
    std::vector<std::string> files_to_recognize;

    if (bf::is_directory (input))
    {
        std::vector<std::string> files;
        std::string start = "";
        std::string ext = std::string ("pcd");
        bf::path dir = input;
        getModelsInDirectory (dir, start, files, ext);
        std::cout << "Number of scenes in directory is:" << files.size () << std::endl;
        for (size_t i = 0; i < files.size (); i++)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud (new pcl::PointCloud<pcl::PointXYZ>);
            std::cout << files[i] << std::endl;
            std::stringstream filestr;
            filestr << scene_file << "/" << files[i];
            std::string file = filestr.str ();
            files_to_recognize.push_back (file);
        }

        std::sort (files_to_recognize.begin (), files_to_recognize.end (), sortFiles);
        or_eval.setScenesDir(scene_file);
        or_eval.setDataSource(local->getDataSource());
        or_eval.loadGTData();
    }
    else
    {
        files_to_recognize.push_back (scene_file);
    }

    for (size_t i = 0; i < files_to_recognize.size (); i++)
    {

        std::vector<std::string> strs1;
        boost::split (strs1, files_to_recognize[i], boost::is_any_of ("/"));

        std::string id_1 = strs1[strs1.size () - 1];
        size_t pos1 = id_1.find (".pcd");

        id_1 = id_1.substr (0, pos1);

        pcl::PointCloud<pcl::PointXYZ>::Ptr scene (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::io::loadPCDFile (files_to_recognize[i], *scene);

        if (Z_DIST_ > 0)
        {
            pcl::PassThrough<PointT> pass_;
            pass_.setFilterLimits (0.f, Z_DIST_);
            pass_.setFilterFieldName ("z");
            pass_.setInputCloud (scene);
            pass_.setKeepOrganized (true);
            pass_.filter (*scene);
        }

        pcl::visualization::PointCloudColorHandlerGenericField<PointT> scene_handler(scene, "z");
        vis.addPointCloud<PointT> (scene, scene_handler, "scene_cloud", v1);

        local->setInputCloud (scene);
        {
            pcl::ScopeTime ttt ("Recognition");
            local->recognize ();
        }

        std::vector < std::string > strs;
        boost::split (strs, files_to_recognize[i], boost::is_any_of ("/"));
        vis.addText (strs[strs.size() - 1], 1, 30, 18, 1, 0, 0, "scene_text", v1);

        //visualize results
        boost::shared_ptr<std::vector<ModelTPtr> > models = local->getModels ();
        boost::shared_ptr<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > transforms = local->getTransforms ();

        std::vector<typename pcl::PointCloud<PointT>::ConstPtr> aligned_models;
        aligned_models.resize (models->size ());

        for (size_t kk = 0; kk < models->size (); kk++)
        {
            ConstPointInTPtr model_cloud = models->at (kk)->getAssembled (0.005f);
            typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
            pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (kk));
            aligned_models[kk] = model_aligned;
        }

        if(SHOW_GT_)
        {
            pcl::visualization::PointCloudColorHandlerCustom<PointT> scene_handler(scene, 125,125,125);
            vis.addPointCloud<PointT> (scene, scene_handler, "scene_cloud_v4", v3);
            or_eval.visualizeGroundTruth(vis, id_1, v3, false);
        }

        boost::shared_ptr<std::vector<ModelTPtr> > verified_models(new std::vector<ModelTPtr>);
        boost::shared_ptr<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > verified_transforms;
        verified_transforms.reset(new std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >);

        for (size_t j = 0; j < aligned_models.size (); j++)
        {
            std::stringstream name;
            name << "cloud_" << j;

            verified_models->push_back(models->at(j));
            verified_transforms->push_back(transforms->at(j));
        }
        /*for (size_t j = 0; j < aligned_models.size (); j++)
        {
            std::stringstream name;
            name << "cloud_" << j;

            verified_models->push_back(models->at(j));
            verified_transforms->push_back(transforms->at(j));

            ConstPointInTPtr model_cloud = models->at (j)->getAssembled (0.001f);
            typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
            pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (j));

            std::stringstream pathPly;
            pathPly << MODELS_DIR_FOR_VIS_ << "/" << models->at (j)->id_ << ".ply";
            vtkSmartPointer < vtkTransform > poseTransform = vtkSmartPointer<vtkTransform>::New ();
            vtkSmartPointer < vtkTransform > scale_models = vtkSmartPointer<vtkTransform>::New ();
            scale_models->Scale(model_scale, model_scale, model_scale);

            vtkSmartPointer < vtkMatrix4x4 > mat = vtkSmartPointer<vtkMatrix4x4>::New ();
            for (size_t kk = 0; kk < 4; kk++)
            {
                for (size_t k = 0; k < 4; k++)
                {
                    mat->SetElement (kk, k, transforms->at (j) (kk, k));
                }
            }

            poseTransform->SetMatrix (mat);
            poseTransform->Modified ();
            poseTransform->Concatenate(scale_models);

            std::stringstream cluster_name;
            cluster_name << "_ply_model_" << j;
            vis.addModelFromPLYFile (pathPly.str (), poseTransform, cluster_name.str (), v3);
        }*/

        or_eval.addRecognitionResults(id_1, verified_models, verified_transforms);

        for (size_t j = 0; j < models->size (); j++)
        {
            std::stringstream name;
            name << "cloud_before_hv_" << j;

            ConstPointInTPtr model_cloud = models->at (j)->getAssembled (0.003f);
            typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
            pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (j));

            float r, g, b;
            r = 255.0f;
            g = 0.0f;
            b = 0.0f;

            if (models->at (j)->id_.compare ("Zoe") == 0)
            {
                r = 0.0f;
                g = 255.0f;
                b = 0.0f;
            }
            else if (models->at (j)->id_.compare ("Kid") == 0)
            {
                r = 0.0f;
                g = 255.0f;
                b = 255.0f;
            }
            else if (models->at (j)->id_.compare ("BigBird") == 0)
            {
                r = 255.0f;
                g = 255.0f;
                b = 0.f;
            }
            else if (models->at (j)->id_.compare ("Angel") == 0)
            {
                r = 0.0f;
                g = 122.5f;
                b = 122.5f;
            }
            else if (models->at (j)->id_.compare ("chef") == 0)
            {
                r = 0.0f;
                g = 255.0f;
                b = 0.0f;
            }
            else if (models->at (j)->id_.compare ("chicken") == 0)
            {
                r = 0.0f;
                g = 255.0f;
                b = 255.0f;
            }
            else if (models->at (j)->id_.compare ("para") == 0)
            {
                r = 255.0f;
                g = 255.0f;
                b = 0.f;
            }
            else if (models->at (j)->id_.compare ("rhino") == 0)
            {
                r = 255.0f;
                g = 105.0f;
                b = 180.f;
            }

            pcl::visualization::PointCloudColorHandlerCustom<PointT> random_handler (model_aligned, r, g, b);
            vis.addPointCloud<PointT> (model_aligned, random_handler, name.str (), v2);
        }

        vis.setBackgroundColor (1.0, 1.0, 1.0);
        if (PLAY_)
        {
            vis.spinOnce (500.f, true);
        }
        else
        {
            vis.spin ();
        }

        vis.removePointCloud ("scene_cloud");
        vis.removeShape ("scene_text");
        vis.removeAllPointClouds();
    }

    or_eval.computeStatistics();
    or_eval.saveStatistics(STATISTIC_OUTPUT_FILE_);
}

typedef pcl::ReferenceFrame RFType;

int CG_SIZE_ = 3;
float CG_THRESHOLD_ = 0.005f;

/*
 ./bin/local_recognition_ply -models_dir /home/aitor/data/queens_dataset/pcd_models/ -pcd_file /home/aitor/data/queens_dataset/hard_scenes/ -training_dir /home/aitor/data/queens_dataset/trained_models -gc_size 5 -icp_type 1 -use_hv 1 -icp_iterations 10 -use_cache 1 -splits 32 -hv_method 0 -model_scale 1 -go_opt_type 0 -thres_hyp 0.5 -go_iterations 5000 -go_use_replace_moves 1 -go_resolution 0.005 -go_inlier_thres 0.01 -go_initial_temp 500 -idx_flann_fn queens_flann.idx -PLAY 1 -vx_size_icp 0.005 -go_require_normals 0 -go_init 0 -go_log_file test.txt -use_hough 0 -gc_threshold 0.01 -gc_ransac_threshold 0.01 -test_sampling_density 0.005 -force_retrain 0 -use_gc_graph 1 -desc_radius 0.04 -GT_DIR /home/aitor/data/queens_dataset/gt_or_format_all_with_occ/ -models_dir_vis /home/aitor/data/queens_dataset/models_for_visualization/ -gc_min_dist_cf 0 -gc_dot_threshold 0.25
 ./bin/local_recognition_ply -models_dir /home/aitor/data/Mians_dataset/models/ -pcd_file /home/aitor/data/Mians_dataset/scenes/pcl_scenes -training_dir /home/aitor/data/Mians_trained_models_voxelsize_0.003/ -gc_size 3 -icp_type 1 -use_hv 1 -icp_iterations 10 -use_cache 1 -splits 32 -hv_method 0 -model_scale 0.001 -go_opt_type 0 -thres_hyp 0.2 -go_iterations 10000 -go_use_replace_moves 1 -go_resolution 0.005 -go_inlier_thres 0.005 -go_initial_temp 500 -idx_flann_fn mian_flann_new.idx -PLAY 1 -vx_size_icp 0.005 -go_require_normals 0 -go_init 0 -go_log_file test.txt -use_hough 0 -gc_threshold 0.01 -gc_ransac_threshold 0.01 -test_sampling_density 0.005 -force_retrain 0 -use_gc_graph 1 -desc_radius 0.04 -GT_DIR /home/aitor/data/Mians_dataset/gt_or_format_rhino_with_occ -models_dir_vis /home/aitor/data/Mians_dataset/models_with_rhino_vis/ -gc_min_dist_cf 0 -use_board 1 -rf_radius_hough 0.04 -gc_dot_threshold 1
 ./bin/local_recognition_ply -models_dir /home/aitor/data/Mians_dataset/models_with_rhino/ -pcd_file /home/aitor/data/Mians_dataset/scenes/pcl_scenes -training_dir /home/aitor/data/Mians_trained_models_voxelsize_0.003/ -gc_size 3 -icp_type 1 -use_hv 1 -icp_iterations 10 -use_cache 1 -splits 32 -hv_method 0 -model_scale 0.001 -go_opt_type 0 -thres_hyp 0.2 -go_iterations 10000 -go_use_replace_moves 1 -go_resolution 0.005 -go_inlier_thres 0.005 -go_initial_temp 500 -idx_flann_fn mian_flann_with_rhino_new.idx -PLAY 1 -vx_size_icp 0.005 -go_require_normals 0 -go_init 0 -go_log_file test.txt -use_hough 0 -gc_threshold 0.01 -gc_ransac_threshold 0.01 -test_sampling_density 0.005 -force_retrain 0 -use_gc_graph 1 -desc_radius 0.04 -GT_DIR /home/aitor/data/Mians_dataset/gt_or_format_rhino_with_occ -models_dir_vis /home/aitor/data/Mians_dataset/models_with_rhino_vis/ -gc_min_dist_cf 0 -use_board 1 -rf_radius_hough 0.04 -gc_dot_threshold 1
 works on hard scenes (Arboricity 15) ./bin/local_recognition_ply -models_dir /home/aitor/data/Mians_dataset/models_with_rhino/ -pcd_file /home/aitor/data/Mians_dataset/scenes/hard_scenes/ -training_dir /home/aitor/data/Mians_trained_models_voxelsize_0.003/ -gc_size 3 -icp_type 1 -use_hv 1 -icp_iterations 10 -use_cache 1 -splits 32 -hv_method 0 -model_scale 0.001 -go_opt_type 0 -thres_hyp 0.2 -go_iterations 10000 -go_use_replace_moves 1 -go_resolution 0.005 -go_inlier_thres 0.005 -go_initial_temp 500 -idx_flann_fn mian_flann_with_rhino_new.idx -PLAY 0 -vx_size_icp 0.005 -go_require_normals 0 -go_init 0 -go_log_file test.txt -use_hough 0 -gc_threshold 0.01 -gc_ransac_threshold 0.01 -test_sampling_density 0.005 -force_retrain 0 -use_gc_graph 1 -desc_radius 0.04 -GT_DIR /home/aitor/data/Mians_dataset/gt_or_format_rhino_with_occ -models_dir_vis /home/aitor/data/Mians_dataset/models_with_rhino_vis/ -gc_min_dist_cf 0 -use_board 0 -rf_radius_hough 0.04 -gc_dot_threshold 1
 works on the whole dataset ./bin/local_recognition_ply -models_dir /home/aitor/data/Mians_dataset/models_with_rhino/ -pcd_file /home/aitor/data/Mians_dataset/scenes/pcl_scenes -training_dir /home/aitor/data/Mians_trained_models_voxelsize_0.003/ -gc_size 3 -icp_type 1 -use_hv 1 -icp_iterations 10 -use_cache 1 -splits 32 -hv_method 0 -model_scale 0.001 -go_opt_type 0 -thres_hyp 0.2 -go_iterations 10000 -go_use_replace_moves 1 -go_resolution 0.005 -go_inlier_thres 0.005 -go_initial_temp 500 -idx_flann_fn mian_flann_with_rhino_new.idx -PLAY 1 -vx_size_icp 0.005 -go_require_normals 0 -go_init 0 -go_log_file test.txt -use_hough 0 -gc_threshold 0.01 -gc_ransac_threshold 0.01 -test_sampling_density 0.005 -force_retrain 0 -use_gc_graph 1 -desc_radius 0.04 -GT_DIR /home/aitor/data/Mians_dataset/gt_or_format_rhino_with_occ -models_dir_vis /home/aitor/data/Mians_dataset/models_with_rhino_vis/ -gc_min_dist_cf 1 -use_board 0 -rf_radius_hough 0.04 -gc_dot_threshold 0.5

 ./bin/local_recognition_ply -models_dir /home/aitor/data/Mians_dataset/models_with_rhino/ -pcd_file /home/aitor/data/Mians_dataset/scenes/hard_scenes/ -training_dir /home/aitor/data/Mians_trained_models_voxelsize_0.003/ -gc_size 3 -icp_type 1 -use_hv 1 -icp_iterations 50 -use_cache 1 -splits 512 -hv_method 0 -model_scale 0.001 -go_opt_type 0 -thres_hyp 0.2 -go_iterations 10000 -go_use_replace_moves 1 -go_resolution 0.005 -go_inlier_thres 0.005 -go_initial_temp 500 -idx_flann_fn mian_flann_with_rhino_new.idx -PLAY 0 -vx_size_icp 0.005 -go_require_normals 0 -go_init 0 -go_log_file test.txt -use_hough 0 -gc_threshold 0.01 -gc_ransac_threshold 0.01 -test_sampling_density 0.005 -force_retrain 0 -use_gc_graph 1 -desc_radius 0.04 -GT_DIR /home/aitor/data/Mians_dataset/gt_or_format_rhino_with_occ -models_dir_vis /home/aitor/data/Mians_dataset/models_with_rhino_vis/ -gc_min_dist_cf 0.5 -use_board 0 -rf_radius_hough 0.04 -gc_dot_threshold 0.25 -visualize_graph 0
 ./bin/local_recognition_ply -models_dir /home/aitor/data/Mians_dataset/models/ -pcd_file /home/aitor/data/Mians_dataset/scenes/hard_scenes/ -training_dir /home/aitor/data/Mians_trained_models_voxelsize_0.003/ -gc_size 3 -icp_type 1 -use_hv 1 -icp_iterations 50 -use_cache 1 -splits 512 -hv_method 0 -model_scale 0.001 -go_opt_type 0 -thres_hyp 0.2 -go_iterations 10000 -go_use_replace_moves 1 -go_resolution 0.005 -go_inlier_thres 0.005 -go_initial_temp 500 -idx_flann_fn mian_flann_new.idx -PLAY 0 -vx_size_icp 0.005 -go_require_normals 0 -go_init 0 -go_log_file test.txt -use_hough 0 -gc_threshold 0.01 -gc_ransac_threshold 0.01 -test_sampling_density 0.005 -force_retrain 0 -use_gc_graph 1 -desc_radius 0.04 -GT_DIR /home/aitor/data/Mians_dataset/gt_or_format_rhino_with_occ -models_dir_vis /home/aitor/data/Mians_dataset/models_with_rhino_vis/ -gc_min_dist_cf 0.5 -use_board 0 -rf_radius_hough 0.04 -gc_dot_threshold 0.25 -visualize_graph 0

 */
int
main (int argc, char ** argv)
{
    std::string path = "";
    std::string desc_name = "shot";
    std::string training_dir = "trained_models/";
    std::string pcd_file = "";
    int force_retrain = 0;
    int icp_iterations = 20;
    int use_cache = 1;
    int splits = 512;
    int scene = -1;
    float thres_hyp_ = 0.2f;
    float desc_radius = 0.04f;
    int icp_type = 0;
    std::string idx_flann_fn;
    float test_sampling_density = 0.005f;
    int tes_level_ = 1;
    float gc_ransac_threshold;
    bool use_gc_graph = true;
    float min_dist_cf_ = 1.f;
    float gc_dot_threshold_ = 1.f;
    bool use_board = false;
    float rf_radius_hough = 0.04f;
    bool visualize_graph = false;
    int cg_method = 0; //0-GGC, 1-IGC, 2-Hough
    bool prune_by_cc = false;

    pcl::console::parse_argument (argc, argv, "-prune_by_cc", prune_by_cc);
    pcl::console::parse_argument (argc, argv, "-cg_method", cg_method);
    pcl::console::parse_argument (argc, argv, "-visualize_graph", visualize_graph);
    pcl::console::parse_argument (argc, argv, "-use_gc_graph", use_gc_graph);
    pcl::console::parse_argument (argc, argv, "-rf_radius_hough", rf_radius_hough);
    pcl::console::parse_argument (argc, argv, "-use_board", use_board);
    pcl::console::parse_argument (argc, argv, "-models_dir", path);
    pcl::console::parse_argument (argc, argv, "-training_dir", training_dir);
    pcl::console::parse_argument (argc, argv, "-descriptor_name", desc_name);
    pcl::console::parse_argument (argc, argv, "-pcd_file", pcd_file);
    pcl::console::parse_argument (argc, argv, "-force_retrain", force_retrain);
    pcl::console::parse_argument (argc, argv, "-icp_iterations", icp_iterations);
    pcl::console::parse_argument (argc, argv, "-use_cache", use_cache);
    pcl::console::parse_argument (argc, argv, "-splits", splits);
    pcl::console::parse_argument (argc, argv, "-gc_size", CG_SIZE_);
    pcl::console::parse_argument (argc, argv, "-gc_threshold", CG_THRESHOLD_);
    pcl::console::parse_argument (argc, argv, "-scene", scene);
    pcl::console::parse_argument (argc, argv, "-thres_hyp", thres_hyp_);
    pcl::console::parse_argument (argc, argv, "-icp_type", icp_type);
    pcl::console::parse_argument (argc, argv, "-vx_size_icp", VX_SIZE_ICP_);
    pcl::console::parse_argument (argc, argv, "-model_scale", model_scale);
    pcl::console::parse_argument (argc, argv, "-idx_flann_fn", idx_flann_fn);
    pcl::console::parse_argument (argc, argv, "-PLAY", PLAY_);
    pcl::console::parse_argument (argc, argv, "-test_sampling_density", test_sampling_density);
    pcl::console::parse_argument (argc, argv, "-tes_level", tes_level_);
    pcl::console::parse_argument (argc, argv, "-Z_DIST", Z_DIST_);
    pcl::console::parse_argument (argc, argv, "-gc_ransac_threshold", gc_ransac_threshold);
    pcl::console::parse_argument (argc, argv, "-desc_radius", desc_radius);
    pcl::console::parse_argument (argc, argv, "-show_gt", SHOW_GT_);
    pcl::console::parse_argument (argc, argv, "-gc_min_dist_cf", min_dist_cf_);
    pcl::console::parse_argument (argc, argv, "-gc_dot_threshold", gc_dot_threshold_);
    pcl::console::parse_argument (argc, argv, "-stat_file", STATISTIC_OUTPUT_FILE_);

    MODELS_DIR_FOR_VIS_ = path;

    pcl::console::parse_argument (argc, argv, "-models_dir_vis", MODELS_DIR_FOR_VIS_);
    pcl::console::parse_argument (argc, argv, "-GT_DIR", GT_DIR_);
    MODELS_DIR_ = path;

    std::cout << "VX_SIZE_ICP_" << VX_SIZE_ICP_ << std::endl;
    if (pcd_file.compare ("") == 0)
    {
        PCL_ERROR("Set the directory containing mians scenes using the -mians_scenes_dir [dir] option\n");
        return -1;
    }

    if (path.compare ("") == 0)
    {
        PCL_ERROR("Set the directory containing the models of mian dataset using the -models_dir [dir] option\n");
        return -1;
    }

    bf::path models_dir_path = path;
    if (!bf::exists (models_dir_path))
    {
        PCL_ERROR("Models dir path %s does not exist, use -models_dir [dir] option\n", path.c_str());
        return -1;
    }
    else
    {
        std::vector<std::string> files;
        std::string start = "";
        std::string ext = std::string ("ply");
        bf::path dir = models_dir_path;
        getModelsInDirectory (dir, start, files, ext);
        std::cout << "Number of models in directory is:" << files.size () << std::endl;
    }

    //configure mesh source
    boost::shared_ptr<faat_pcl::rec_3d_framework::MeshSource<pcl::PointXYZ> > mesh_source (new faat_pcl::rec_3d_framework::MeshSource<pcl::PointXYZ>);
    mesh_source->setPath (path);
    mesh_source->setResolution (250);
    mesh_source->setTesselationLevel (tes_level_);
    mesh_source->setViewAngle (57.f);
    mesh_source->setRadiusSphere (1.5f);
    mesh_source->setModelScale (model_scale);
    //mesh_source->setRadiusNormals (radius_normals_go_);
    mesh_source->generate (training_dir);

    boost::shared_ptr<faat_pcl::rec_3d_framework::Source<pcl::PointXYZ> > cast_source;
    cast_source = boost::static_pointer_cast<faat_pcl::rec_3d_framework::MeshSource<pcl::PointXYZ> > (mesh_source);

    //configure normal estimator
    boost::shared_ptr<faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<pcl::PointXYZ, pcl::Normal> > normal_estimator;
    normal_estimator.reset (new faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<pcl::PointXYZ, pcl::Normal>);
    normal_estimator->setCMR (false);
    normal_estimator->setDoVoxelGrid (true);
    normal_estimator->setRemoveOutliers (true);
    normal_estimator->setValuesForCMRFalse (0.003f, 0.012f);

    //configure keypoint extractor
    boost::shared_ptr<faat_pcl::rec_3d_framework::UniformSamplingExtractor<pcl::PointXYZ> >
            uniform_keypoint_extractor (
                new faat_pcl::rec_3d_framework::UniformSamplingExtractor<
                pcl::PointXYZ>);

    uniform_keypoint_extractor->setSamplingDensity (0.01f);
    //uniform_keypoint_extractor->setSamplingDensity (0.005f);
    uniform_keypoint_extractor->setFilterPlanar (true);

    boost::shared_ptr<faat_pcl::rec_3d_framework::KeypointExtractor<pcl::PointXYZ> > keypoint_extractor;
    keypoint_extractor = boost::static_pointer_cast<faat_pcl::rec_3d_framework::KeypointExtractor<pcl::PointXYZ> > (uniform_keypoint_extractor);

    //ISS keypoint extractor
    boost::shared_ptr<faat_pcl::rec_3d_framework::ISSKeypointExtractor<pcl::PointXYZ> >
            iss_keypoint_extractor (
                new faat_pcl::rec_3d_framework::ISSKeypointExtractor<
                pcl::PointXYZ>);

    iss_keypoint_extractor->setNonMaximaRadius(0.01f);
    iss_keypoint_extractor->setSupportRadius(0.04f);

    if(desc_name.compare("shot_iss") == 0)
    {
        keypoint_extractor = boost::static_pointer_cast<faat_pcl::rec_3d_framework::KeypointExtractor<pcl::PointXYZ> > (iss_keypoint_extractor);
    }

    boost::shared_ptr<faat_pcl::rec_3d_framework::Recognizer<pcl::PointXYZ> > cast_recog;

    boost::shared_ptr<faat_pcl::rec_3d_framework::SHOTLocalEstimationOMP<pcl::PointXYZ, pcl::Histogram<352> > > estimator;
    estimator.reset (new faat_pcl::rec_3d_framework::SHOTLocalEstimationOMP<pcl::PointXYZ, pcl::Histogram<352> >);
    estimator->setNormalEstimator (normal_estimator);
    estimator->addKeypointExtractor (keypoint_extractor);
    estimator->setSupportRadius (desc_radius);

    boost::shared_ptr<faat_pcl::rec_3d_framework::LocalEstimator<pcl::PointXYZ, pcl::Histogram<352> > > cast_estimator;
    cast_estimator = boost::dynamic_pointer_cast<faat_pcl::rec_3d_framework::LocalEstimator<pcl::PointXYZ, pcl::Histogram<352> > > (estimator);

    //configure cg algorithm (geometric consistency grouping)
    boost::shared_ptr<pcl::CorrespondenceGrouping<pcl::PointXYZ, pcl::PointXYZ> > cast_cg_alg;

    if(cg_method == 2) //hough
    {
        boost::shared_ptr<pcl::Hough3DGrouping<pcl::PointXYZ, pcl::PointXYZ, pcl::ReferenceFrame, pcl::ReferenceFrame> >
                hough_3d_voting_cg_alg (
                    new pcl::Hough3DGrouping<
                    pcl::PointXYZ,
                    pcl::PointXYZ,
                    pcl::ReferenceFrame,
                    pcl::ReferenceFrame>);

        hough_3d_voting_cg_alg->setHoughBinSize (CG_THRESHOLD_);
        hough_3d_voting_cg_alg->setHoughThreshold (CG_SIZE_);
        hough_3d_voting_cg_alg->setUseInterpolation (true);
        hough_3d_voting_cg_alg->setUseDistanceWeight (false);
        //hough_3d_voting_cg_alg->setRansacThreshold (gc_ransac_threshold);

        cast_cg_alg = boost::static_pointer_cast<pcl::Hough3DGrouping<pcl::PointXYZ, pcl::PointXYZ> > (hough_3d_voting_cg_alg);
        boost::shared_ptr<faat_pcl::rec_3d_framework::LocalRecognitionHoughGroupingPipeline<flann::L1, pcl::PointXYZ, pcl::Histogram<352> > > local;
        local.reset (new faat_pcl::rec_3d_framework::LocalRecognitionHoughGroupingPipeline<flann::L1, pcl::PointXYZ, pcl::Histogram<352> > (idx_flann_fn));
        local->setDataSource (cast_source);
        local->setTrainingDir (training_dir);
        local->setDescriptorName (desc_name);
        local->setFeatureEstimator (cast_estimator);
        local->setCGAlgorithm (cast_cg_alg);
        local->setUseBoard(use_board);
        local->setRFRadius(rf_radius_hough);
        //if (use_hv)
        //local.setHVAlgorithm (cast_hv_alg);

        local->setUseCache (static_cast<bool> (use_cache));
        local->setVoxelSizeICP (VX_SIZE_ICP_);

        local->initialize (static_cast<bool> (force_retrain));
        local->setThresholdAcceptHyp (thres_hyp_);

        uniform_keypoint_extractor->setSamplingDensity (test_sampling_density);
        local->setICPIterations (icp_iterations);
        local->setKdtreeSplits (splits);
        local->setICPType (icp_type);

        cast_recog
                = boost::static_pointer_cast<faat_pcl::rec_3d_framework::LocalRecognitionHoughGroupingPipeline<flann::L1, pcl::PointXYZ, pcl::Histogram<352> > > (local);
    }
    else
    {
        if(cg_method == 1)
        {
            boost::shared_ptr<pcl::GeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ> > gcg_alg (
                        new pcl::GeometricConsistencyGrouping<pcl::PointXYZ,
                        pcl::PointXYZ>);
            gcg_alg->setGCThreshold (CG_SIZE_);
            gcg_alg->setGCSize (CG_THRESHOLD_);
            //gcg_alg->setRansacThreshold (gc_ransac_threshold);

            cast_cg_alg = boost::static_pointer_cast<pcl::CorrespondenceGrouping<pcl::PointXYZ, pcl::PointXYZ> > (gcg_alg);
        }
        else if(cg_method == 0)
        {
            boost::shared_ptr<faat_pcl::GraphGeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ> > gcg_alg (
                        new faat_pcl::GraphGeometricConsistencyGrouping<pcl::PointXYZ,
                        pcl::PointXYZ>);
            gcg_alg->setGCThreshold (CG_SIZE_);
            gcg_alg->setGCSize (CG_THRESHOLD_);
            gcg_alg->setRansacThreshold (gc_ransac_threshold);
            gcg_alg->setUseGraph(use_gc_graph);
            gcg_alg->setDistForClusterFactor(min_dist_cf_);
            gcg_alg->setPrune(false);
            gcg_alg->setVisualizeGraph(visualize_graph);
            gcg_alg->setDotDistance(gc_dot_threshold_);
            gcg_alg->setPrune(prune_by_cc);
            cast_cg_alg = boost::static_pointer_cast<pcl::CorrespondenceGrouping<pcl::PointXYZ, pcl::PointXYZ> > (gcg_alg);
        }

        boost::shared_ptr<faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, pcl::PointXYZ, pcl::Histogram<352> > > local;
        local.reset (new faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, pcl::PointXYZ, pcl::Histogram<352> > (idx_flann_fn));
        local->setDataSource (cast_source);
        local->setTrainingDir (training_dir);
        local->setDescriptorName (desc_name);
        local->setFeatureEstimator (cast_estimator);
        local->setCGAlgorithm (cast_cg_alg);

        local->setUseCache (static_cast<bool> (use_cache));
        local->setVoxelSizeICP (VX_SIZE_ICP_);

        local->initialize (static_cast<bool> (force_retrain));
        local->setThresholdAcceptHyp (thres_hyp_);

        iss_keypoint_extractor->setNonMaximaRadius(test_sampling_density);
        uniform_keypoint_extractor->setSamplingDensity (test_sampling_density);
        local->setICPIterations (icp_iterations);
        local->setKdtreeSplits (splits);
        local->setICPType (icp_type);

        cast_recog
                = boost::static_pointer_cast<faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, pcl::PointXYZ, pcl::Histogram<352> > > (local);

    }

    recognizeAndVisualize<pcl::PointXYZ> (cast_recog, pcd_file);
}
