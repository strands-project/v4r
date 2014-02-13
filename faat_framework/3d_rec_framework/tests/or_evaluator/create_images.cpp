/*
 * local_recognition_mian_dataset.cpp
 *
 *  Created on: Mar 24, 2012
 *      Author: aitor
 */
#include <pcl/console/parse.h>
#include <faat_pcl/3d_rec_framework/pc_source/model_only_source.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <faat_pcl/3d_rec_framework/tools/or_evaluator.h>
#include <opencv2/opencv.hpp>
#include <faat_pcl/utils/pcl_opencv.h>

float VX_SIZE_ICP_ = 0.005f;
bool PLAY_ = false;
std::string go_log_file_ = "test.txt";
float Z_DIST_ = 1.5f;
std::string GT_DIR_;
std::string MODELS_DIR_;
std::string MODELS_DIR_FOR_VIS_;
float model_scale = 1.f;
bool use_HV = true;

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
            std::string so_far = rel_path_so_far + (itr->path ().filename ()).string() + "/";
#else
            std::string so_far = rel_path_so_far + (itr->path ()).filename () + "/";
#endif

            bf::path curr_path = itr->path ();
            getModelsInDirectory (curr_path, so_far, relative_paths, ext);
        }
        else
        {
            //check that it is a ply file and then add, otherwise ignore..
            std::vector < std::string > strs;
#if BOOST_FILESYSTEM_VERSION == 3
            std::string file = (itr->path ().filename ()).string();
#else
            std::string file = (itr->path ()).filename ();
#endif

            boost::split (strs, file, boost::is_any_of ("."));
            std::string extension = strs[strs.size () - 1];

            if (extension.compare (ext) == 0)
            {
#if BOOST_FILESYSTEM_VERSION == 3
                std::string path = rel_path_so_far + (itr->path ().filename ()).string();
#else
                std::string path = rel_path_so_far + (itr->path ()).filename ();
#endif

                relative_paths.push_back (path);
            }
        }
    }
}

std::string OUTPUT_DIR_IMAGES_;
std::string pcd_file = "";

template<typename PointT>
void
recognizeAndVisualize (std::string & scene_file)
{


    bf::path or_path = OUTPUT_DIR_IMAGES_;
    if(!bf::exists(or_path))
    {
        bf::create_directory(or_path);
    }

    std::string scene_images_path;
    std::string scene_images_gt_path;

    {
        std::stringstream t;
        t << OUTPUT_DIR_IMAGES_  << "/scenes/";
        scene_images_path = t.str();
        bf::path scene_images = scene_images_path;
        if(!bf::exists(scene_images))
        {
            bf::create_directory(scene_images);
        }
    }

    {
        std::stringstream t;
        t << OUTPUT_DIR_IMAGES_  << "/gt/";
        scene_images_gt_path = t.str();
        bf::path scene_images = scene_images_gt_path;
        if(!bf::exists(scene_images))
        {
            bf::create_directory(scene_images);
        }
    }

    faat_pcl::rec_3d_framework::or_evaluator::OREvaluator<PointT> or_eval;
    or_eval.setGTDir(GT_DIR_);
    or_eval.setModelsDir(MODELS_DIR_);
    or_eval.setModelFileExtension("pcd");
    or_eval.setReplaceModelExtension(false);
    or_eval.useMaxOcclusion(false);
    or_eval.setMaxOcclusion(0.9f);
    or_eval.setCheckPose(true);
    or_eval.setMaxCentroidDistance(0.03f);

    boost::shared_ptr < faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>
            > source (new faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>);
    source->setPath (MODELS_DIR_);
    source->setLoadViews (false);
    source->setLoadIntoMemory(false);
    std::string test = "irrelevant";
    source->generate (test);

    pcl::visualization::PCLVisualizer vis ("Recognition results");
    int v1, v2;
    vis.createViewPort (0.0, 0.0, 1, 0.5, v1);
    vis.createViewPort (0.0, 0.5, 1, 1, v2);

    vis.addText ("Ground truth", 1, 30, 18, 1, 0, 0, "gt_text", v2);
    vis.addText ("Scene", 1, 30, 18, 1, 0, 0, "scene_texttt", v1);

    bf::path input = scene_file;
    std::vector<std::string> files_to_recognize;

    if (bf::is_directory (input))
    {
        std::vector < std::string > files;
        std::string start = "";
        std::string ext = std::string ("pcd");
        bf::path dir = input;
        getModelsInDirectory (dir, start, files, ext);
        std::cout << "Number of scenes in directory is:" << files.size () << std::endl;
        for (size_t i = 0; i < files.size (); i++)
        {
            typename pcl::PointCloud<PointT>::Ptr scene_cloud (new pcl::PointCloud<PointT>);
            std::cout << files[i] << std::endl;
            std::stringstream filestr;
            filestr << scene_file << files[i];
            std::string file = filestr.str ();
            files_to_recognize.push_back (file);
        }

        std::sort(files_to_recognize.begin(),files_to_recognize.end());
        or_eval.setScenesDir(scene_file);
        or_eval.setDataSource(source);
        or_eval.loadGTData();
    }
    else
    {
        PCL_ERROR("You should pass a directory\n");
        return;
    }

    for(size_t i=0; i < files_to_recognize.size(); i++)
    {
        typename pcl::PointCloud<PointT>::Ptr scene (new pcl::PointCloud<PointT>);
        pcl::io::loadPCDFile (files_to_recognize[i], *scene);

        std::string gt_name = files_to_recognize[i];
        boost::replace_all(gt_name, pcd_file, "");
        std::vector<std::string> strs;
        boost::split (strs, gt_name, boost::is_any_of ("/"));

        std::string scene_name = strs[strs.size() - 1];
        boost::replace_all(scene_name, ".pcd", "");

        {
            std::stringstream rel_path;
            rel_path << scene_images_path << "/";
            for(size_t k=0; k < (strs.size() - 1); k++)
            {
                rel_path << strs[k] << "/";
                bf::path p = rel_path.str();
                if(!bf::exists(p))
                {
                    bf::create_directory(p);
                }
            }

            std::cout << rel_path.str() << "/" << scene_name << ".jpg" << std::endl;

            std::stringstream image_path;
            image_path << rel_path.str() << "/" << scene_name << ".jpg";


            cv::Mat_ < cv::Vec3b > colorImage;
            PCLOpenCV::ConvertPCLCloud2Image<PointT> (scene, colorImage);
            cv::imwrite(image_path.str(), colorImage);
        }

        std::string file_to_recognize(files_to_recognize[i]);
        boost::replace_all (file_to_recognize, scene_file, "");
        boost::replace_all (file_to_recognize, ".pcd", "");
        std::string id_1 = file_to_recognize;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr gt_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        or_eval.getGroundTruthPointCloud(id_1, gt_cloud); //maybe add model resolution here

        //transform gt_cloud to organized point cloud and then to image
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr gt_cloud_organized(new pcl::PointCloud<pcl::PointXYZRGB>);
        gt_cloud_organized->width = scene->width;
        gt_cloud_organized->height = scene->height;
        gt_cloud_organized->is_dense = scene->is_dense;
        gt_cloud_organized->points.resize(scene->points.size());
        for(size_t kk=0; kk < gt_cloud_organized->points.size(); kk++)
        {
            gt_cloud_organized->points[kk].x = gt_cloud_organized->points[kk].y = gt_cloud_organized->points[kk].z =
                    std::numeric_limits<float>::quiet_NaN();

            gt_cloud_organized->points[kk].r = gt_cloud_organized->points[kk].g = gt_cloud_organized->points[kk].b = 0;
        }

        float f = 525.f;
        float cx = (static_cast<float> (scene->width) / 2.f - 0.5f);
        float cy = (static_cast<float> (scene->height) / 2.f - 0.5f);

        int ws2 = 1;
        for (size_t kk = 0; kk < gt_cloud->points.size (); kk++)
        {
          float x = gt_cloud->points[kk].x;
          float y = gt_cloud->points[kk].y;
          float z = gt_cloud->points[kk].z;
          int u = static_cast<int> (f * x / z + cx);
          int v = static_cast<int> (f * y / z + cy);

          for(int uu = (u-ws2); uu < (u+ws2); uu++)
          {
              for(int vv = (v-ws2); vv < (v+ws2); vv++)
              {
                  //Not out of bounds
                    if ((uu >= static_cast<int> (scene->width)) ||
                        (vv >= static_cast<int> (scene->height)) || (uu < 0) || (vv < 0))
                      continue;

                    float z_oc = gt_cloud_organized->at (uu, vv).z;

                    if(pcl_isnan(z_oc))
                    {
                        gt_cloud_organized->at (uu, vv) = gt_cloud->points[kk];
                    }
                    else
                    {
                        if(z < z_oc)
                        {
                            gt_cloud_organized->at (uu, vv) = gt_cloud->points[kk];
                        }
                    }
              }
          }

          /*//Not out of bounds
          if ((u >= static_cast<int> (scene->width)) ||
              (v >= static_cast<int> (scene->height)) || (u < 0) || (v < 0))
            continue;

          float z_oc = gt_cloud_organized->at (u, v).z;

          if(pcl_isnan(z_oc))
          {
              gt_cloud_organized->at (u, v) = gt_cloud->points[kk];
          }
          else
          {
              if(z < z_oc)
              {
                  gt_cloud_organized->at (u, v) = gt_cloud->points[kk];
              }
          }*/
        }

        {
            std::stringstream rel_path;
            rel_path << scene_images_gt_path << "/";
            for(size_t k=0; k < (strs.size() - 1); k++)
            {
                rel_path << strs[k] << "/";
                bf::path p = rel_path.str();
                if(!bf::exists(p))
                {
                    bf::create_directory(p);
                }
            }

            std::cout << rel_path.str() << "/" << scene_name << ".jpg" << std::endl;

            std::stringstream image_path;
            image_path << rel_path.str() << "/" << scene_name << ".jpg";


            cv::Mat_ < cv::Vec3b > colorImage;
            PCLOpenCV::ConvertPCLCloud2Image<PointT> (gt_cloud_organized, colorImage);
            /*cv::namedWindow("image");
            cv::imshow("image", colorImage);
            cv::waitKey(0);*/
            cv::imwrite(image_path.str(), colorImage);
        }

        {
            vis.removeAllPointClouds();

            {
                pcl::visualization::PointCloudColorHandlerRGBField<PointT> scene_handler (scene);
                vis.addPointCloud<PointT> (scene, scene_handler, "scene_cloud_z_coloured", v1);
            }

            {
                pcl::visualization::PointCloudColorHandlerRGBField<PointT> scene_handler (gt_cloud_organized);
                vis.addPointCloud<PointT> (gt_cloud_organized, scene_handler, "gt", v2);
            }

            vis.spin ();
        }
    }
}

int
main (int argc, char ** argv)
{

    std::string path = "";
    std::string desc_name = "shot_omp";
    std::string training_dir = "trained_models/";
    std::string training_dir_shot = "";

    int force_retrain = 0;
    int icp_iterations = 20;
    int use_cache = 1;
    int splits = 512;
    int scene = -1;
    int detect_clutter = 1;
    float thres_hyp_ = 0.2f;
    float desc_radius = 0.04f;
    int icp_type = 0;
    int go_opt_type = 2;
    int go_iterations = 7000;
    bool go_use_replace_moves = true;
    float go_inlier_thres = 0.01f;
    float go_resolution = 0.005f;
    float go_regularizer = 2.f;
    float go_clutter_regularizer = 5.f;
    float go_radius_clutter = 0.05f;
    float init_temp = 1000.f;
    std::string idx_flann_fn;
    float radius_normals_go_ = 0.02f;
    bool go_require_normals = false;
    bool go_log = true;
    bool go_init = false;
    float test_sampling_density = 0.005f;
    int tes_level_ = 1;
    int tes_level_our_cvfh_ = 1;
    std::string pipelines_to_use_ = "shot_omp,our_cvfh";
    bool normalize_ourcvfh_bins = false;
    int max_our_cvfh_hyp_ = 30;
    bool use_hough = false;
    bool load_views = true;
    int seg_type = 0;
    bool add_planes = true;
    bool cg_prune_hyp = false;
    bool use_codebook = false;
    std::string training_input_structure = "";
    std::string training_dir_sift = "";
    bool visualize_graph = false;
    float min_dist_cf_ = 1.f;
    float gc_dot_threshold_ = 1.f;
    bool prune_by_cc = false;
    float go_color_sigma = 0.25f;
    bool go_use_supervoxels = false;
    int knn_sift_ = 1;
    int knn_shot_ = 1;
    int our_cvfh_debug_level = 0;
    float ourcvfh_max_distance = 0.35f;
    bool check_normals_orientation = true;
    bool shot_use_iss = true;

    pcl::console::parse_argument (argc, argv, "-output_dir_images", OUTPUT_DIR_IMAGES_);
    pcl::console::parse_argument (argc, argv, "-shot_use_iss", shot_use_iss);
    pcl::console::parse_argument (argc, argv, "-check_normals_orientation", check_normals_orientation);
    pcl::console::parse_argument (argc, argv, "-ourcvfh_max_distance", ourcvfh_max_distance);
    pcl::console::parse_argument (argc, argv, "-tes_level_our_cvfh", tes_level_our_cvfh_);
    pcl::console::parse_argument (argc, argv, "-debug_level", our_cvfh_debug_level);
    pcl::console::parse_argument (argc, argv, "-knn_shot", knn_shot_);
    pcl::console::parse_argument (argc, argv, "-knn_sift", knn_sift_);
    pcl::console::parse_argument (argc, argv, "-go_color_sigma", go_color_sigma);
    pcl::console::parse_argument (argc, argv, "-go_use_supervoxels", go_use_supervoxels);
    pcl::console::parse_argument (argc, argv, "-prune_by_cc", prune_by_cc);
    pcl::console::parse_argument (argc, argv, "-visualize_graph", visualize_graph);
    pcl::console::parse_argument (argc, argv, "-training_dir_sift", training_dir_sift);
    pcl::console::parse_argument (argc, argv, "-training_input_structure", training_input_structure);
    pcl::console::parse_argument (argc, argv, "-use_codebook", use_codebook);
    pcl::console::parse_argument (argc, argv, "-cg_prune_hyp", cg_prune_hyp);
    pcl::console::parse_argument (argc, argv, "-seg_type", seg_type);
    pcl::console::parse_argument (argc, argv, "-add_planes", add_planes);
    pcl::console::parse_argument (argc, argv, "-use_hv", use_HV);
    pcl::console::parse_argument (argc, argv, "-max_our_cvfh_hyp_", max_our_cvfh_hyp_);
    pcl::console::parse_argument (argc, argv, "-normalize_ourcvfh_bins", normalize_ourcvfh_bins);
    pcl::console::parse_argument (argc, argv, "-models_dir", path);
    pcl::console::parse_argument (argc, argv, "-training_dir", training_dir);
    pcl::console::parse_argument (argc, argv, "-training_dir_shot", training_dir_shot);
    pcl::console::parse_argument (argc, argv, "-descriptor_name", desc_name);
    pcl::console::parse_argument (argc, argv, "-pcd_file", pcd_file);
    pcl::console::parse_argument (argc, argv, "-force_retrain", force_retrain);
    pcl::console::parse_argument (argc, argv, "-icp_iterations", icp_iterations);
    pcl::console::parse_argument (argc, argv, "-use_cache", use_cache);
    pcl::console::parse_argument (argc, argv, "-splits", splits);
    pcl::console::parse_argument (argc, argv, "-scene", scene);
    pcl::console::parse_argument (argc, argv, "-detect_clutter", detect_clutter);
    pcl::console::parse_argument (argc, argv, "-thres_hyp", thres_hyp_);
    pcl::console::parse_argument (argc, argv, "-icp_type", icp_type);
    pcl::console::parse_argument (argc, argv, "-vx_size_icp", VX_SIZE_ICP_);
    pcl::console::parse_argument (argc, argv, "-model_scale", model_scale);
    pcl::console::parse_argument (argc, argv, "-go_opt_type", go_opt_type);
    pcl::console::parse_argument (argc, argv, "-go_iterations", go_iterations);
    pcl::console::parse_argument (argc, argv, "-go_use_replace_moves", go_use_replace_moves);
    pcl::console::parse_argument (argc, argv, "-go_inlier_thres", go_inlier_thres);
    pcl::console::parse_argument (argc, argv, "-go_resolution", go_resolution);
    pcl::console::parse_argument (argc, argv, "-go_initial_temp", init_temp);
    pcl::console::parse_argument (argc, argv, "-go_require_normals", go_require_normals);
    pcl::console::parse_argument (argc, argv, "-go_regularizer", go_regularizer);
    pcl::console::parse_argument (argc, argv, "-go_clutter_regularizer", go_clutter_regularizer);
    pcl::console::parse_argument (argc, argv, "-go_radius_clutter", go_radius_clutter);
    pcl::console::parse_argument (argc, argv, "-go_log", go_log);
    pcl::console::parse_argument (argc, argv, "-go_init", go_init);
    pcl::console::parse_argument (argc, argv, "-idx_flann_fn", idx_flann_fn);
    pcl::console::parse_argument (argc, argv, "-PLAY", PLAY_);
    pcl::console::parse_argument (argc, argv, "-go_log_file", go_log_file_);
    pcl::console::parse_argument (argc, argv, "-test_sampling_density", test_sampling_density);
    pcl::console::parse_argument (argc, argv, "-tes_level", tes_level_);
    pcl::console::parse_argument (argc, argv, "-Z_DIST", Z_DIST_);
    pcl::console::parse_argument (argc, argv, "-pipelines_to_use", pipelines_to_use_);
    pcl::console::parse_argument (argc, argv, "-use_hough", use_hough);
    pcl::console::parse_argument (argc, argv, "-load_views", load_views);
    pcl::console::parse_argument (argc, argv, "-gc_min_dist_cf", min_dist_cf_);
    pcl::console::parse_argument (argc, argv, "-gc_dot_threshold", gc_dot_threshold_);

    MODELS_DIR_FOR_VIS_ = path;
    pcl::console::parse_argument (argc, argv, "-models_dir_vis", MODELS_DIR_FOR_VIS_);
    pcl::console::parse_argument (argc, argv, "-GT_DIR", GT_DIR_);
    MODELS_DIR_ = path;

    typedef pcl::PointXYZRGB PointT;

    std::cout << "VX_SIZE_ICP_" << VX_SIZE_ICP_ << std::endl;
    if (pcd_file.compare ("") == 0)
    {
        PCL_ERROR("Set the directory containing scenes\n");
        return -1;
    }

    if (path.compare ("") == 0)
    {
        PCL_ERROR("Set the directory containing the models using the -models_dir [dir] option\n");
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
        std::vector < std::string > files;
        std::string start = "";
        std::string ext = std::string ("pcd");
        bf::path dir = models_dir_path;
        getModelsInDirectory (dir, start, files, ext);
        std::cout << "Number of models in directory is:" << files.size() << std::endl;
    }

    recognizeAndVisualize<PointT> (pcd_file);
}
