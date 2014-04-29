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
        std::cout << pcd_file << " gt_name:" << gt_name << std::endl;
        boost::replace_all(gt_name, pcd_file, "");
        std::vector<std::string> strs;
        boost::split (strs, gt_name, boost::is_any_of ("\\/"));

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

    pcl::console::parse_argument (argc, argv, "-pcd_file", pcd_file);
    pcl::console::parse_argument (argc, argv, "-output_dir_images", OUTPUT_DIR_IMAGES_);
    pcl::console::parse_argument (argc, argv, "-models_dir", path);
    pcl::console::parse_argument (argc, argv, "-PLAY", PLAY_);
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
