/*
 * create_images.cpp
 *
 *  Created on: Mar, 2014
 *      Author: Aitor Aldoma, Thomas Faeulhammer
 */
#include <opencv2/opencv.hpp>
#include <pcl/console/parse.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <v4r/recognition/model_only_source.h>
#include <v4r/common/pcl_opencv.h>
#include <v4r/io/filesystem.h>
#include <v4r/io/eigen.h>

// ./create_images -models_dir /media/Data/datasets/TUW/models/ -GT_DIR /home/thomas/Projects/thomas.faeulhammer/eval/set_00015_0_mv/ -pcd_file /media/Data/datasets/TUW/test_set/set_00015/ -output_dir_images /media/Data/datasets/TUW/annotated_images_tmp -visualize 1
// ./create_images -models_dir /media/Data/datasets/TUW/models/ -GT_DIR /media/Data/datasets/TUW/annotations_cvww/ -output_dir_images /media/Data/datasets/TUW/annotated_images_tmp_new/ -pcd_file /media/Data/datasets/TUW/test_set/ -visualize 0


//std::string GT_DIR_;
//std::string MODELS_DIR_;
//std::string OUTPUT_DIR_IMAGES_;
//std::string SCENES_DIR = "";

namespace v4r
{

template<typename PointT>
class imageCreator
{
private:
    boost::shared_ptr < v4r::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>
            > source_;

public:
    std::string gt_dir_;
    std::string models_dir_;
    std::string output_dir_;
    std::string scenes_dir_;
    bool visualize_;

    imageCreator()
    {
        source_.reset(new v4r::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>);

        scenes_dir_ = "";
        visualize_ = true;
    }

    void init_source()
    {
        source_->setPath (models_dir_);
        source_->setLoadViews (false);
        source_->setLoadIntoMemory(false);
        std::string test = "irrelevant";
        source_->generate (test);
    }

    void recognizeAndVisualize(const std::string & sub_folder_name = "");

    static void printUsage(int argc, char ** argv);
};

template<typename PointT>
void imageCreator<PointT>::recognizeAndVisualize (const std::string & sub_folder_name)
{
    typedef v4r::Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;

    std::string scene_full_path (scenes_dir_);
    scene_full_path.append("/");
    scene_full_path.append(sub_folder_name);

    bf::path or_path = output_dir_;
    if(!bf::exists(or_path))
    {
        bf::create_directory(or_path);
    }

    std::string scene_images_path;
    std::string scene_images_gt_path;

    {
        std::stringstream t;
        t << output_dir_  << "/scenes/";
        bf::path scene_images = t.str();
        if(!bf::exists(scene_images))
        {
            bf::create_directory(scene_images);
        }
        t << sub_folder_name << "/";
        scene_images_path = t.str();
        scene_images = scene_images_path;
        if(!bf::exists(scene_images))
        {
            bf::create_directory(scene_images);
        }
    }

    {
        std::stringstream t;
        t << output_dir_ << "/gt/";
        bf::path gt_images = t.str();
        if(!bf::exists(gt_images))
        {
            bf::create_directory(gt_images);
        }
        t << sub_folder_name << "/";
        scene_images_gt_path = t.str();
        gt_images = t.str();
        if(!bf::exists(gt_images))
        {
            bf::create_directory(gt_images);
        }
    }

    std::vector < std::string > scene_files;
    if (v4r::io::getFilesInDirectory( scene_full_path, scene_files, "", ".*.pcd", true) != -1)
    {
        std::cout << "Number of scenes in directory is:" << scene_files.size () << std::endl;

        std::vector < std::string > gt_files;
        std::string annotations_dir (gt_dir_);
        annotations_dir.append("/");
        annotations_dir.append(sub_folder_name);

        if( v4r::io::getFilesInDirectory( annotations_dir, gt_files, "", ".*.txt", true) == -1)
        {
            std::cerr << "Could not find any annotations in " << annotations_dir << ". " << std::endl;
        }
        pcl::visualization::PCLVisualizer gt_output_sv ("gt_output_sv", true);

        for (size_t i = 0; i < scene_files.size (); i++)
        {
            std::vector<std::string> strs;
            boost::split (strs, scene_files[i], boost::is_any_of ("."));
            std::string scene_file_wo_ext = strs[0];
            std::stringstream gt_occ_check_ss;
            gt_occ_check_ss << scene_file_wo_ext << "_occlusion_";

            gt_output_sv.removeAllPointClouds();
            std::stringstream scene_full_file_path_ss;
            scene_full_file_path_ss << scene_full_path << "/" << scene_files[i];
            typename pcl::PointCloud<PointT>::Ptr pScenePCl(new pcl::PointCloud<PointT>);
            pcl::io::loadPCDFile(scene_full_file_path_ss.str(), *pScenePCl);
            pcl::visualization::PointCloudColorHandlerRGBField<PointT> scene_handler(pScenePCl);
            gt_output_sv.addPointCloud(pScenePCl, scene_handler, scene_full_file_path_ss.str());
            //            pcl::visualization::PointCloudColorHandlerRGBField<PointT> scene_handler(single_scenes_[i]);
            //            gt_output_sv.addPointCloud(single_scenes_[i], scene_handler, "single_view_scene");

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr gt_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

            for(size_t pose_file_id=0; pose_file_id<gt_files.size(); pose_file_id++)
            {
                if(gt_files[pose_file_id].compare(0, scene_file_wo_ext.size(), scene_file_wo_ext) == 0
                        && gt_files[pose_file_id].compare(0, gt_occ_check_ss.str().size(), gt_occ_check_ss.str()))
                {
                    std::cout << gt_files[pose_file_id] << std::endl;
                    std::string model_name = gt_files[pose_file_id].substr(scene_file_wo_ext.size() + 1);
                    size_t found = model_name.find_last_of("_");
                    std::string times_text ("times.txt");
                    if (!std::strcmp(model_name.c_str(), times_text.c_str()))
                    {
                        std::cout << "skipping this one" << std::endl;
                        continue;
                    }
                    model_name = model_name.substr(0,found) + ".pcd";
                    std::cout << "Model: " << model_name << std::endl;
                    ModelTPtr pModel;
                    source_->getModelById(model_name, pModel);

                    std::stringstream gt_full_file_path_ss;
                    gt_full_file_path_ss << annotations_dir << "/" << gt_files[pose_file_id];
                    Eigen::Matrix4f transform;
                    v4r::io::readMatrixFromFile(gt_full_file_path_ss.str(), transform);

                    typename pcl::PointCloud<PointT>::ConstPtr model_cloud = pModel->getAssembled(0.003f);
                    typename pcl::PointCloud<PointT>::Ptr model_aligned(new pcl::PointCloud<PointT>());
                    pcl::transformPointCloud(*model_cloud, *model_aligned, transform);

                    pcl::visualization::PointCloudColorHandlerRGBField<PointT> model_handler(model_aligned);
                    gt_output_sv.addPointCloud(model_aligned, model_handler, gt_full_file_path_ss.str());

                    *gt_cloud += *model_aligned;
                }
            }
            std::cout << std::endl;
            if (visualize_)
            {
                gt_output_sv.spin();
            }

            //            files_to_recognize.push_back (file);

            //transform gt_cloud to organized point cloud and then to image
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr gt_cloud_organized(new pcl::PointCloud<pcl::PointXYZRGB>);
            gt_cloud_organized->width = pScenePCl->width;
            gt_cloud_organized->height = pScenePCl->height;
            gt_cloud_organized->is_dense = pScenePCl->is_dense;
            gt_cloud_organized->points.resize(pScenePCl->points.size());
            for(size_t kk=0; kk < gt_cloud_organized->points.size(); kk++)
            {
                gt_cloud_organized->points[kk].x = gt_cloud_organized->points[kk].y = gt_cloud_organized->points[kk].z =
                        std::numeric_limits<float>::quiet_NaN();

                gt_cloud_organized->points[kk].r = gt_cloud_organized->points[kk].g = gt_cloud_organized->points[kk].b = 0;
            }

            float f = 525.f;
            float cx = (static_cast<float> (pScenePCl->width) / 2.f - 0.5f);
            float cy = (static_cast<float> (pScenePCl->height) / 2.f - 0.5f);

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
                        if ((uu >= static_cast<int> (pScenePCl->width)) ||
                                (vv >= static_cast<int> (pScenePCl->height)) || (uu < 0) || (vv < 0))
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
            }

            //transform scene_cloud to organized point cloud and then to image
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr pScenePCl_organized(new pcl::PointCloud<pcl::PointXYZRGB>);
            pScenePCl_organized->width = pScenePCl->width;
            pScenePCl_organized->height = pScenePCl->height;
            pScenePCl_organized->is_dense = pScenePCl->is_dense;
            pScenePCl_organized->points.resize(pScenePCl->points.size());
            for(size_t kk=0; kk < pScenePCl_organized->points.size(); kk++)
            {
                pScenePCl_organized->points[kk].x = pScenePCl_organized->points[kk].y = pScenePCl_organized->points[kk].z =
                        std::numeric_limits<float>::quiet_NaN();

                pScenePCl_organized->points[kk].r = pScenePCl_organized->points[kk].g = pScenePCl_organized->points[kk].b = 0;
            }

            for (size_t kk = 0; kk < pScenePCl->points.size (); kk++)
            {
                float x = pScenePCl->points[kk].x;
                float y = pScenePCl->points[kk].y;
                float z = pScenePCl->points[kk].z;
                int u = static_cast<int> (f * x / z + cx);
                int v = static_cast<int> (f * y / z + cy);

                for(int uu = (u-ws2); uu < (u+ws2); uu++)
                {
                    for(int vv = (v-ws2); vv < (v+ws2); vv++)
                    {
                        //Not out of bounds
                        if ((uu >= static_cast<int> (pScenePCl->width)) ||
                                (vv >= static_cast<int> (pScenePCl->height)) || (uu < 0) || (vv < 0))
                            continue;

                        float z_oc = pScenePCl_organized->at (uu, vv).z;

                        if(pcl_isnan(z_oc))
                        {
                            pScenePCl_organized->at (uu, vv) = pScenePCl->points[kk];
                        }
                        else
                        {
                            if(z < z_oc)
                            {
                                pScenePCl_organized->at (uu, vv) = pScenePCl->points[kk];
                            }
                        }
                    }
                }
            }

            std::stringstream rel_path;
            rel_path << scene_images_gt_path << "/";
            bf::path p = rel_path.str();
            if(!bf::exists(p))
            {
                bf::create_directory(p);
            }

            std::stringstream image_path;
            image_path << rel_path.str() << "/" << scene_file_wo_ext << ".jpg";

            cv::Mat_ < cv::Vec3b > colorImage;
            PCLOpenCV::ConvertPCLCloud2Image<PointT> (gt_cloud_organized, colorImage);
            /*cv::namedWindow("image");
                cv::imshow("image", colorImage);
                cv::waitKey(0);*/
            cv::imwrite(image_path.str(), colorImage);

            rel_path.str("");
            rel_path << scene_images_path << "/";
            p = rel_path.str();
            if(!bf::exists(p))
            {
                bf::create_directory(p);
            }

            image_path.str("");
            image_path << rel_path.str() << "/" << scene_file_wo_ext << ".jpg";

            PCLOpenCV::ConvertPCLCloud2Image<PointT> (pScenePCl_organized, colorImage);
            cv::imwrite(image_path.str(), colorImage);
        }
    }
    else
    {
        PCL_ERROR("You should pass a directory\n");
        return;
    }
}

    template<typename PointT>
    void imageCreator<PointT>::printUsage(int argc, char ** argv)
    {
        (void)argc;
        std::cout << std::endl << std::endl
                  << "Usage " << argv[0]
                  << "-models_dir /path/to/models/ "
                  << "-GT_DIR /path/to/annotations/ "
                  << "-pcd_file /path/to/input_PCDs/ "
                  << "-output_dir_images /path/to/output/ "
                  << "[-visualize 1] "
                  << std::endl << std::endl;
    }
}


int
main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PointT;
    v4r::imageCreator<PointT> imgCreator;
    pcl::console::parse_argument (argc, argv, "-pcd_file", imgCreator.scenes_dir_);
    pcl::console::parse_argument (argc, argv, "-output_dir_images", imgCreator.output_dir_);
    pcl::console::parse_argument (argc, argv, "-models_dir", imgCreator.models_dir_);
    pcl::console::parse_argument (argc, argv, "-GT_DIR", imgCreator.gt_dir_);
    pcl::console::parse_argument (argc, argv, "-visualize", imgCreator.visualize_);


    if (imgCreator.scenes_dir_.compare ("") == 0)
    {
        PCL_ERROR("Set the directory containing scenes. Usage -pcd_file files [dir].\n");
        v4r::imageCreator<PointT>::printUsage(argc, argv);
        return -1;
    }

    if (imgCreator.output_dir_.compare ("") == 0)
    {
        PCL_ERROR("Set the directory containing the models using the -models_dir [dir] option\n");
        v4r::imageCreator<PointT>::printUsage(argc, argv);
        return -1;
    }

    bf::path models_dir_path = imgCreator.models_dir_;
    if (!bf::exists (models_dir_path))
    {
        PCL_ERROR("Models dir path %s does not exist, use -models_dir [dir] option\n", imgCreator.models_dir_.c_str());
        v4r::imageCreator<PointT>::printUsage(argc, argv);
        return -1;
    }

    imgCreator.init_source();

    std::vector< std::string> sub_folder_names;
    if(!v4r::io::getFoldersInDirectory(imgCreator.scenes_dir_, "", sub_folder_names))
    {
        std::cerr << "No subfolders in directory " << imgCreator.scenes_dir_ << ". " << std::endl;
        sub_folder_names.push_back("");
    }
    for (size_t sub_folder_id=0; sub_folder_id < sub_folder_names.size(); sub_folder_id++)
    {
        imgCreator.recognizeAndVisualize (sub_folder_names[ sub_folder_id ]);
    }
}
