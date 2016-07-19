/*
 *  Created on: Mar, 2014
 *      Author: Thomas Faeulhammer, Aitor Aldoma
 */
#include <opencv2/opencv.hpp>
#include <pcl/console/parse.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <v4r/recognition/model_only_source.h>
#include <v4r/common/pcl_opencv.h>
#include <v4r/io/filesystem.h>
#include <v4r/io/eigen.h>
#include <boost/filesystem/convenience.hpp>

#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;


//-m /media/Data/datasets/TUW/model_database_new -g /media/Data/datasets/TUW/annotations -i /media/Data/datasets/TUW/test_set/ -v

using namespace v4r;

typedef pcl::PointXYZRGB PointT;
typedef Model<PointT> ModelT;
typedef boost::shared_ptr<ModelT> ModelTPtr;

int
main (int argc, char ** argv)
{
    std::string gt_dir;
    std::string models_dir;
    std::string output_dir = "/tmp/annotated_images/";
    std::string scenes_dir;
    bool visualize = false;

    po::options_description desc("Create annotated images from annotation files\n======================================\n**Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("input_dir,i", po::value<std::string>(&scenes_dir)->required(), "Directory containing the point clouds to be annotated")
            ("models_dir,m", po::value<std::string>(&models_dir)->required(), "Directory containing the 3D object models")
            ("gt_dir,g", po::value<std::string>(&gt_dir)->required(), "directory containing ground-truth information about object pose in each point cloud")
            ("output_dir,o", po::value<std::string>(&output_dir)->default_value(output_dir), "output directory")
//            ("cloud_prefix,c", po::value<std::string>(&cloud_prefix)->default_value(cloud_prefix), "")
            ("visualize,v", po::bool_switch(&visualize), "visualize annotations")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return false;
    }

    try  { po::notify(vm); }
    catch(std::exception& e)  {
        std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
        return false;
    }

    ModelOnlySource<pcl::PointXYZRGBNormal, PointT> source;
    source.setPath (models_dir);
    source.setLoadViews (false);
    source.setLoadIntoMemory(false);
    source.generate();

    boost::shared_ptr<pcl::visualization::PCLVisualizer> vis_;

    std::vector<std::string> sub_folder_names = v4r::io::getFoldersInDirectory(scenes_dir);

    if(sub_folder_names.empty())
        sub_folder_names.push_back("");

    for (const std::string &sub_folder_name : sub_folder_names)
    {
        std::string scene_full_path = scenes_dir + "/" + sub_folder_name;

        std::vector < std::string > scene_files = v4r::io::getFilesInDirectory( scene_full_path, ".*.pcd", true );
        if ( !scene_files.empty() )
        {
            std::cout << "Number of scenes in directory is:" << scene_files.size () << std::endl;

            std::string annotations_dir = gt_dir + "/" + sub_folder_name;
            std::vector<std::string> gt_files = v4r::io::getFilesInDirectory( annotations_dir, ".*.txt", true );

            if( gt_files.empty() )
                std::cerr << "Could not find any annotations in " << annotations_dir << ". " << std::endl;

            if(visualize && !vis_)
                vis_.reset(new pcl::visualization::PCLVisualizer("gt_output_sv", true) );

            for (const std::string &scene_file : scene_files)
            {
                std::string scene_file_wo_ext = bf::basename(scene_file);
                const std::string gt_occ_check = scene_file_wo_ext + "_occlusion_";

                const std::string scene_file_path = scene_full_path + "/" + scene_file;
                typename pcl::PointCloud<PointT>::Ptr scene(new pcl::PointCloud<PointT>);
                pcl::io::loadPCDFile(scene_file_path, *scene);

                // reset view point otherwise pcl visualization is potentially messed up
                scene->sensor_orientation_ = Eigen::Quaternionf::Identity();
                scene->sensor_origin_ = Eigen::Vector4f::Zero(4);

                //            pcl::visualization::PointCloudColorHandlerRGBField<PointT> scene_handler(single_scenes_[i]);
                //            vis_->addPointCloud(single_scenes_[i], scene_handler, "single_view_scene");

                if (visualize) {
                    vis_->removeAllPointClouds();
                    pcl::visualization::PointCloudColorHandlerRGBField<PointT> scene_handler(scene);
                    vis_->addPointCloud(scene, scene_handler, scene_file_path);
                }

                pcl::PointCloud<PointT> gt_cloud;

                for(const std::string &gt_file : gt_files)
                {
                    if(gt_file.compare(0, scene_file_wo_ext.size(), scene_file_wo_ext) == 0
                            && gt_file.compare(0, gt_occ_check.size(), gt_occ_check)
                            && gt_file.compare("times.txt") )   // iterate over all ground truth files for the same scen
                    {
                        std::cout << gt_file << std::endl;

                        std::string model_name = bf::basename(gt_file);
                        model_name = model_name.substr(scene_file_wo_ext.size() + 1);
                        model_name = model_name.substr(0, model_name.find_last_of("_"));

                        ModelTPtr pModel;
                        source.getModelById(model_name, pModel);

                        const Eigen::Matrix4f gt_pose = v4r::io::readMatrixFromFile( annotations_dir+"/"+gt_file );

                        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = pModel->getAssembled(3);
                        typename pcl::PointCloud<PointT>::Ptr model_aligned(new pcl::PointCloud<PointT>());
                        pcl::transformPointCloud(*model_cloud, *model_aligned, gt_pose);
                        gt_cloud += *model_aligned;

                        if(visualize) {
                            pcl::visualization::PointCloudColorHandlerRGBField<PointT> model_handler(model_aligned);
                            vis_->addPointCloud(model_aligned, model_handler, gt_file);
                        }
                    }
                }
                std::cout << std::endl;

                if (visualize) {
                    vis_->spin();
                }

                // write images
                const std::string out_gt_fn = output_dir +"/gt/"+ sub_folder_name +"/"+ scene_file_wo_ext+".jpg";
                const std::string out_scene_fn = output_dir +"/scenes/"+ sub_folder_name +"/"+ scene_file_wo_ext+".jpg";
                v4r::io::createDirForFileIfNotExist(out_gt_fn);
                v4r::io::createDirForFileIfNotExist(out_scene_fn);
                cv::imwrite(out_gt_fn, ConvertUnorganizedPCLCloud2Image(gt_cloud, false, 255.0, 255.0, 255.0));
                cv::imwrite(out_scene_fn, ConvertUnorganizedPCLCloud2Image(*scene, false, 255.0, 255.0, 255.0));
            }
        }
        else
        {
            std::cerr << "There are no .pcd files in " << scene_full_path << "!" << std::endl;
        }
    }
}
