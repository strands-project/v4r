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

template<typename PointT>
void
recognizeAndVisualize (std::string & scene_file)
{


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
        std::cout << files_to_recognize[i] << std::endl;
        typename pcl::PointCloud<PointT>::Ptr scene (new pcl::PointCloud<PointT>);
        pcl::io::loadPCDFile (files_to_recognize[i], *scene);

        vis.removeAllPointClouds();

        {
            pcl::visualization::PointCloudColorHandlerRGBField<PointT> scene_handler (scene);
            vis.addPointCloud<PointT> (scene, scene_handler, "scene_cloud_z_coloured", v1);
        }

        std::string file_to_recognize(files_to_recognize[i]);
        boost::replace_all (file_to_recognize, scene_file, "");
        boost::replace_all (file_to_recognize, ".pcd", "");
        std::string id_1 = file_to_recognize;

        or_eval.visualizeGroundTruth(vis, id_1, v2);
        vis.spin ();
    }
}

int
main (int argc, char ** argv)
{

    std::string path = "";
    std::string pcd_file = "";
    pcl::console::parse_argument (argc, argv, "-pcd_file", pcd_file);
    pcl::console::parse_argument (argc, argv, "-models_dir", path);
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
