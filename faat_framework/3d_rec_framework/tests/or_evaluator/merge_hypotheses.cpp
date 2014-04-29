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

std::string GT_DIR_;
std::string MODELS_DIR_;
std::string MODELS_DIR_FOR_VIS_;
std::string GT_DIR_TO_EVALUATE_;
float model_scale = 1.f;

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

int
main (int argc, char ** argv)
{

    std::string path = "";
    std::string pcd_file = "";
    std::string gt_dirs = "";
    std::string output_dir = "";
    std::string model_ext = "pcd";
    int source_models = 0;

    MODELS_DIR_FOR_VIS_ = path;

    pcl::console::parse_argument (argc, argv, "-source_models", source_models);
    pcl::console::parse_argument (argc, argv, "-models_dir", path);
    pcl::console::parse_argument (argc, argv, "-pcd_file", pcd_file);
    pcl::console::parse_argument (argc, argv, "-models_dir_vis", MODELS_DIR_FOR_VIS_);
    pcl::console::parse_argument (argc, argv, "-GT_DIR", GT_DIR_);
    pcl::console::parse_argument (argc, argv, "-GT_DIR_to_evaluate", GT_DIR_TO_EVALUATE_);

    pcl::console::parse_argument (argc, argv, "-gt_dirs", gt_dirs);
    pcl::console::parse_argument (argc, argv, "-output_dir", output_dir);
    pcl::console::parse_argument (argc, argv, "-model_ext", model_ext);

    MODELS_DIR_ = path;

    typedef pcl::PointXYZRGB PointT;

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
        bf::path dir = models_dir_path;
        getModelsInDirectory (dir, start, files, model_ext);
        std::cout << "Number of models in directory is:" << files.size() << std::endl;
    }

    bf::path out_dir_path = output_dir;
    if (!bf::exists (out_dir_path))
    {
        bf::create_directory(out_dir_path);
    }

    std::vector < std::string > strs_2;
    boost::split (strs_2, gt_dirs, boost::is_any_of (","));
    for(size_t i=0; i < strs_2.size(); i++)
    {
        std::cout << strs_2[i] << std::endl;

        GT_DIR_ = strs_2[i];

        if(source_models == 0)
        {

            faat_pcl::rec_3d_framework::or_evaluator::OREvaluator<PointT> or_eval;
            or_eval.setGTDir(GT_DIR_);
            or_eval.setModelsDir(MODELS_DIR_);
            or_eval.setModelFileExtension(model_ext);
            or_eval.setReplaceModelExtension(false);

            boost::shared_ptr < faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>
                    > source (new faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>);
            source->setPath (MODELS_DIR_);
            source->setLoadViews (false);
            source->setLoadIntoMemory(false);
            std::string test = "irrelevant";
            source->generate (test);
            or_eval.setDataSource(source);

            or_eval.setScenesDir(pcd_file);
            or_eval.loadGTData();

            or_eval.copyToDirectory(output_dir);
        }
        else
        {
            faat_pcl::rec_3d_framework::or_evaluator::OREvaluator<pcl::PointXYZ> or_eval;
            or_eval.setGTDir(GT_DIR_);
            or_eval.setModelsDir(MODELS_DIR_);
            or_eval.setModelFileExtension(model_ext);
            or_eval.setReplaceModelExtension(false);

            boost::shared_ptr < faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZ, pcl::PointXYZ>
                    > source (new faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZ, pcl::PointXYZ>);
            source->setPath (MODELS_DIR_);
            source->setLoadViews (false);
            source->setModelScale(model_scale);
            source->setLoadIntoMemory(false);

            std::string test = "irrelevant";
            std::cout << "calling generate" << std::endl;
            source->setExtension("ply");
            source->generate (test);
            or_eval.setDataSource(source);

            or_eval.setScenesDir(pcd_file);
            or_eval.loadGTData();

            or_eval.copyToDirectory(output_dir);
        }


    }

}
