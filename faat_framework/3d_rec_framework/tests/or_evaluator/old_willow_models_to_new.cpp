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

inline bool
readMatrixFromFile2 (std::string file, Eigen::Matrix4f & matrix, int ignore = 0)
{

  std::ifstream in;
  in.open (file.c_str (), std::ifstream::in);

  char linebuf[1024];
  in.getline (linebuf, 1024);
  std::string line (linebuf);
  std::vector < std::string > strs_2;
  boost::split (strs_2, line, boost::is_any_of (" "));

  std::vector < std::string > strs;
  for(size_t i=ignore; i < strs_2.size(); i++)
  {
      strs.push_back(strs_2[i]);
  }

  strs_2 = strs;

  for (int i = 0; i < 16; i++)
  {
    matrix (i / 4, i % 4) = static_cast<float> (atof (strs_2[i].c_str ()));
  }

  return true;
}

int
main (int argc, char ** argv)
{

    std::string path_old, path_new;
    std::string path_pcd_calib_file;
    std::string pose_old, pose_new;
    std::string model_id;
    std::string out_pose_file;

    pcl::console::parse_argument (argc, argv, "-path_old", path_old);
    pcl::console::parse_argument (argc, argv, "-path_new", path_new);
    pcl::console::parse_argument (argc, argv, "-path_pcd_calib_file", path_pcd_calib_file);
    pcl::console::parse_argument (argc, argv, "-pose_old", pose_old);
    pcl::console::parse_argument (argc, argv, "-pose_new", pose_new);
    pcl::console::parse_argument (argc, argv, "-model_id", model_id);
    pcl::console::parse_argument (argc, argv, "-out_pose_file", out_pose_file);

    bf::path models_dir_path_old = path_old;
    bf::path models_dir_path_new = path_new;

    if (!bf::exists (models_dir_path_old))
    {
        PCL_ERROR("Models dir path (OLD) %s does not exist\n", path_old.c_str());
        return -1;
    }

    if (!bf::exists (models_dir_path_new))
    {
        PCL_ERROR("Models dir path (NEW) %s does not exist\n", path_new.c_str());
        return -1;
    }

    Eigen::Matrix4f new_pose, old_pose;
    faat_pcl::rec_3d_framework::PersistenceUtils::readMatrixFromFile2(pose_new, new_pose);
    std::cout << "new_pose:" << new_pose << std::endl;
    new_pose = new_pose.inverse().eval();

    readMatrixFromFile2(pose_old, old_pose, 1);
    std::cout << "old_pose:" << old_pose << std::endl;

    std::string test = "irrelevant";

    boost::shared_ptr < faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>
            > source_old (new faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>);
    source_old->setPath (path_old);
    source_old->setLoadViews (false);
    source_old->setLoadIntoMemory(false);
    source_old->generate (test);

    boost::shared_ptr < faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>
            > source_new (new faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>);
    source_new->setPath (path_new);
    source_new->setLoadViews (false);
    source_new->setLoadIntoMemory(false);
    source_new->generate (test);

    typedef faat_pcl::rec_3d_framework::Model<pcl::PointXYZRGB> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;

    ModelTPtr model_old;
    bool found = source_old->getModelById (model_id, model_old);
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr old_model_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    old_model_cloud = model_old->getAssembled(-1);

    ModelTPtr model_new;
    found = source_new->getModelById (model_id, model_new);
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr new_model_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    new_model_cloud = model_new->getAssembled(-1);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr calib_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile(path_pcd_calib_file, *calib_cloud);

    //transform clouds
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr new_model_cloud_trans (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(*new_model_cloud, *new_model_cloud_trans, new_pose);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr old_model_cloud_trans (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(*old_model_cloud, *old_model_cloud_trans, old_pose);

    Eigen::Matrix4f new_to_old;
    new_to_old = old_pose.inverse() * new_pose;
    std::cout << new_to_old << std::endl;

    pcl::visualization::PCLVisualizer vis("CALIBRATING OLD TO NEW");
    int v1,v2,v3;
    vis.createViewPort(0,0,0.33,1,v1);
    vis.createViewPort(0.33,0,0.66,1,v2);
    vis.createViewPort(0.66,0,1,1,v3);
    vis.addPointCloud(calib_cloud, "calib", v1);
    vis.addPointCloud(old_model_cloud_trans, "old model", v2);
    vis.addPointCloud(new_model_cloud_trans, "new model", v3);
    vis.spin();

    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr new_model_cloud_trans (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::transformPointCloud(*new_model_cloud, *new_model_cloud_trans, new_to_old);

        pcl::visualization::PCLVisualizer vis("CALIBRATING OLD TO NEW");
        int v1,v2,v3;
        vis.createViewPort(0,0,0.33,1,v1);
        vis.createViewPort(0.33,0,0.66,1,v2);
        vis.createViewPort(0.66,0,1,1,v3);
        vis.addPointCloud(old_model_cloud, "old model", v1);
        vis.addPointCloud(new_model_cloud_trans, "new model", v2);

        {
            vis.addPointCloud(old_model_cloud, "old model_v3", v3);
            vis.addPointCloud(new_model_cloud_trans, "new model_v3", v3);
        }
        vis.addCoordinateSystem(0.3);
        vis.spin();
    }

    faat_pcl::rec_3d_framework::PersistenceUtils::writeMatrixToFile(out_pose_file, new_to_old);
    //recognizeAndVisualize<PointT> (pcd_file);
}
