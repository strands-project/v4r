/*
 * mit.cpp
 *
 *  Created on: Feb 20, 2013
 *      Author: aitor
 */

#include <pcl/console/parse.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <faat_pcl/utils/filesystem_utils.h>

/*template<typename PointT>
void visualizeGroundTruth(std::string & scene_file, pcl::visualization::PCLVisualizer & vis, int viewport, std::string & models_folder,
                          std::vector<typename pcl::PointCloud<PointT>::Ptr> & gt_models_aligned,
                          std::vector<std::string> & gt_ids)
{
  std::vector < std::string > strs;
  boost::split (strs, scene_file, boost::is_any_of ("/"));
  std::string cloud_fn = strs[strs.size () - 1];

  std::string model_path (scene_file);
  boost::replace_all (model_path, cloud_fn, "");

  strs.clear();
  boost::split (strs, cloud_fn, boost::is_any_of ("."));
  std::string cloud_n = strs[0];

  std::cout << scene_file << " " << cloud_fn << " " << cloud_n << std::endl;

  std::vector < std::string > pose_filenames;
  bf::path model_dir = model_path;
  std::string ext = std::string("txt");
  getDesiredPaths(model_dir, pose_filenames, cloud_n, ext);

  for(size_t i=0; i < pose_filenames.size(); i++)
  {

    std::vector < std::string > strs;
    boost::split (strs, pose_filenames[i], boost::is_any_of ("_"));
    std::string model_name = strs[1];
    boost::replace_all (model_name, ".txt", "");

    std::stringstream file_to_read;
    file_to_read << model_path << "/" << pose_filenames[i];
    std::cout << file_to_read.str() << " " << model_name << std::endl;

    std::string file_pose(file_to_read.str());
    //std::ifstream infile(file_pose.c_str(), ifstream::in);
    boost::replace_all (file_pose, "///", "/");
    std::ifstream in;
    std::cout << "Trying to open..." << file_pose << std::endl;
    in.open (file_pose.c_str (), std::ifstream::in);

    if(in) {

      char linebuf[256];
      in.getline (linebuf, 256);
      std::string pose_line (linebuf);
      std::cout << pose_line << std::endl;

      in.close();

      strs.clear();
      boost::split (strs, pose_line, boost::is_any_of (" "));

      std::vector<float> non_empty;
      for(size_t k=0; k < strs.size(); k++)
      {
        if(strs[k] != "") {
          non_empty.push_back(atof(strs[k].c_str()));
        }
      }

      Eigen::Vector3f trans(non_empty[0],non_empty[1],non_empty[2]);
      Eigen::Quaternionf rot(non_empty[3],non_empty[4],non_empty[5],non_empty[6]);

      Eigen::Matrix3f rot_mat = rot.toRotationMatrix();
      Eigen::Matrix4f pose_mat;
      pose_mat.block<3,3>(0,0) = rot_mat;
      pose_mat.block<3,1>(0,3) = trans;

      typename pcl::PointCloud<PointT>::Ptr gt_model (new pcl::PointCloud<PointT>);
      std::stringstream  model_file;
      model_file << models_folder << "/" << model_name << ".pcd";
      pcl::io::loadPCDFile (model_file.str(), *gt_model);

      pcl::transformPointCloud(*gt_model, *gt_model, pose_mat);

      std::stringstream name;
      name << "gt_model" << i;
      pcl::visualization::PointCloudColorHandlerRandom<PointT> random_handler (gt_model);
      vis.addPointCloud<PointT> (gt_model, random_handler, name.str (), viewport);

      gt_models_aligned.push_back(gt_model);
      gt_ids.push_back(model_name);
    } else {
      std::cout << "Could not read file..." << file_to_read.str() << std::endl;
    }

  }
}*/

inline bool
writeMatrixToFile (std::string file, Eigen::Matrix4f & matrix)
{
    std::ofstream out (file.c_str ());
    if (!out)
    {
        std::cout << "Cannot open file.\n";
        return false;
    }

    for (size_t i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            out << matrix (i, j);
            if (!(i == 3 && j == 3))
                out << " ";
        }
    }
    out.close ();

    return true;
}


int
main (int argc, char ** argv)
{
  std::string models_dir = "";
  std::string input_gt_dir = "";
  std::string output_gt_dir = "";

  pcl::console::parse_argument (argc, argv, "-models_dir", models_dir);
  pcl::console::parse_argument (argc, argv, "-input_gt_dir", input_gt_dir);
  pcl::console::parse_argument (argc, argv, "-output_gt_dir", output_gt_dir);

  //iterate through input_gt_dir, creating folder structure on output_gt_dir together with pose files
  std::vector<std::string> files;
  std::string pattern_scenes = ".*cloud.*.txt";

  bf::path input_dir = input_gt_dir;
  std::string so_far = "";
  faat_pcl::utils::getFilesInDirectoryRecursive(input_dir, so_far, files, pattern_scenes);

  for(size_t i=0; i < files.size(); i++)
  {
      std::cout << files[i] << std::endl;

      std::vector < std::string > strs_2;
      boost::split (strs_2, files[i], boost::is_any_of ("/\\"));
      std::string folder = strs_2[0];
      std::string pose_file = strs_2[1];

      std::stringstream out_dir_seq;
      out_dir_seq << output_gt_dir << "/" << folder;
      bf::path out_dir_path = out_dir_seq.str();
      if(!bf::exists(out_dir_path))
          bf::create_directory(out_dir_path);

      std::vector < std::string > strs;
      boost::split (strs, pose_file, boost::is_any_of ("_"));

      std::vector < std::string > strs3;
      boost::split (strs3, strs[1], boost::is_any_of ("."));

      std::stringstream out_name_ss;
      out_name_ss << output_gt_dir << "/" << folder << "/" << strs[0] << "_" << strs3[0] << "_0.txt";

      std::cout << out_name_ss.str() << std::endl;

      //read pose file and save it to out_dir
      std::stringstream file_to_read;
      file_to_read << input_gt_dir << "/" << files[i];
      std::cout << file_to_read.str() << std::endl;

      std::ifstream in;
      std::cout << "Trying to open..." << file_to_read.str().c_str () << std::endl;
      in.open (file_to_read.str().c_str (), std::ifstream::in);

      if(in)
      {

        char linebuf[256];
        in.getline (linebuf, 256);
        std::string pose_line (linebuf);
        std::cout << pose_line << std::endl;

        in.close();

        std::vector < std::string > strs;
        boost::split (strs, pose_line, boost::is_any_of (" "));

        std::vector<float> non_empty;
        for(size_t k=0; k < strs.size(); k++)
        {
          if(strs[k] != "") {
            non_empty.push_back(atof(strs[k].c_str()));
          }
        }

        Eigen::Vector3f trans(non_empty[0],non_empty[1],non_empty[2]);
        Eigen::Quaternionf rot(non_empty[3],non_empty[4],non_empty[5],non_empty[6]);

        Eigen::Matrix3f rot_mat = rot.toRotationMatrix();
        Eigen::Matrix4f pose_mat = Eigen::Matrix4f::Identity();
        pose_mat.block<3,3>(0,0) = rot_mat;
        pose_mat.block<3,1>(0,3) = trans;
        std::cout << pose_mat << std::endl;

        writeMatrixToFile(out_name_ss.str(), pose_mat);
      }
  }

}
