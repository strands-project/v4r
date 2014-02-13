#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <fstream>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/common/transforms.h>
#include <boost/filesystem.hpp>

namespace bf = boost::filesystem;

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

inline bool
writeFloatToFile (std::string file, float value)
{
  std::ofstream out (file.c_str ());
  if (!out)
  {
    std::cout << "Cannot open file.\n";
    return false;
  }

  out << value;
  out.close ();

  return true;
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

void
readPose (std::string path, Eigen::Matrix4f & pose, int scene_id)
{
  std::ifstream in_pose;
  in_pose.open (path.c_str (), std::ifstream::in);

  char linebuf[1024];
  int row = 0;
  while (in_pose.getline (linebuf, 1024))
  {
    std::string line (linebuf);
    std::vector<std::string> strs_2;
    boost::split (strs_2, line, boost::is_any_of (" "));
    for (size_t j = 0; j < (strs_2.size () - 1); j++)
    {
      pose (row, j) = atof (strs_2[j].c_str ());
    }
    row++;
  }

  std::cout << pose << std::endl;

  Eigen::Matrix4f scale_models;
  scale_models.setIdentity ();

  if (scene_id == 6 && (path.compare ("/home/aitor/data/Mians_dataset/GroundTruth_3Dscenes/parasaurolophus-rs6.xf") == 0))
  {
    std::cout << "Pose found" << std::endl;
    pose = pose * scale_models;
    return;
  }

  //scale_models (0, 0) = scale_models (1, 1) = scale_models (2, 2) = 0.001;

  Eigen::Matrix4f reflec_x;
  reflec_x.setIdentity ();
  reflec_x (0, 0) = -1;

  Eigen::Matrix4f reflec_z;
  reflec_z.setIdentity ();
  reflec_z (2, 2) = -1;

  Eigen::Vector3f z_vector = Eigen::Vector3f::UnitZ ();
  Eigen::Affine3f rotZ (Eigen::AngleAxisf (0.0174532925 * -90, z_vector));

  //modify the pose to use the same coordinate system that we use...
  Eigen::Matrix4f trans;
  trans.setIdentity ();
  trans = scale_models * reflec_x * rotZ * reflec_z * pose;
  //trans = scale_models * reflec_x * reflec_z * pose;
  trans(0,3) *= 0.001f;
  trans(1,3) *= 0.001f;
  trans(2,3) *= 0.001f;
  pose = trans;

  std::cout << pose << std::endl;

  //pose = pose * scale_models;
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

void
readOcclusionFile (std::map<int, std::map<std::string, float> > & occlusions)
{
  std::ifstream in;
  in.open ("/home/aitor/data/Mians_dataset/GroundTruth_3Dscenes/occlusion.txt", std::ifstream::in);

  char linebuf[1024];
  in.getline (linebuf, 1024); //discard first line
  while (in.getline (linebuf, 1024))
  {
    std::string line (linebuf);
    std::vector<std::string> strs_2;
    boost::split (strs_2, line, boost::is_any_of ("\t"));
    //std::cout << strs_2[0] << " " << strs_2[1] << " " << strs_2[2] << std::endl;
    int s_id = atoi (strs_2[0].c_str ());
    std::string id = std::string (strs_2[1]);
    float occlusion = atof (strs_2[2].c_str ());

    std::map<int, std::map<std::string, float> >::iterator it;
    it = occlusions.find(s_id);
    if(it == occlusions.end())
    {
      std::map<std::string, float> map_occ;
      map_occ.insert(std::pair<std::string, float>(id, occlusion / 100.f));
      occlusions.insert(std::pair<int, std::map<std::string, float> >(s_id, map_occ));
    }
    else
    {
      it->second.insert(std::pair<std::string, float>(id, occlusion / 100.f));
    }
  }
}

int
main (int argc, char ** argv)
{
  std::string occlusion_file = "";
  std::string gt_dir = "";
  std::string scene_dir = "";
  std::string out_dir = "/home/aitor/data/Mians_dataset/gt_or_format/";

  pcl::console::parse_argument (argc, argv, "-occlusion_file", occlusion_file);
  pcl::console::parse_argument (argc, argv, "-gt_dir", gt_dir);
  pcl::console::parse_argument (argc, argv, "-scene_dir", scene_dir);

  std::vector<std::string> files;
  std::string start = "";
  std::string ext = std::string ("pcd");
  bf::path dir = scene_dir;
  getModelsInDirectory (dir, start, files, ext);
  std::cout << "Number of scenes in directory is:" << files.size () << std::endl;

  std::sort (files.begin (), files.end (), sortFiles);
  std::vector<std::string> model_out_names;
  std::vector<std::string> model_pose_names;
  model_out_names.push_back ("chef");
  model_out_names.push_back ("para");
  model_out_names.push_back ("chicken");
  model_out_names.push_back ("trex");

  model_pose_names.push_back ("chef");
  model_pose_names.push_back ("parasaurolophus");
  model_pose_names.push_back ("chicken");
  model_pose_names.push_back ("T-rex");

  std::map<int, std::map<std::string, float> > occlusions;
  readOcclusionFile(occlusions);

  for (size_t i = 0; i < files.size (); i++)
  {
    std::string model_path (files[i]);
    boost::replace_all (model_path, ".ply.pcd", "");
    std::cout << model_path << std::endl;

    std::string scene_name (files[i]);
    boost::replace_all (scene_name, ".pcd", "");

    for (size_t m = 0; m < model_pose_names.size (); m++)
    {
      std::stringstream pose_file;
      pose_file << gt_dir << "/" << model_pose_names[m] << "-" << model_path << ".xf";
      std::cout << pose_file.str () << std::endl;

      bf::path pose_path = pose_file.str ();
      if (bf::exists (pose_path))
      {
        Eigen::Matrix4f pose;
        readPose (pose_file.str (), pose, (i + 1));

        //write pose file
        std::stringstream pose_file_out;
        pose_file_out << out_dir << "/" << scene_name << "_" << model_out_names[m] << "_0.txt";
        writeMatrixToFile (pose_file_out.str (), pose);
        //write occlusion file
        std::stringstream occ_file_out;
        occ_file_out << out_dir << "/" << scene_name << "_occlusion_" << model_out_names[m] << "_0.txt";
        float occlusion;
        std::map<int, std::map<std::string, float> >::iterator it_occlusions;
        std::map<std::string, float>::iterator it_occlusions_scene;

        it_occlusions = occlusions.find(static_cast<int>(i+1));
        if(it_occlusions != occlusions.end())
        {
          it_occlusions_scene = it_occlusions->second.find(model_out_names[m]);
          if(it_occlusions_scene != it_occlusions->second.end())
          {
            writeFloatToFile(occ_file_out.str (), it_occlusions_scene->second);
          }
        }
      }
    }
  }
}
