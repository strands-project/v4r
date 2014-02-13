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

namespace bf = boost::filesystem;

namespace evalute_utils
{
    void
    getFilesInDirectory (bf::path & dir, std::string & rel_path_so_far, std::vector<std::string> & relative_paths, std::string & ext)
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
          getFilesInDirectory (curr_path, so_far, relative_paths, ext);
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
    getDirectories (bf::path & dir, std::vector<std::string> & relative_paths)
    {
      bf::directory_iterator end_itr;
      for (bf::directory_iterator itr (dir); itr != end_itr; ++itr)
      {
        //check if its a directory, then get models in it
        if (bf::is_directory (*itr))
        {
#if BOOST_FILESYSTEM_VERSION == 3
            std::string path = (itr->path ().filename ()).string ();
#else
            std::string path = (itr->path ()).filename ();
#endif
          relative_paths.push_back(path);
        }
      }
    }
}

inline void
readCSVFile (std::string & file,
               std::map<int, std::vector<std::string> > & ids_per_scene, std::map<int, std::vector<Eigen::Matrix4f> > & transforms)
{
  std::ifstream in;
  in.open (file.c_str (), std::ifstream::in);

  std::string s = "";
  std::getline (in, s); //ignore first line
  int last_scene = -1;
  std::vector<std::string> obj_ids;
  std::vector<Eigen::Matrix4f> obj_trans;
  //ids_per_scene.resize(max_scene+1);
  //transforms.resize(max_scene+1);

  while (std::getline (in, s))
  {
    //std::cout << s << std::endl;
    std::vector<std::string> strs;
    boost::split (strs, s, boost::is_any_of (","));
    int scene = atoi (strs[3].c_str ());
    std::string obj_id = strs[4];
    Eigen::Matrix4f trans;
    trans.setIdentity ();
    trans (0, 0) = static_cast<float> (atof (strs[5].c_str ()));
    trans (0, 1) = static_cast<float> (atof (strs[6].c_str ()));
    trans (0, 2) = static_cast<float> (atof (strs[7].c_str ()));
    trans (1, 0) = static_cast<float> (atof (strs[8].c_str ()));
    trans (1, 1) = static_cast<float> (atof (strs[9].c_str ()));
    trans (1, 2) = static_cast<float> (atof (strs[10].c_str ()));
    trans (2, 0) = static_cast<float> (atof (strs[11].c_str ()));
    trans (2, 1) = static_cast<float> (atof (strs[12].c_str ()));
    trans (2, 2) = static_cast<float> (atof (strs[13].c_str ()));
    trans (0, 3) = static_cast<float> (atof (strs[14].c_str ()));
    trans (1, 3) = static_cast<float> (atof (strs[15].c_str ()));
    trans (2, 3) = static_cast<float> (atof (strs[16].c_str ()));

    if(obj_id.compare("object_x") != 0) {
      std::map<int, std::vector<std::string> >::iterator it_ids;
      it_ids = ids_per_scene.find(scene);
      if(it_ids == ids_per_scene.end())
      {
        //does not exist, insert in map
        std::vector<std::string> ids;
        ids.push_back(obj_id);
        std::vector<Eigen::Matrix4f> transformss;
        transformss.push_back(trans);

        ids_per_scene.insert(std::make_pair(scene, ids));
        transforms.insert(std::make_pair(scene, transformss));
      }
      else
      {
        it_ids->second.push_back (obj_id);
        std::map<int, std::vector<Eigen::Matrix4f> >::iterator it_trans = transforms.find(scene);
        it_trans->second.push_back (trans);
      }
    }
  }

  //std::cout << transforms.size () << " " << ids_per_scene.size () << std::endl;
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

    std::string test_directory;
    std::string MODELS_DIR_;
    std::string gt_output_dir;
    std::string new_to_old_poses_dir;

    pcl::console::parse_argument (argc, argv, "-test_directory", test_directory);
    pcl::console::parse_argument (argc, argv, "-models_dir", MODELS_DIR_);
    pcl::console::parse_argument (argc, argv, "-gt_output_dir", gt_output_dir);
    pcl::console::parse_argument (argc, argv, "-new_to_old_poses_dir", new_to_old_poses_dir);

    bf::path test_dir = test_directory;
    std::vector<std::string> directories;
    evalute_utils::getDirectories(test_dir, directories);

    std::sort(directories.begin(), directories.end());

    for(size_t i=0; i < directories.size(); i++)
    {
      std::stringstream dir_path;
      dir_path << test_directory << "/" << directories[i];
      std::cout << dir_path.str() << std::endl;

      std::stringstream out_dir;
      out_dir << gt_output_dir << "/" << directories[i];
      bf::path out_dir_path = out_dir.str();
      if(!bf::exists(out_dir_path))
      {
          bf::create_directory(out_dir_path);
      }

      std::vector<std::string> strs;
      boost::split (strs, directories[i], boost::is_any_of ("/"));
      std::string extension = strs[strs.size () - 1];

      std::stringstream csv_file;
      csv_file << dir_path.str() << "/" << extension << ".bag.csv";

      std::string csv = csv_file.str();
      std::map<int, std::vector<std::string> > ids_per_scene_manual_gt;
      std::map<int, std::vector<Eigen::Matrix4f> > gt_per_scene_manual_gt;
      readCSVFile(csv, ids_per_scene_manual_gt, gt_per_scene_manual_gt);

      std::cout << dir_path.str() << std::endl;
      std::cout << ids_per_scene_manual_gt.size() << " " << gt_per_scene_manual_gt.size() << std::endl;

      std::map<int, std::vector<std::string> >::iterator s;
      for(s=ids_per_scene_manual_gt.begin(); s != ids_per_scene_manual_gt.end(); s++)
      {
          std::cout << "id:" << s->first << " " << s->second.size() << std::endl;
          std::map<int, std::vector<Eigen::Matrix4f> >::iterator pose_it;
          pose_it = gt_per_scene_manual_gt.find(s->first);

          std::stringstream scene_name;
          scene_name << "cloud_" << std::setw(10) << std::setfill('0') << s->first;
          for(size_t k=0; k < s->second.size(); k++)
          {
              //std::cout << s->second[k] << std::endl;
              //std::cout << pose_it->second[k] << std::endl;
              std::stringstream pose_file;
              pose_file << out_dir.str() << "/" << scene_name.str() << "_" << s->second[k] << "_0.txt";
              std::cout << pose_file.str() << std::endl;

              std::stringstream new_to_old_pose;
              new_to_old_pose << new_to_old_poses_dir << "/" << s->second[k] << ".txt";

              Eigen::Matrix4f new_to_old;
              readMatrixFromFile2(new_to_old_pose.str(), new_to_old);

              std::cout << new_to_old << std::endl;

              Eigen::Matrix4f pose = pose_it->second[k];
              pose = pose.inverse().eval();
              pose = pose * new_to_old;
              writeMatrixToFile (pose_file.str (), pose);
          }
      }
    }

    return -1;

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
      }
    }
  }
}
