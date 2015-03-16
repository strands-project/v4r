#include <pcl/console/parse.h>
#include <v4r/ORUtils/filesystem_utils.h>
#include <v4r/Registration/MvLMIcp.h>
#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

struct IndexPoint
{
  int idx;
};

POINT_CLOUD_REGISTER_POINT_STRUCT (IndexPoint,
(int, idx, idx)
)

int
main (int argc, char ** argv)
{
  typedef pcl::PointXYZRGB PointInT;

  std::string directory = "";
  std::string pose_str = "pose";
  float vx_size = 0.003f;
  double max_dist = 0.01f;
  int max_iterations = 10;
  int diff_type = 2;
  std::string output_directory = "";

  pcl::console::parse_argument (argc, argv, "-output_directory", output_directory);
  pcl::console::parse_argument (argc, argv, "-directory", directory);
  pcl::console::parse_argument (argc, argv, "-pose_str", pose_str);
  pcl::console::parse_argument (argc, argv, "-vx_size", vx_size);
  pcl::console::parse_argument (argc, argv, "-max_dist", max_dist);
  pcl::console::parse_argument (argc, argv, "-max_iterations", max_iterations);
  pcl::console::parse_argument (argc, argv, "-diff_type", diff_type);

  std::vector<std::string> to_process;
  std::string so_far = "";
  std::string pattern = ".*cloud.*.pcd";
  v4r::utils::getFilesInDirectory(directory, to_process, so_far, pattern, true);

  std::sort(to_process.begin(), to_process.end());

  std::vector<pcl::PointCloud<PointInT>::Ptr> original_clouds;
  std::vector<pcl::PointCloud<PointInT>::Ptr> clouds;
  std::vector<Eigen::Matrix4f> poses;
  std::vector<pcl::PointCloud<IndexPoint> > object_indices_clouds;

  for(size_t i=0; i < to_process.size(); i++)
  {
      std::cout << to_process[i] << std::endl;

      std::stringstream view_file;
      view_file << directory << "/" << to_process[i];
      pcl::PointCloud<PointInT>::Ptr cloud (new pcl::PointCloud<PointInT> ());
      pcl::io::loadPCDFile (view_file.str (), *cloud);

      original_clouds.push_back(cloud);

      std::cout << view_file.str() << std::endl;

      std::string file_replaced1 (view_file.str());
      boost::replace_last (file_replaced1, "cloud", pose_str);
      boost::replace_last (file_replaced1, ".pcd", ".txt");

      std::cout << file_replaced1 << std::endl;

      //read pose as well
      Eigen::Matrix4f pose;
      faat_pcl::utils::readMatrixFromFile (file_replaced1, pose);

      //the recognizer assumes transformation from M to CC - i think!
      Eigen::Matrix4f pose_inv = pose; //.inverse();

      std::string file_replaced2 (view_file.str());
      boost::replace_last (file_replaced2, "cloud", "object_indices");

      std::cout << file_replaced2 << std::endl;

      pcl::PointCloud<IndexPoint> obj_indices_cloud;
      pcl::io::loadPCDFile (file_replaced2, obj_indices_cloud);

      object_indices_clouds.push_back(obj_indices_cloud);

      pcl::PointIndices indices;
      indices.indices.resize(obj_indices_cloud.points.size());
      for(size_t kk=0; kk < obj_indices_cloud.points.size(); kk++)
        indices.indices[kk] = obj_indices_cloud.points[kk].idx;

      pcl::PointCloud<PointInT>::Ptr cloud_segmented (new pcl::PointCloud<PointInT> ());
      pcl::copyPointCloud(*cloud, indices, *cloud_segmented);

      pcl::PointCloud<PointInT>::Ptr cloud_voxel (new pcl::PointCloud<PointInT> ());

      pcl::VoxelGrid<PointInT> filter;
      filter.setInputCloud(cloud_segmented);
      filter.setDownsampleAllData(true);
      filter.setLeafSize(vx_size,vx_size,vx_size);
      filter.filter(*cloud_voxel);
      clouds.push_back(cloud_voxel);
      poses.push_back(pose_inv);

  }

  pcl::PointCloud<PointInT>::Ptr big_cloud(new pcl::PointCloud<PointInT>);

  for(size_t i=0; i < to_process.size(); i++)
  {
    pcl::PointCloud<PointInT>::Ptr trans (new pcl::PointCloud<PointInT> ());
    pcl::transformPointCloud(*clouds[i], *trans, poses[i]);
    *big_cloud += *trans;
  }

  pcl::visualization::PCLVisualizer vis("test");
  pcl::visualization::PointCloudColorHandlerRGBField<PointInT> handler(big_cloud);
  vis.addPointCloud<PointInT>(big_cloud, handler);
  vis.addCoordinateSystem(0.2f);
  vis.spin();

  std::cout << "DIFF TYPE:" << diff_type << std::endl;

  v4r::Registration::MvLMIcp<PointInT> nl_icp;
  nl_icp.setInputClouds(clouds);
  nl_icp.setPoses(poses);
  nl_icp.setMaxCorrespondenceDistance(max_dist);
  nl_icp.setMaxIterations(max_iterations);
  nl_icp.setDiffType(diff_type);
  nl_icp.compute();

  std::vector<Eigen::Matrix4f> final_poses = nl_icp.getFinalPoses();

  {
      pcl::PointCloud<PointInT>::Ptr big_cloud_after(new pcl::PointCloud<PointInT>);

      for(size_t i=0; i < to_process.size(); i++)
      {
        pcl::PointCloud<PointInT>::Ptr trans (new pcl::PointCloud<PointInT> ());
        pcl::transformPointCloud(*clouds[i], *trans, final_poses[i]);
        *big_cloud_after += *trans;
      }

      pcl::visualization::PCLVisualizer vis("test");
      int v1,v2;
      vis.createViewPort(0,0,0.5,1,v1);
      vis.createViewPort(0.5,0,1,1,v2);
      pcl::visualization::PointCloudColorHandlerRGBField<PointInT> handler_after(big_cloud_after);
      vis.addPointCloud<PointInT>(big_cloud_after, handler_after, "after", v2);
      vis.addPointCloud<PointInT>(big_cloud, handler, "before", v1);
      vis.spin();
  }

  if(output_directory.compare("") != 0)
  {

      bf::path dir = output_directory;
      if(!bf::exists(dir))
      {
          bf::create_directory(dir);
      }

      //save the data with new poses
      for(size_t i=0; i < to_process.size(); i++)
      {
          std::stringstream view_file;
          view_file << output_directory << "/" << to_process[i];
          pcl::io::savePCDFileBinary (view_file.str (), *(original_clouds[i]));
          std::cout << view_file.str() << std::endl;

          std::string file_replaced1 (view_file.str());
          boost::replace_last (file_replaced1, "cloud", pose_str);
          boost::replace_last (file_replaced1, ".pcd", ".txt");

          std::cout << file_replaced1 << std::endl;

          //read pose as well
          faat_pcl::utils::writeMatrixToFile(file_replaced1, final_poses[i]);

          std::string file_replaced2 (view_file.str());
          boost::replace_last (file_replaced2, "cloud", "object_indices");

          std::cout << file_replaced2 << std::endl;

          pcl::io::savePCDFileBinary (file_replaced2, object_indices_clouds[i]);

      }
  }
}
