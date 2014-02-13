/*
 * kinect.cpp
 *
 *  Created on: Aug 6, 2013
 *      Author: aitor
 */

/*
 * test_training.cpp
 *
 *  Created on: Mar 9, 2012
 *      Author: aitor
 */

#include <pcl/pcl_macros.h>
#include <faat_pcl/3d_rec_framework/tools/openni_frame_source.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/apps/dominant_plane_segmentation.h>
#include <pcl/console/parse.h>
#include <pcl/common/time.h>

template<typename PointT>
void
segment (std::string & save_directory)
{
  //get point cloud from the kinect, segment it and classify it
  OpenNIFrameSource::OpenNIFrameSource camera;
  OpenNIFrameSource::PointCloudPtr frame;
  camera.setSaveDirectory(save_directory);

  pcl::visualization::PCLVisualizer vis ("kinect");
  int v1,v2;
  vis.createViewPort(0,0,0.5,1.0, v1);
  vis.createViewPort(0.5,0,1.0,1.0, v2);

  //keyboard callback to stop getting frames and finalize application
  boost::function<void
  (const pcl::visualization::KeyboardEvent&)> keyboard_cb = boost::bind (&OpenNIFrameSource::OpenNIFrameSource::onKeyboardEvent, &camera, _1);
  vis.registerKeyboardCallback (keyboard_cb);
  size_t previous_cluster_size = 0;
  size_t previous_categories_size = 0;

  float Z_DIST_ = 1.25f;
  float text_scale = 0.015f;

  while (camera.isActive ())
  {
    pcl::ScopeTime frame_process ("Global frame processing ------------- ");
    frame = camera.snap ();

    std::cout << frame->points.size() << std::endl;
    typename pcl::PointCloud<PointT>::Ptr xyz_points (new pcl::PointCloud<PointT>);
    pcl::copyPointCloud (*frame, *xyz_points);

    //Step 1 -> Segment
    pcl::apps::DominantPlaneSegmentation<PointT> dps;
    dps.setInputCloud (xyz_points);
    dps.setMaxZBounds (Z_DIST_);
    dps.setObjectMinHeight (0.005);
    dps.setMinClusterSize (1000);
    dps.setWSize (9);
    dps.setDistanceBetweenClusters (0.1f);

    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;
    std::vector<pcl::PointIndices> indices;
    dps.setDownsamplingSize (0.02f);
    dps.compute_fast (clusters);
    dps.getIndicesClusters (indices);
    Eigen::Vector4f table_plane_;
    Eigen::Vector3f normal_plane_ = Eigen::Vector3f (table_plane_[0], table_plane_[1], table_plane_[2]);
    dps.getTableCoefficients (table_plane_);

    vis.removePointCloud ("frame");
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> random_handler (frame);
    vis.addPointCloud<OpenNIFrameSource::PointT> (frame, random_handler, "frame", v1);

    for (size_t i = 0; i < previous_cluster_size; i++)
    {
      std::stringstream cluster_name;
      cluster_name << "cluster_" << i;
      vis.removePointCloud (cluster_name.str ());
    }

    for (size_t i = 0; i < clusters.size (); i++)
    {
      std::stringstream cluster_name;
      cluster_name << "cluster_" << i;
      pcl::visualization::PointCloudColorHandlerRandom<PointT> random_handler (clusters[i]);
      vis.addPointCloud<PointT> (clusters[i], random_handler, cluster_name.str (), v2);
    }

    previous_cluster_size = clusters.size ();

    vis.spinOnce ();
  }
}

//bin/pcl_global_classification -models_dir /home/aitor/data/3d-net_one_class/ -descriptor_name esf -training_dir /home/aitor/data/3d-net_one_class_trained_level_1 -nn 10

int
main (int argc, char ** argv)
{

  std::string path = "";
  pcl::console::parse_argument (argc, argv, "-save_dir", path);

  segment<pcl::PointXYZRGBA>(path);

}
