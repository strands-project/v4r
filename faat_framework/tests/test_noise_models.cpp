/*
 * test_noise_models.cpp
 *
 *  Created on: Oct 28, 2013
 *      Author: aitor
 */

#include <faat_pcl/utils/noise_models.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d_omp.h>
#include <faat_pcl/utils/pcl_opencv.h>
#include <faat_pcl/utils/depth_inpainting.h>

namespace bf = boost::filesystem;

//./bin/GO3D -input_dir /home/aitor/aldoma_employee_svn/code/thomas/code/T_16_GO3D/

int
main (int argc, char ** argv)
{
  std::string input_dir_;
  bool organized_normals = true;
  float w_t = 0.5f;
  bool depth_edges = true;
  float max_angle = 70.f;
  float lateral_sigma = 0.002f;
  int interpolate_depth = 0;
  float max_pixels_interpolated = 5.f;
  pcl::console::parse_argument (argc, argv, "-interpolate_depth", interpolate_depth);
  pcl::console::parse_argument (argc, argv, "-input_dir", input_dir_);
  pcl::console::parse_argument (argc, argv, "-organized_normals", organized_normals);
  pcl::console::parse_argument (argc, argv, "-w_t", w_t);
  pcl::console::parse_argument (argc, argv, "-depth_edges", depth_edges);
  pcl::console::parse_argument (argc, argv, "-max_angle", max_angle);
  pcl::console::parse_argument (argc, argv, "-lateral_sigma", lateral_sigma);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::io::loadPCDFile(input_dir_, *cloud);

  if(interpolate_depth > 0)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr inpainted_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    switch(interpolate_depth)
    {
      case 1:
        {
          //depth inpainting
          faat_pcl::utils::depth_inpainting::NNDepthInpainter<pcl::PointXYZRGB> din;
          din.setInputCloud(cloud);
          din.setCameraParameters(320, 240, 525.f, 525.f);
          din.setMaxDistInPixels(max_pixels_interpolated);
          din.inpaint(inpainted_cloud);
        }
        break;
      case 2:
        {
          faat_pcl::utils::depth_inpainting::RMFDepthInpainter<pcl::PointXYZRGB> din;
          din.setInputCloud(cloud);
          din.setCameraParameters(320, 240, 525.f, 525.f);
          din.setMaxDistInPixels(max_pixels_interpolated);
          din.inpaint(inpainted_cloud);
        }
        break;
    }

    {
      pcl::visualization::PCLVisualizer vis ("origin vs interpolated");
      int v1, v2;
      vis.createViewPort (0, 0, 0.5, 1, v1);
      vis.createViewPort (0.5, 0, 1, 1, v2);

      {
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (cloud);
        vis.addPointCloud (cloud, handler, "big", v1);
      }

      {
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (inpainted_cloud);
        vis.addPointCloud (inpainted_cloud, handler, "big_interp", v2);
      }

      vis.spin ();
    }
    cloud = inpainted_cloud;
  }

  pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
  if(organized_normals)
  {
    std::cout << "Organized normals" << std::endl;
    pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
    ne.setMaxDepthChangeFactor (0.02f);
    ne.setNormalSmoothingSize (20.0f);
    ne.setBorderPolicy (pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal>::BORDER_POLICY_MIRROR);
    ne.setInputCloud (cloud);
    ne.compute (*normal_cloud);
  }
  else
  {
    std::cout << "Not organized normals" << std::endl;
    pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setInputCloud (cloud);
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
    ne.setSearchMethod (tree);
    ne.setRadiusSearch (0.02);
    ne.compute (*normal_cloud);
  }

  faat_pcl::utils::noise_models::NguyenNoiseModel<pcl::PointXYZRGB> nm;
  nm.setInputCloud(cloud);
  nm.setInputNormals(normal_cloud);
  nm.setLateralSigma(lateral_sigma);
  nm.setMaxAngle(max_angle);
  nm.setUseDepthEdges(depth_edges);
  nm.compute();
  std::vector<float> weights;
  nm.getWeights(weights);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered;
  nm.getFilteredCloud(filtered, w_t);

  pcl::PointCloud<pcl::PointXYZ>::Ptr edges;
  nm.getDiscontinuityEdges(edges);

  pcl::visualization::PCLVisualizer vis ("registered cloud");
  int v1, v2, v3;
  vis.createViewPort (0, 0, 0.33, 1, v1);
  vis.createViewPort (0.33, 0, 0.66, 1, v2);
  vis.createViewPort (0.66, 0, 1, 1, v3);

  {
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (cloud);
    vis.addPointCloud (cloud, handler, "big", v1);
  }

  {
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler (edges, 255, 0, 0);
    vis.addPointCloud (edges, handler, "edges", v2);
  }

  {
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (filtered);
    vis.addPointCloud (filtered, handler, "filtered", v3);
  }

  vis.addCoordinateSystem(0.2);
  vis.spin();

  {
    cv::Mat_ < cv::Vec3b > colorImage;
    PCLOpenCV::ConvertPCLCloud2Image<pcl::PointXYZRGB> (filtered, colorImage);
    cv::namedWindow("test");
    cv::imshow("test", colorImage);
  }

  {
    cv::Mat_ < cv::Vec3b > colorImage;
    PCLOpenCV::ConvertPCLCloud2Image<pcl::PointXYZRGB> (cloud, colorImage);
    cv::namedWindow("original");
    cv::imshow("original", colorImage);
  }
  cv::waitKey(0);
}
