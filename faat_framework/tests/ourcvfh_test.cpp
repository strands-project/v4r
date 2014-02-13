/*
 * local_recognition_mian_dataset.cpp
 *
 *  Created on: Mar 24, 2012
 *      Author: aitor
 */
#include <pcl/console/parse.h>
#include <faat_pcl/3d_rec_framework/pc_source/partial_pcd_source.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/global/organized_color_ourcvfh_estimator.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/passthrough.h>
#include <faat_pcl/3d_rec_framework/defines/faat_3d_rec_framework_defines.h>

POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::Histogram<1327>,
    (float[1327], histogram, histogram1327)
)

int
main (int argc, char ** argv)
{

    std::string pcd_file = "";
    float axis_threshold = 1.f;
    bool normalize_ = false;
    std::string cluster_thresholds = "";
    std::vector<float> cluster_thresholds_float;

    pcl::console::parse_argument (argc, argv, "-pcd_file", pcd_file);
    pcl::console::parse_argument (argc, argv, "-axis_threshold", axis_threshold);
    pcl::console::parse_argument (argc, argv, "-normalize", normalize_);
    pcl::console::parse_argument (argc, argv, "-cluster_thresholds", cluster_thresholds);

    if(cluster_thresholds.compare("") == 0)
    {
        cluster_thresholds_float.push_back(1.f);
    }
    else
    {
        std::vector<std::string> strs;
        boost::split (strs, cluster_thresholds, boost::is_any_of (","));
        for(size_t i=0; i < strs.size(); i++)
        {
            cluster_thresholds_float.push_back(atof(strs[i].c_str()));
        }
    }

  typedef pcl::PointXYZRGB PointT;
  typedef pcl::Histogram<1327> FeatureT;

  if (pcd_file.compare ("") == 0)
  {
    PCL_ERROR("Set the directory containing scenes\n");
    return -1;
  }

  pcl::PointCloud<PointT>::Ptr view(new pcl::PointCloud<PointT>);
  pcl::io::loadPCDFile(pcd_file, *view);

  //configure normal estimator
  boost::shared_ptr<faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointT, pcl::Normal> > normal_estimator;
  normal_estimator.reset (new faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointT, pcl::Normal>);
  normal_estimator->setCMR (false);
  normal_estimator->setDoVoxelGrid (true);
  normal_estimator->setRemoveOutliers (true);
  normal_estimator->setValuesForCMRFalse (0.001f, 0.02f);

    /*pcl::PassThrough<PointT> pass_;
    pass_.setFilterLimits (0.f, 1.5f);
    pass_.setFilterFieldName ("z");
    pass_.setInputCloud (view);
    pass_.setKeepOrganized (true);
    pass_.filter (*view);*/

  pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
  /*pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
  ne.setRadiusSearch(0.02f);
  ne.setInputCloud (view);
  ne.compute (*normal_cloud);*/

  pcl::PointCloud<PointT>::Ptr processed_normals(new pcl::PointCloud<PointT>);
  normal_estimator->estimate_organized(view, processed_normals, normal_cloud);

  //boost::shared_ptr<faat_pcl::rec_3d_framework::ColorOURCVFHEstimator<PointT, pcl::Histogram<1327> > > vfh_estimator;
  //vfh_estimator.reset (new faat_pcl::rec_3d_framework::ColorOURCVFHEstimator<PointT, pcl::Histogram<1327> >);
  boost::shared_ptr<faat_pcl::rec_3d_framework::OrganizedColorOURCVFHEstimator<PointT, pcl::Histogram<1327> > > vfh_estimator;
  vfh_estimator.reset (new faat_pcl::rec_3d_framework::OrganizedColorOURCVFHEstimator<PointT, pcl::Histogram<1327> >);
  //vfh_estimator->setNormalEstimator (normal_estimator);
  vfh_estimator->setNormals(normal_cloud);
  vfh_estimator->setNormalizeBins (normalize_);
  vfh_estimator->setUseRFForColor (true);
  vfh_estimator->setRefineClustersParam (100.f);
  vfh_estimator->setInternalNormalRadiusAndResolution(0.02f, 0.001f);
  //vfh_estimator->setAdaptativeMLS (true);

  vfh_estimator->setAxisRatio (1.f);
  vfh_estimator->setMinAxisValue (1.f);

  {
    //segmentation parameters for training
    std::vector<float> eps_thresholds, cur_thresholds, clus_thresholds;
    eps_thresholds.push_back (0.15);

    for(size_t i=0; i < cluster_thresholds_float.size(); i++)
    {
        cur_thresholds.push_back (cluster_thresholds_float[i]);
    }

    //cur_thresholds.push_back (0.015f);
    //cur_thresholds.push_back (1.f);
    clus_thresholds.push_back (10.f);

    vfh_estimator->setClusterToleranceVector (clus_thresholds);
    vfh_estimator->setEpsAngleThresholdVector (eps_thresholds);
    vfh_estimator->setCurvatureThresholdVector (cur_thresholds);
  }

  //vfh_estimator->setCVFHParams (0.125f, 0.0175f, 2.5f);
  //vfh_estimator->setCVFHParams (0.15f, 0.0175f, 3.f); //willow_challenge_trained

  /*{
    //segmentation parameters for recognition
    std::vector<float> eps_thresholds, cur_thresholds, clus_thresholds;
    eps_thresholds.push_back (0.15);
    cur_thresholds.push_back (0.015f);
    //cur_thresholds.push_back (0.02f);
    //cur_thresholds.push_back (0.035f);
    clus_thresholds.push_back (2.5f);

    vfh_estimator->setClusterToleranceVector (clus_thresholds);
    vfh_estimator->setEpsAngleThresholdVector (eps_thresholds);
    vfh_estimator->setCurvatureThresholdVector (cur_thresholds);

    vfh_estimator->setAdaptativeMLS (true);
  }*/

  vfh_estimator->setAxisRatio (axis_threshold);
  vfh_estimator->setMinAxisValue (axis_threshold);

  pcl::PointCloud<PointT>::Ptr processed(new pcl::PointCloud<PointT>);
  std::vector<pcl::PointCloud<FeatureT>, Eigen::aligned_allocator<pcl::PointCloud<FeatureT> > > signatures;
  std::vector < Eigen::Vector3f > centroids;
  vfh_estimator->estimate (processed_normals, processed, signatures, centroids);

  std::vector<bool> valid_trans;
  std::vector < Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms;

  vfh_estimator->getValidTransformsVec (valid_trans);
  vfh_estimator->getTransformsVec (transforms);
  std::vector<pcl::PointIndices> cluster_indices;
  vfh_estimator->getClusterIndices(cluster_indices);
  pcl::visualization::PCLVisualizer vis("ourcvfh_test");
  //pcl::visualization::PointCloudColorHandlerRGBField<PointT> handler(processed);
  //vis.addPointCloud<PointT>(processed, handler, "processed_cloud");
  vis.addCoordinateSystem(0.1f);

  std::cout << cluster_indices.size() << " " << signatures.size() << std::endl;

  for(size_t i=0; i < transforms.size(); i++)
  {
      if(valid_trans[i])
      {
          std::cout << cluster_indices[i].indices.size() << std::endl;
          pcl::PointCloud<PointT>::Ptr transformed(new pcl::PointCloud<PointT>);
          pcl::PointCloud<PointT>::Ptr region_cloud(new pcl::PointCloud<PointT>);
          /*Eigen::Vector4f centroid;
          centroid[0] = centroids[i][0];
          centroid[1] = centroids[i][1];
          centroid[2] = centroids[i][2];
          pcl::demeanPointCloud(*processed, centroid, *transformed);*/
          Eigen::Matrix4f trans = transforms[i]; //.inverse();
          pcl::transformPointCloud(*processed_normals, *transformed, trans);
          pcl::copyPointCloud(*transformed, cluster_indices[i].indices, *region_cloud);

          {
            pcl::visualization::PointCloudColorHandlerRGBField<PointT> handler(transformed);
            vis.addPointCloud<PointT>(transformed, handler, "transformed");
          }

          {
            pcl::visualization::PointCloudColorHandlerCustom<PointT> handler(region_cloud, 0, 255, 0);
            vis.addPointCloud<PointT>(region_cloud, handler, "region");
          }

          vis.spin();
          vis.removePointCloud("transformed");
          vis.removePointCloud("region");

          pcl::io::savePCDFileBinary("histogram.pcd", signatures[i]);
          /*float sum = 0.f;
          for(size_t k=0; k < 1327; k++)
          {
              if(k == 45)
              {
                  std::cout << std::endl;
                  std::cout << sum << std::endl;
              }

              if(k == 90)
              {
                  std::cout << std::endl;
                  std::cout << sum << std::endl;
              }

              if(k == (90+45))
              {
                  std::cout << std::endl;
                  std::cout << sum << std::endl;
              }

              if(k == (90+45+13))
              {
                  std::cout << std::endl;
                  std::cout << sum << std::endl;
              }

              if(k == (90+45+13*2))
              {
                  std::cout << std::endl;
                  std::cout << sum << std::endl;
              }

              if(k == (90+45+13*3))
              {
                  std::cout << std::endl;
                  std::cout << sum << std::endl;
              }

              if(k == (90+45+13*4))
              {
                  std::cout << std::endl;
                  std::cout << sum << std::endl;
              }

              if(k == (90+45+13*5))
              {
                  std::cout << std::endl;
                  std::cout << sum << std::endl;
              }

              if(k == (90+45+13*6))
              {
                  std::cout << std::endl;
                  std::cout << sum << std::endl;
              }

              if(k == (90+45+13*7))
              {
                  std::cout << std::endl;
                  std::cout << sum << std::endl;
              }

              if(k == (90+45+13*8))
              {
                  std::cout << std::endl;
                  std::cout << sum << std::endl;
              }

              if(k == 303)
              {
                  std::cout << std::endl;
                  std::cout << sum << std::endl;
              }

              if(k == 303 + 128)
              {
                  std::cout << std::endl;
                  std::cout << sum << std::endl;
              }

              if(k == 303 + 128 * 2)
              {
                  std::cout << std::endl;
                  std::cout << sum << std::endl;
              }

              if(k == 303 + 128 * 3)
              {
                  std::cout << std::endl;
                  std::cout << sum << std::endl;
              }

              if(k == 303 + 128 * 4)
              {
                  std::cout << std::endl;
                  std::cout << sum << std::endl;
              }

              if(k == 303 + 128 * 5)
              {
                  std::cout << std::endl;
                  std::cout << sum << std::endl;
              }

              if(k == 303 + 128 * 6)
              {
                  std::cout << std::endl;
                  std::cout << sum << std::endl;
              }

              if(k == 303 + 128 * 7)
              {
                  std::cout << std::endl;
                  std::cout << sum << std::endl;
              }

              std::cout << signatures[i].points[0].histogram[k] << " ";
              sum += signatures[i].points[0].histogram[k];
          }

          std::cout << std::endl;
          std::cout << sum << std::endl;*/
      }
  }
}
