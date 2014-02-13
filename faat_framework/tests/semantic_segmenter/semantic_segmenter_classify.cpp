/*
 * labeling_from_orgt.cpp
 *
 *  Created on: Oct 15, 2013
 *      Author: aitor
 */

#include <faat_pcl/3d_rec_framework/tools/or_evaluator.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/image/sift_local_estimator.h>
#include <faat_pcl/3d_rec_framework/pc_source/partial_pcd_source.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/fpfh_local_estimator_omp.h>
#include <faat_pcl/utils/filesystem_utils.h>
#include <pcl/console/parse.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/integral_image_normal.h>
#include <v4r/RandomForest/forest.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include "mrf.h"

//./bin/classify_semantic_segmenter -input_dir /home/aitor/data/willow_dataset/T_01_willow_dataset/cloud_0000000001.pcd -z_dist 2 -labels_dir /home/aitor/data/willow_dataset_labels/T_01_willow_dataset/cloud_0000000001.pcd

namespace bf = boost::filesystem;

int
main (int argc, char ** argv)
{
  float Z_DIST_ = 2.5f;
  std::string labels_dir_, classification_data_output_dir_, input_dir_;

  pcl::console::parse_argument (argc, argv, "-labels_dir", labels_dir_);
  pcl::console::parse_argument (argc, argv, "-input_dir", input_dir_);
  pcl::console::parse_argument (argc, argv, "-classification_data_output_dir", classification_data_output_dir_);
  pcl::console::parse_argument (argc, argv, "-z_dist", Z_DIST_);

  int UNKNOWN_LABEL_ = 0;
  int PLANE_LABEL_ = 1;
  int KNOWN_LABEL_ = 2;

  std::vector<int> labels;
  labels.push_back(0);   // wall
  labels.push_back(1);   // floor
  labels.push_back(2);    // cabinet

  pcl::visualization::PCLVisualizer vis ("Labeling results");
  int v1,v2,v3,v4;
  vis.createViewPort (0.0, 0.0, 0.5, 0.5, v1);
  vis.createViewPort (0.0, 0.5, 0.5, 1, v2);
  vis.createViewPort (0.5, 0, 1, 0.5, v3);
  vis.createViewPort (0.5, 0.5, 1, 1, v4);

  boost::shared_ptr < faat_pcl::rec_3d_framework::SIFTLocalEstimation<pcl::PointXYZRGB, pcl::Histogram<128> > > estimator;
  estimator.reset (new faat_pcl::rec_3d_framework::SIFTLocalEstimation<pcl::PointXYZRGB, pcl::Histogram<128> >);

  boost::shared_ptr<faat_pcl::rec_3d_framework::FPFHLocalEstimationOMP<pcl::PointXYZRGB, pcl::FPFHSignature33 > > fpfh_estimator;
  fpfh_estimator.reset (new faat_pcl::rec_3d_framework::FPFHLocalEstimationOMP<pcl::PointXYZRGB, pcl::FPFHSignature33 >);

  //compute sift keypoints
  std::cout << "recognizing " << input_dir_ << std::endl;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::io::loadPCDFile (input_dir_, *scene);

  pcl::PointCloud<pcl::PointXYZL>::Ptr gt_labels (new pcl::PointCloud<pcl::PointXYZL>);
  pcl::io::loadPCDFile (labels_dir_, *gt_labels);

  /*pcl::PassThrough<pcl::PointXYZRGB> pass_;
  pass_.setFilterLimits (0.f, Z_DIST_);
  pass_.setFilterFieldName ("z");
  pass_.setKeepOrganized (true);
  pass_.setInputCloud (scene);
  pass_.filter (*scene);*/

  {
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZL> scene_handler (gt_labels, "label");
    vis.addPointCloud<pcl::PointXYZL> (gt_labels, scene_handler, "gt_labels", v2);
  }

  {
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> scene_handler (scene);
    vis.addPointCloud<pcl::PointXYZRGB> (scene, scene_handler, "scene_cloud_z_coloured", v1);
  }

  //dense sift
  boost::shared_ptr<faat_pcl::rec_3d_framework::UniformSamplingExtractor<pcl::PointXYZRGB> >
    uniform_keypoint_extractor ( new faat_pcl::rec_3d_framework::UniformSamplingExtractor<pcl::PointXYZRGB>);

  uniform_keypoint_extractor->setSamplingDensity (0.02f);
  uniform_keypoint_extractor->setFilterPlanar (false);
  uniform_keypoint_extractor->setInputCloud(scene);
  std::vector<int> all_indices;
  uniform_keypoint_extractor->compute(all_indices);

  std::vector<int> feature_indices;
  for(size_t i=0; i < all_indices.size(); i++)
  {
    if(scene->points[all_indices[i]].z > Z_DIST_)
      continue;

    feature_indices.push_back(all_indices[i]);
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::copyPointCloud(*scene, feature_indices, *keypoints);

  pcl::PointCloud<pcl::Histogram<128> >::Ptr signatures (new pcl::PointCloud<pcl::Histogram<128> >);
  estimator->setIndices(feature_indices);
  estimator->estimate (scene, signatures);

  pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
  ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
  ne.setMaxDepthChangeFactor (0.02f);
  ne.setNormalSmoothingSize (20.0f);
  ne.setBorderPolicy (pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal>::BORDER_POLICY_IGNORE);
  ne.setInputCloud (scene);
  pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
  ne.compute (*normal_cloud);

  pcl::PointCloud<pcl::FPFHSignature33 >::Ptr fpfh_signatures (new pcl::PointCloud<pcl::FPFHSignature33 >);
  fpfh_estimator->setSupportRadius (0.04f);
  fpfh_estimator->estimate(scene, normal_cloud, feature_indices, fpfh_signatures);

  //normal sift
  /*pcl::PointCloud<pcl::PointXYZRGB>::Ptr processed (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::Histogram<128> >::Ptr signatures (new pcl::PointCloud<pcl::Histogram<128> >);
  estimator->estimate (scene, processed, keypoints, signatures);*/

  /*{
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> scene_handler (keypoints);
    vis.addPointCloud<pcl::PointXYZRGB> (keypoints, scene_handler, "keypoints", v2);
  }*/

  Forest rf("myforest.txt");

  //classify each of the feature points
  pcl::PointCloud<pcl::PointXYZL>::Ptr labels_cloud (new pcl::PointCloud<pcl::PointXYZL>);

  std::vector<std::vector<float> > probs;
  probs.resize(signatures->points.size());

  int total_feature_size = 128 + 3 + 33;

  for(size_t k=0; k < signatures->points.size(); k++)
  {
    std::vector<float> feature(total_feature_size,0);
    for(size_t j=0; j < 128; j++)
      feature[j] = signatures->points[k].histogram[j];

    //copy rgb
    feature[128] = scene->points[feature_indices[k]].r;
    feature[129] = scene->points[feature_indices[k]].g;
    feature[130] = scene->points[feature_indices[k]].b;

    //copy fpfh
    for(size_t kk=0; kk < 33; kk++)
    {
      feature[131+kk] = fpfh_signatures->points[k].histogram[kk];
    }

    int ID = rf.ClassifyPoint(feature);
    probs[k] = rf.SoftClassify(feature);
    std::cout << probs[k][0] << " " << probs[k][1] << " " << probs[k][2] << std::endl;
    //probs[k][0] = probs[k][1] = probs[k][2] = 1.f;
    std::cout << "id:" << ID << " " << labels[ID] << std::endl;

    pcl::PointXYZL p;
    p.label = ID;
    p.getVector3fMap() = keypoints->points[k].getVector3fMap();
    labels_cloud->points.push_back(p);
  }

  pcl::PointCloud<pcl::PointXYZL>::Ptr cc(new pcl::PointCloud<pcl::PointXYZL>);
  std::vector<std::vector<int> > ccomps;
  unsigned int knn = 8;
  cc = hombreViejo::solveMrfViaBP_kNN<pcl::PointXYZRGB>(keypoints, probs, ccomps, 5.f, 30, knn, true);

  pcl::PointCloud<pcl::PointXYZL>::Ptr labels_cloud_mrf (new pcl::PointCloud<pcl::PointXYZL>);

  for(size_t k=0; k < signatures->points.size(); k++)
  {
    //max prob
    int max=0;
    float max_f = std::numeric_limits<float>::max();
    for(int i=0; i < 3; i++)
    {
      //std::cout << probs[k][i] << std::endl;
      if(probs[k][i] < max_f)
      {
        max_f = probs[k][i];
        max = i;
      }
    }

    pcl::PointXYZL p;
    p.label = max;
    p.getVector3fMap() = keypoints->points[k].getVector3fMap();
    labels_cloud_mrf->points.push_back(p);
  }

  {
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZL> scene_handler (labels_cloud, "label");
    vis.addPointCloud<pcl::PointXYZL> (labels_cloud, scene_handler, "labels", v3);
    vis.addText("labels (no MRF)", 15, 15, "text", v3);
  }

  /*{
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZL> scene_handler (cc, "label");
    vis.addPointCloud<pcl::PointXYZL> (cc, scene_handler, "labels_mrf", v4);
  }*/

  {
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZL> scene_handler (labels_cloud_mrf, "label");
    vis.addPointCloud<pcl::PointXYZL> (labels_cloud_mrf, scene_handler, "labels_mrf", v4);
  }

  vis.spin();
  vis.removeAllPointClouds();
  vis.removeAllShapes();
}
