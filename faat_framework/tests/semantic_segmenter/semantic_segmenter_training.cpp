/*
 * labeling_from_orgt.cpp
 *
 *  Created on: Oct 15, 2013
 *      Author: aitor
 */

#include <faat_pcl/3d_rec_framework/tools/or_evaluator.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/image/sift_local_estimator.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/fpfh_local_estimator_omp.h>
#include <faat_pcl/3d_rec_framework/pc_source/partial_pcd_source.h>
#include <faat_pcl/utils/filesystem_utils.h>
#include <pcl/console/parse.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/integral_image_normal.h>
#include <v4r/RandomForest/forest.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <fstream>

namespace bf = boost::filesystem;

// ./bin/train_semantic_segmenter -labels_dir /home/aitor/data/willow_dataset_labels/T_01_willow_dataset/ -input_dir /home/aitor/data/willow_dataset/T_01_willow_dataset/ -z_dist 2 -classification_data_output_dir /home/aitor/data/forest_data

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

  std::vector<boost::shared_ptr<std::ofstream> > categories_files;
  categories_files.resize(3);

  if(classification_data_output_dir_.compare("") != 0)
  {
    bf::path p = classification_data_output_dir_;
    if(!bf::exists(p))
      bf::create_directory(p);

    //write categories file
    std::stringstream file_str;
    file_str << classification_data_output_dir_ << "/categories.txt";
    std::ofstream out (file_str.str().c_str ());
    out << UNKNOWN_LABEL_ << "\tunknown" << std::endl;
    out << PLANE_LABEL_ << "\tplane" << std::endl;
    out << KNOWN_LABEL_ << "\tknown" << std::endl;
    out.close();

    for(size_t i=0; i < 3; i++)
    {
      std::stringstream file_str;
      file_str << classification_data_output_dir_ << "/" << std::setw(4) << std::setfill('0') << i << ".data";
      std::cout << file_str.str() << std::endl;
      categories_files[i].reset(new std::ofstream(file_str.str().c_str()));
    }
  }

  //take all scenes in labels_dir_

  bf::path input = labels_dir_;
  std::vector<std::string> files_to_label;
  std::vector<std::string> orig_scenes;

  if (bf::is_directory (labels_dir_))
  {
    std::vector<std::string> files;
    std::string start = "";
    std::string ext = std::string ("pcd");
    bf::path dir = input;
    faat_pcl::utils::getFilesInDirectory (dir, start, files, ext);
    std::cout << "Number of scenes in directory is:" << files.size () << std::endl;
    for (size_t i = 0; i < files.size (); i++)
    {
      typename pcl::PointCloud<pcl::PointXYZL>::Ptr scene_cloud (new pcl::PointCloud<pcl::PointXYZL>);
      std::cout << files[i] << std::endl;

      {
        std::stringstream filestr;
        filestr << labels_dir_ << files[i];
        std::string file = filestr.str ();
        files_to_label.push_back (file);
      }

      {
        std::stringstream filestr;
        filestr << input_dir_ << files[i];
        std::string file = filestr.str ();
        orig_scenes.push_back (file);
      }
    }

    std::sort (files_to_label.begin (), files_to_label.end ());
    std::sort (orig_scenes.begin (), orig_scenes.end ());
  }
  else
  {
    PCL_ERROR("Expecting a directory as input!!, aborting\n");
    exit (-1);
  }

  assert(files_to_label.size() == orig_scenes.size());

  pcl::visualization::PCLVisualizer vis ("Labeling results");
  int v1,v2,v3;
  vis.createViewPort (0.0, 0.0, 0.33, 1, v1);
  vis.createViewPort (0.33, 0, 0.66, 1, v2);
  vis.createViewPort (0.66, 0, 1, 1, v3);

  std::cout << files_to_label.size () << std::endl;

  boost::shared_ptr < faat_pcl::rec_3d_framework::SIFTLocalEstimation<pcl::PointXYZRGB, pcl::Histogram<128> > > estimator;
  estimator.reset (new faat_pcl::rec_3d_framework::SIFTLocalEstimation<pcl::PointXYZRGB, pcl::Histogram<128> >);

  boost::shared_ptr<faat_pcl::rec_3d_framework::FPFHLocalEstimationOMP<pcl::PointXYZRGB, pcl::FPFHSignature33 > > fpfh_estimator;
  fpfh_estimator.reset (new faat_pcl::rec_3d_framework::FPFHLocalEstimationOMP<pcl::PointXYZRGB, pcl::FPFHSignature33 >);

  for (size_t i = 0; i < files_to_label.size (); i++)
  {

    //compute sift keypoints
    std::cout << "recognizing " << files_to_label[i] << std::endl;
    pcl::PointCloud<pcl::PointXYZL>::Ptr labels (new pcl::PointCloud<pcl::PointXYZL>);
    pcl::io::loadPCDFile (files_to_label[i], *labels);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile (orig_scenes[i], *scene);

    /*pcl::PassThrough<pcl::PointXYZRGB> pass_;
    pass_.setFilterLimits (0.f, Z_DIST_);
    pass_.setFilterFieldName ("z");
    pass_.setKeepOrganized (true);
    pass_.setInputCloud (scene);
    pass_.filter (*scene);*/

    {
      pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZL> scene_handler (labels, "label");
      vis.addPointCloud<pcl::PointXYZL> (labels, scene_handler, "labels", v3);
    }

    {
      pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> scene_handler (scene);
      vis.addPointCloud<pcl::PointXYZRGB> (scene, scene_handler, "scene_cloud_z_coloured", v1);
    }

    int total_feature_size = 128 + 3 + 33;
    std::vector< std::vector< std::vector<float> > > features_by_label;
    features_by_label.resize(3);

    //dense sift
    boost::shared_ptr<faat_pcl::rec_3d_framework::UniformSamplingExtractor<pcl::PointXYZRGB> >
      uniform_keypoint_extractor ( new faat_pcl::rec_3d_framework::UniformSamplingExtractor<pcl::PointXYZRGB>);

    uniform_keypoint_extractor->setSamplingDensity (0.02f);
    uniform_keypoint_extractor->setFilterPlanar (false);
    uniform_keypoint_extractor->setInputCloud(scene);
    std::vector<int> feature_indices;
    uniform_keypoint_extractor->compute(feature_indices);

    pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
    ne.setMaxDepthChangeFactor (0.02f);
    ne.setNormalSmoothingSize (20.0f);
    ne.setBorderPolicy (pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal>::BORDER_POLICY_IGNORE);
    ne.setInputCloud (scene);
    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
    ne.compute (*normal_cloud);

    pcl::PointCloud<pcl::Histogram<128> >::Ptr signatures (new pcl::PointCloud<pcl::Histogram<128> >);
    estimator->setIndices(feature_indices);
    estimator->estimate (scene, signatures);

    pcl::PointCloud<pcl::FPFHSignature33 >::Ptr fpfh_signatures (new pcl::PointCloud<pcl::FPFHSignature33 >);
    fpfh_estimator->setSupportRadius (0.04f);
    fpfh_estimator->estimate(scene, normal_cloud, feature_indices, fpfh_signatures);

    std::vector<int> final_indices;
    for(size_t k=0; k < feature_indices.size(); k++)
    {
      if(!pcl_isfinite(labels->points[feature_indices[k]].z))
        continue;

      final_indices.push_back(feature_indices[k]);

      std::vector<float> feat_vector;
      feat_vector.resize(total_feature_size);

      //copy sift
      for(size_t kk=0; kk < 128; kk++)
      {
        feat_vector[kk] = signatures->points[k].histogram[kk];
      }

      //copy rgb
      feat_vector[128] = scene->points[feature_indices[k]].r;
      feat_vector[129] = scene->points[feature_indices[k]].g;
      feat_vector[130] = scene->points[feature_indices[k]].b;

      //copy fpfh
      for(size_t kk=0; kk < 33; kk++)
      {
        feat_vector[131+kk] = fpfh_signatures->points[k].histogram[kk];
      }

      //add feature to the correct label
      features_by_label[labels->points[feature_indices[k]].label].push_back(feat_vector);
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*scene, final_indices, *keypoints);
    {
      pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> scene_handler (keypoints);
      vis.addPointCloud<pcl::PointXYZRGB> (keypoints, scene_handler, "keypoints", v2);
    }

    //normal sift
    /*pcl::PointCloud<pcl::PointXYZRGB>::Ptr processed (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::Histogram<128> >::Ptr signatures (new pcl::PointCloud<pcl::Histogram<128> >);
    estimator->estimate (scene, processed, keypoints, signatures);
    pcl::PointIndices sift_keypoints;
    estimator->getKeypointIndices(sift_keypoints);

    {
      pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> scene_handler (keypoints);
      vis.addPointCloud<pcl::PointXYZRGB> (keypoints, scene_handler, "keypoints", v2);
    }

    //take keypoint position and get label, group features by label
    std::cout << sift_keypoints.indices.size() << std::endl;


    assert(sift_keypoints.indices.size() == signatures->points.size());
    for(size_t k=0; k < sift_keypoints.indices.size(); k++)
    {
      //std::cout << sift_keypoints.indices[k] << std::endl;

      if(sift_keypoints.indices[k] >= labels->points.size())
        continue;

      if(!pcl_isfinite(scene->points[sift_keypoints.indices[k]].z))
        continue;

      if(!pcl_isfinite(labels->points[sift_keypoints.indices[k]].z))
        continue;

      //std::cout << labels->points[sift_keypoints.indices[k]].label << std::endl;
      features_by_label[labels->points[sift_keypoints.indices[k]].label].push_back(signatures->points[k]);
    }*/

    for(size_t k=0; k < features_by_label.size(); k++)
    {
      std::cout << "size:" << features_by_label[k].size() << std::endl;
      for(size_t kk=0; kk < features_by_label[k].size(); kk++)
      {
        for(size_t j=0; j < total_feature_size; j++)
        {
          *categories_files[k] << std::setw(12) << std::setfill(' ') << features_by_label[k][kk][j];
          if(j < (total_feature_size - 1))
          {
            *categories_files[k] << " ";
          }
        }

        *categories_files[k] << std::endl;
      }
    }

    vis.spin();
    vis.removeAllPointClouds();
    vis.removeAllShapes();
  }

  for(size_t i=0; i < 3; i++)
    categories_files[i]->close();

  std::vector<int> labels;
  labels.push_back(0);   // wall
  labels.push_back(1);   // floor
  labels.push_back(2);    // cabinet

  // load training data from files
  ClassificationData trainingData;
  trainingData.LoadFromDirectory(classification_data_output_dir_, labels);

  // define Random Forest
  //   parameters:
  //   int nTrees
  //   int maxDepth
  //   float baggingRatio
  //   int nFeaturesToTryAtEachNode
  //   float minInformationGain
  //   int nMinNumberOfPointsToSplit
  Forest rf(10, 20 , 0.1, 200, 0.02, 5);

  // train forest
  //   parameters:
  //   ClassificationData data
  //   bool refineWithAllDataAfterwards
  //   int verbosityLevel (0 - quiet, 3 - most detail)
  rf.TrainLarge(trainingData, false, 3);

  // save after training
  rf.SaveToFile("myforest.txt");


}
