/*
 * local_recognition_mian_dataset.cpp
 *
 *  Created on: Mar 24, 2012
 *      Author: aitor
 */

#include <pcl/common/common.h>
#include <pcl/console/parse.h>
#include <faat_pcl/3d_rec_framework/pc_source/registered_views_source.h>
#include <faat_pcl/3d_rec_framework/pipeline/local_recognizer.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/image/sift_local_estimator.h>
#include <faat_pcl/recognition/cg/graph_geometric_consistency.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
typedef faat_pcl::rec_3d_framework::Model<pcl::PointXYZRGB> ModelT;
typedef boost::shared_ptr<ModelT> ModelTPtr;

//./bin/est_reg_views_source -models_dir /home/aitor/data/willow_object_clouds/models_ml_reduced/ -training_dir /home/aitor/test_reg_views_source/ -training_input_structure /home/aitor/data/willow_structure/ -pcd_file /home/aitor/data/willow_test/T_03/cloud_0000000002.pcd -use_cache 0

int
main (int argc, char ** argv)
{
  std::string path = "";
  std::string training_dir = "trained_models/";
  std::string training_input_structure = "";
  std::string pcd_file = "";
  bool use_cache = true;
  float hyp_thres = 0.2f;

  pcl::console::parse_argument (argc, argv, "-use_cache", use_cache);
  pcl::console::parse_argument (argc, argv, "-hyp_thres", hyp_thres);
  pcl::console::parse_argument (argc, argv, "-models_dir", path);
  pcl::console::parse_argument (argc, argv, "-pcd_file", pcd_file);
  pcl::console::parse_argument (argc, argv, "-training_dir", training_dir);
  pcl::console::parse_argument (argc, argv, "-training_input_structure", training_input_structure);
  std::string desc_name = "sift";
  std::string idx_flann_fn = "sift_flann.idx";
  int CG_SIZE_ = 5;
  float CG_THRESHOLD_ = 0.01f;

  //configure mesh source
  typedef pcl::PointXYZRGB PointT;
  boost::shared_ptr<faat_pcl::rec_3d_framework::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT> >
                                                                                                                                    mesh_source (
                                                                                                                                                 new faat_pcl::rec_3d_framework::RegisteredViewsSource<
                                                                                                                                                     pcl::PointXYZRGBNormal,
                                                                                                                                                     pcl::PointXYZRGB,
                                                                                                                                                     pcl::PointXYZRGB>);
  mesh_source->setPath (path);
  mesh_source->setModelStructureDir (training_input_structure);
  mesh_source->generate (training_dir);

  boost::shared_ptr<faat_pcl::rec_3d_framework::Source<PointT> > cast_source;
  cast_source = boost::static_pointer_cast<faat_pcl::rec_3d_framework::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT> > (mesh_source);

  boost::shared_ptr<pcl::CorrespondenceGrouping<PointT, PointT> > cast_cg_alg;
  {
    /*boost::shared_ptr<faat_pcl::GraphGeometricConsistencyGrouping<PointT, PointT> > gcg_alg (
                                                                                                 new faat_pcl::GraphGeometricConsistencyGrouping<PointT, PointT>);
    gcg_alg->setGCThreshold (CG_SIZE_);
    gcg_alg->setGCSize (CG_THRESHOLD_);
    gcg_alg->setRansacThreshold (CG_THRESHOLD_);
    gcg_alg->setUseGraph(true);
    gcg_alg->setDistForClusterFactor(2);
    gcg_alg->setDotDistance(1.f);*/

    boost::shared_ptr<pcl::GeometricConsistencyGrouping<PointT, PointT> > gcg_alg (
                                                                                                     new pcl::GeometricConsistencyGrouping<PointT, PointT>);
    gcg_alg->setGCThreshold (CG_SIZE_);
    gcg_alg->setGCSize (CG_THRESHOLD_);
    cast_cg_alg = boost::static_pointer_cast<pcl::CorrespondenceGrouping<PointT, PointT> > (gcg_alg);
  }

  boost::shared_ptr<faat_pcl::rec_3d_framework::SIFTLocalEstimation<PointT, pcl::Histogram<128> > > estimator;
  estimator.reset (new faat_pcl::rec_3d_framework::SIFTLocalEstimation<PointT, pcl::Histogram<128> >);

  boost::shared_ptr<faat_pcl::rec_3d_framework::LocalEstimator<PointT, pcl::Histogram<128> > > cast_estimator;
  cast_estimator = boost::dynamic_pointer_cast<faat_pcl::rec_3d_framework::SIFTLocalEstimation<PointT, pcl::Histogram<128> > > (estimator);

  boost::shared_ptr<faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<128> > > local;
  local.reset (new faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<128> > (idx_flann_fn));
  local->setDataSource (cast_source);
  local->setTrainingDir (training_dir);
  local->setDescriptorName (desc_name);
  local->setFeatureEstimator (cast_estimator);
  local->setCGAlgorithm (cast_cg_alg);
  local->setICPIterations(0);
  local->setThresholdAcceptHyp(hyp_thres);
  local->setUseCache (use_cache);
  local->initialize (false);

  pcl::PointCloud<PointT>::Ptr scene(new pcl::PointCloud<PointT>);
  pcl::io::loadPCDFile(pcd_file, *scene);

  local->setInputCloud(scene);
  local->recognize();

  boost::shared_ptr < std::vector<ModelTPtr> > models = local->getModels ();
  boost::shared_ptr < std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > transforms = local->getTransforms ();

  std::cout << models->size() << std::endl;

  pcl::visualization::PCLVisualizer vis("LOLO");
  for (size_t j = 0; j < models->size (); j++)
  {
    std::cout << models->at(j)->id_ << std::endl;
    std::stringstream name;
    name << "cloud_" << j;

    std::cout << transforms->at(j) << std::endl;
    pcl::PointCloud<PointT>::ConstPtr model_cloud = models->at (j)->getAssembled (0.003f);
    pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
    Eigen::Matrix4f inv = transforms->at (j); //.inverse();
    pcl::transformPointCloud (*model_cloud, *model_aligned, inv);

    pcl::visualization::PointCloudColorHandlerRGBField < PointT> handler_rgb ( model_aligned );
    vis.addPointCloud< PointT > ( model_aligned, handler_rgb, name.str ());
  }

  pcl::visualization::PointCloudColorHandlerRGBField < PointT> handler_rgb ( scene );
  vis.addPointCloud< PointT > ( scene, handler_rgb, "scene");
  vis.spin();
}
