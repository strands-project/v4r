/*
 * shape_simple_classifier_node.cpp
 *
 *  Created on: Sep 7, 2013
 *      Author: aitor
 */

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <faat_pcl/3d_rec_framework/pc_source/registered_views_source.h>
#include <faat_pcl/3d_rec_framework/pc_source/partial_pcd_source.h>
#include "segmenter.h"
#include <faat_pcl/3d_rec_framework/pipeline/global_nn_recognizer_cvfh.h>
#include <faat_pcl/3d_rec_framework/pipeline/local_recognizer.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/global/color_ourcvfh_estimator.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/global/ourcvfh_estimator.h>
#include "faat_pcl/3d_rec_framework/utils/metrics.h"
#include <faat_pcl/3d_rec_framework/pc_source/registered_views_source.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/image/sift_local_estimator.h>
#include <faat_pcl/recognition/cg/graph_geometric_consistency.h>
#include <faat_pcl/3d_rec_framework/pipeline/multi_pipeline_recognizer.h>
#include <faat_pcl/recognition/hv/hv_go_1.h>
#include <faat_pcl/3d_rec_framework/segmentation/multiplane_segmentation.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/global/organized_color_ourcvfh_estimator.h>
#include <faat_pcl/3d_rec_framework/pipeline/global_nn_recognizer_cvfh.h>
#include <faat_pcl/3d_rec_framework/utils/metrics.h>
#include "recognition_srv_definitions/recognize.h"
#include <pcl/apps/dominant_plane_segmentation.h>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/image/opencv_sift_local_estimator.h>

#define USE_SIFT_GPU 
//#define SOC_VISUALIZE

struct camPosConstraints
{
    bool
    operator() (const Eigen::Vector3f & pos) const
    {
        if (pos[2] > 0)
            return true;

        return false;
    }
    ;
};

class Recognizer
{
private:
  typedef pcl::PointXYZRGB PointT;
  std::string models_dir_;
  std::string training_dir_sift_;
  std::string sift_structure_;
  std::string training_dir_ourcvfh_;
  bool do_sift_;
  bool do_ourcvfh_;
  float chop_at_z_;
  int icp_iterations_;
  std::vector<std::string> text_3d_;
  boost::shared_ptr<faat_pcl::rec_3d_framework::MultiRecognitionPipeline<PointT> > multi_recog_;
  int v1_,v2_, v3_;
  ros::ServiceServer recognize_;
  ros::NodeHandle n_;
#ifdef SOC_VISUALIZE
  boost::shared_ptr<pcl::visualization::PCLVisualizer> vis_;
#endif

  bool
  recognize (recognition_srv_definitions::recognize::Request & req, recognition_srv_definitions::recognize::Response & response)
  {
    typedef faat_pcl::rec_3d_framework::Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;
    typedef typename pcl::PointCloud<PointT>::ConstPtr ConstPointInTPtr;

    pcl::PointCloud<PointT>::Ptr scene (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg (req.cloud, *scene);

    float go_resolution_ = 0.005f;
    bool add_planes = true;
    float assembled_resolution = 0.003f;
    float color_sigma = 0.5f;

    //initialize go
    boost::shared_ptr<faat_pcl::GlobalHypothesesVerification_1<PointT, PointT> > go (
                    new faat_pcl::GlobalHypothesesVerification_1<PointT,
                    PointT>);

    go->setSmoothSegParameters(0.1, 0.035, 0.005);
    //go->setRadiusNormals(0.03f);
    go->setResolution (go_resolution_);
    go->setInlierThreshold (0.01);
    go->setRadiusClutter (0.03f);
    go->setRegularizer (2);
    go->setClutterRegularizer (5);
    go->setDetectClutter (true);
    go->setOcclusionThreshold (0.01f);
    go->setOptimizerType(0);
    go->setUseReplaceMoves(true);
    go->setRadiusNormals(0.02);
    go->setRequiresNormals(false);
    go->setInitialStatus(false);
    go->setIgnoreColor(false);
    go->setColorSigma(color_sigma);
    go->setUseSuperVoxels(false);

    typename pcl::PointCloud<PointT>::Ptr occlusion_cloud (new pcl::PointCloud<PointT>(*scene));
    if(chop_at_z_ > 0)
    {
        pcl::PassThrough<PointT> pass_;
        pass_.setFilterLimits (0.f, chop_at_z_);
        pass_.setFilterFieldName ("z");
        pass_.setInputCloud (scene);
        pass_.setKeepOrganized (true);
        pass_.filter (*scene);
    }

    pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
    ne.setRadiusSearch(0.02f);
    ne.setInputCloud (scene);
    ne.compute (*normal_cloud);

    //Multiplane segmentation
    faat_pcl::MultiPlaneSegmentation<PointT> mps;
    mps.setInputCloud(scene);
    mps.setMinPlaneInliers(1000);
    mps.setResolution(go_resolution_);
    mps.setNormals(normal_cloud);
    mps.setMergePlanes(true);
    std::vector<faat_pcl::PlaneModel<PointT> > planes_found;
    mps.segment();
    planes_found = mps.getModels();

    if(planes_found.size() == 0 && scene->isOrganized())
    {
        PCL_WARN("No planes found, doing segmentation with standard method\n");
        mps.segment(true);
        planes_found = mps.getModels();
    }

    std::vector<pcl::PointIndices> indices;
    Eigen::Vector4f table_plane;
    doSegmentation<PointT>(scene, normal_cloud, indices, table_plane);

    std::vector<int> indices_above_plane;
    for (int k = 0; k < scene->points.size (); k++)
    {
        Eigen::Vector3f xyz_p = scene->points[k].getVector3fMap ();
        if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
            continue;

        float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];
        if (val >= 0.01)
            indices_above_plane.push_back (static_cast<int> (k));
    }

    multi_recog_->setSceneNormals(normal_cloud);
    multi_recog_->setSegmentation(indices);
    multi_recog_->setIndices(indices_above_plane);
    multi_recog_->setInputCloud (scene);
    {
        pcl::ScopeTime ttt ("Recognition");
        multi_recog_->recognize ();
    }

    //HV
    //transforms models
    boost::shared_ptr < std::vector<ModelTPtr> > models = multi_recog_->getModels ();
    boost::shared_ptr < std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > transforms = multi_recog_->getTransforms ();

    std::vector<typename pcl::PointCloud<PointT>::ConstPtr> aligned_models;
    aligned_models.resize (models->size ());
    std::vector<std::string> model_ids;
    for (size_t kk = 0; kk < models->size (); kk++)
    {
        ConstPointInTPtr model_cloud = models->at (kk)->getAssembled (go_resolution_);
        typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
        pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (kk));
        aligned_models[kk] = model_aligned;
        model_ids.push_back(models->at (kk)->id_);
    }

    go->setSceneCloud (scene);
    go->setNormalsForClutterTerm(normal_cloud);
    go->setOcclusionCloud (occlusion_cloud);
    //addModels
    go->addModels (aligned_models, true);
    //append planar models
    if(add_planes)
    {
        go->addPlanarModels(planes_found);
        for(size_t kk=0; kk < planes_found.size(); kk++)
        {
            std::stringstream plane_id;
            plane_id << "plane_" << kk;
            model_ids.push_back(plane_id.str());
        }
    }

    go->setObjectIds(model_ids);
    //verify
    {
        pcl::ScopeTime t("Go verify");
        go->verify ();
    }
    std::vector<bool> mask_hv;
    go->getMask (mask_hv);

    std::vector<int> coming_from;
    coming_from.resize(aligned_models.size() + planes_found.size());
    for(size_t j=0; j < aligned_models.size(); j++)
        coming_from[j] = 0;

    for(size_t j=0; j < planes_found.size(); j++)
        coming_from[aligned_models.size() + j] = 1;

#ifdef SOC_VISUALIZE
    vis_->removeAllPointClouds();
    vis_->addPointCloud(scene, "scene", v1_);

    for(size_t kk=0; kk < planes_found.size(); kk++)
    {
        std::stringstream pname;
        pname << "plane_" << kk;
        pcl::visualization::PointCloudColorHandlerRandom<PointT> scene_handler(planes_found[kk].plane_cloud_);
        vis_->addPointCloud<PointT> (planes_found[kk].plane_cloud_, scene_handler, pname.str(), v2_);
        pname << "chull";
        vis_->addPolygonMesh (*planes_found[kk].convex_hull_, pname.str(), v2_);
    }
#endif

    boost::shared_ptr<std::vector<ModelTPtr> > verified_models(new std::vector<ModelTPtr>);
    boost::shared_ptr<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > verified_transforms;
    verified_transforms.reset(new std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >);

#ifdef SOC_VISUALIZE
    if(models)
    {
        for (size_t j = 0; j < mask_hv.size (); j++)
        {
            std::stringstream name;
            name << "cloud_" << j;

            if(!mask_hv[j])
            {
                if(coming_from[j] == 0)
                {
                    ConstPointInTPtr model_cloud = models->at (j)->getAssembled (assembled_resolution);
                    typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
                    pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (j));
                    pcl::visualization::PointCloudColorHandlerRGBField<PointT> random_handler (model_aligned);
                    vis_->addPointCloud<PointT> (model_aligned, random_handler, name.str (), v2_);
                }
                continue;
            }

            if(coming_from[j] == 0)
            {
                verified_models->push_back(models->at(j));
                verified_transforms->push_back(transforms->at(j));

                ConstPointInTPtr model_cloud = models->at (j)->getAssembled (-1);
                typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
                pcl::transformPointCloud (*model_cloud, *model_aligned, transforms->at (j));
                std::cout << models->at (j)->id_ << std::endl;

                pcl::visualization::PointCloudColorHandlerRGBField<PointT> random_handler (model_aligned);
                vis_->addPointCloud<PointT> (model_aligned, random_handler, name.str (), v3_);
            }
            else
            {
                std::stringstream pname;
                pname << "plane_v2_" << j;
                pcl::visualization::PointCloudColorHandlerRandom<PointT> scene_handler(planes_found[j - models->size()].plane_cloud_);
                vis_->addPointCloud<PointT> (planes_found[j - models->size()].plane_cloud_, scene_handler, pname.str(), v3_);
                pname << "chull_v2";
                vis_->addPolygonMesh (*planes_found[j - models->size()].convex_hull_, pname.str(), v3_);
            }
        }
    }
#else
    if(models)
    {
        for (size_t j = 0; j < mask_hv.size (); j++)
        {
            if(!mask_hv[j])
                continue;

            if(coming_from[j] == 0)
            {
                verified_models->push_back(models->at(j));
                verified_transforms->push_back(transforms->at(j));
            }
            else
            {
                /*std::stringstream pname;
                pname << "plane_v2_" << j;
                pcl::visualization::PointCloudColorHandlerRandom<PointT> scene_handler(planes_found[j - models->size()].plane_cloud_);
                vis_->addPointCloud<PointT> (planes_found[j - models->size()].plane_cloud_, scene_handler, pname.str(), v3_);
                pname << "chull_v2";
                vis_->addPolygonMesh (*planes_found[j - models->size()].convex_hull_, pname.str(), v3_);*/
            }
        }
    }
#endif

    std::cout << "Number of models:" << models->size() << std::endl;
    std::cout << "Number of verified models:" << verified_models->size() << std::endl;

    //parse verified_models and generate response to service call
      //vector of id + pose
    for (size_t j = 0; j < verified_models->size (); j++)
    {
      std_msgs::String ss;
      ss.data = verified_models->at(j)->id_;
      response.ids.push_back(ss);

      Eigen::Matrix4f trans = verified_transforms->at(j);
      geometry_msgs::Transform tt;
      tt.translation.x = trans(0,3);
      tt.translation.y = trans(1,3);
      tt.translation.z = trans(2,3);

      Eigen::Matrix3f rotation = trans.block<3,3>(0,0);
      Eigen::Quaternionf q(rotation);
      tt.rotation.x = q.x();
      tt.rotation.y = q.y();
      tt.rotation.z = q.z();
      tt.rotation.w = q.w();
      response.transforms.push_back(tt);
    }

#ifdef SOC_VISUALIZE
    vis_->spin ();
#endif

    return true;
  }
public:
  Recognizer ()
  {
    //default values
    chop_at_z_ = 1.f;
    do_sift_ = true;
    do_ourcvfh_ = false;
    icp_iterations_ = 0;

#ifdef SOC_VISUALIZE
    vis_.reset (new pcl::visualization::PCLVisualizer ("classifier visualization"));
    vis_->createViewPort(0,0,0.33,1.f, v1_);
    vis_->createViewPort(0.33,0,0.66,1.f, v2_);
    vis_->createViewPort(0.66,0,1,1.f, v3_);
#endif
  }

  void
  initialize (int argc, char ** argv)
  {

    pcl::console::parse_argument (argc, argv, "-models_dir", models_dir_);
    pcl::console::parse_argument (argc, argv, "-training_dir_sift", training_dir_sift_);
    pcl::console::parse_argument (argc, argv, "-recognizer_structure_sift", sift_structure_);
    pcl::console::parse_argument (argc, argv, "-training_dir_ourcvfh", training_dir_ourcvfh_);
    pcl::console::parse_argument (argc, argv, "-chop_z", chop_at_z_);
    pcl::console::parse_argument (argc, argv, "-icp_iterations", icp_iterations_);
    pcl::console::parse_argument (argc, argv, "-do_sift", do_sift_);
    pcl::console::parse_argument (argc, argv, "-do_ourcvfh", do_ourcvfh_);

    if (models_dir_.compare ("") == 0)
    {
      PCL_ERROR ("Set -models_dir option in the command line, ABORTING");
      return;
    }

    if (do_sift_ && training_dir_sift_.compare ("") == 0)
    {
      PCL_ERROR ("do_sift is activated but training_dir_sift_ is empty! Set -training_dir_sift option in the command line, ABORTING");
      return;
    }

    if (do_ourcvfh_ && training_dir_ourcvfh_.compare ("") == 0)
    {
      PCL_ERROR ("do_ourcvfh is activated but training_dir_ourcvfh_ is empty! Set -training_dir_ourcvfh option in the command line, ABORTING");
      return;
    }

    boost::function<bool (const Eigen::Vector3f &)> campos_constraints;
    campos_constraints = camPosConstraints ();

    multi_recog_.reset (new faat_pcl::rec_3d_framework::MultiRecognitionPipeline<PointT>);
    boost::shared_ptr < pcl::CorrespondenceGrouping<PointT, PointT> > cast_cg_alg;
    boost::shared_ptr < faat_pcl::GraphGeometricConsistencyGrouping<PointT, PointT> > gcg_alg (
                                                                                               new faat_pcl::GraphGeometricConsistencyGrouping<
                                                                                                   PointT, PointT>);

    gcg_alg->setGCThreshold (5);
    gcg_alg->setGCSize (0.015);
    gcg_alg->setRansacThreshold (0.01);
    gcg_alg->setUseGraph (true);
    gcg_alg->setDistForClusterFactor (0);
    gcg_alg->setDotDistance (0.2);
    cast_cg_alg = boost::static_pointer_cast<pcl::CorrespondenceGrouping<PointT, PointT> > (gcg_alg);

    if (do_sift_)
    {

      std::string idx_flann_fn = "sift_flann.idx";
      std::string desc_name = "sift";

      boost::shared_ptr < faat_pcl::rec_3d_framework::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT>
          > mesh_source (new faat_pcl::rec_3d_framework::RegisteredViewsSource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB, pcl::PointXYZRGB>);
      mesh_source->setPath (models_dir_);
      mesh_source->setModelStructureDir (sift_structure_);
      mesh_source->setLoadViews (false);
      mesh_source->generate (training_dir_sift_);

      boost::shared_ptr < faat_pcl::rec_3d_framework::Source<PointT> > cast_source;
      cast_source = boost::static_pointer_cast<faat_pcl::rec_3d_framework::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT> > (mesh_source);

#ifdef USE_SIFT_GPU
      boost::shared_ptr < faat_pcl::rec_3d_framework::SIFTLocalEstimation<PointT, pcl::Histogram<128> > > estimator;
      estimator.reset (new faat_pcl::rec_3d_framework::SIFTLocalEstimation<PointT, pcl::Histogram<128> >);

      boost::shared_ptr < faat_pcl::rec_3d_framework::LocalEstimator<PointT, pcl::Histogram<128> > > cast_estimator;
      cast_estimator = boost::dynamic_pointer_cast<faat_pcl::rec_3d_framework::SIFTLocalEstimation<PointT, pcl::Histogram<128> > > (estimator);
#else
	  boost::shared_ptr < faat_pcl::rec_3d_framework::OpenCVSIFTLocalEstimation<PointT, pcl::Histogram<128> > > estimator;	
      estimator.reset (new faat_pcl::rec_3d_framework::OpenCVSIFTLocalEstimation<PointT, pcl::Histogram<128> >);

      boost::shared_ptr < faat_pcl::rec_3d_framework::LocalEstimator<PointT, pcl::Histogram<128> > > cast_estimator;
      cast_estimator = boost::dynamic_pointer_cast<faat_pcl::rec_3d_framework::OpenCVSIFTLocalEstimation<PointT, pcl::Histogram<128> > > (estimator);
#endif

      boost::shared_ptr<faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<128> > > new_sift_local_;
      new_sift_local_.reset (new faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<128> > (idx_flann_fn));
      new_sift_local_->setDataSource (cast_source);
      new_sift_local_->setTrainingDir (training_dir_sift_);
      new_sift_local_->setDescriptorName (desc_name);
      new_sift_local_->setICPIterations (0);
      new_sift_local_->setFeatureEstimator (cast_estimator);
      new_sift_local_->setUseCache (true);
      new_sift_local_->setCGAlgorithm (cast_cg_alg);
      new_sift_local_->setKnn (5);
      new_sift_local_->setUseCache (true);
      new_sift_local_->initialize (false);

      boost::shared_ptr < faat_pcl::rec_3d_framework::Recognizer<PointT> > cast_recog;
      cast_recog = boost::static_pointer_cast<faat_pcl::rec_3d_framework::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<128> > > (
                                                                                                                                        new_sift_local_);
      multi_recog_->addRecognizer (cast_recog);
    }

    if(do_ourcvfh_)
    {
      boost::shared_ptr<faat_pcl::rec_3d_framework::PartialPCDSource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB> >
                          source (
                              new faat_pcl::rec_3d_framework::PartialPCDSource<
                              pcl::PointXYZRGBNormal,
                              pcl::PointXYZRGB>);
      source->setPath (models_dir_);
      source->setModelScale (1.f);
      source->setRadiusSphere (1.f);
      source->setTesselationLevel (1);
      source->setDotNormal (-1.f);
      source->setUseVertices(false);
      source->setLoadViews (false);
      source->setCamPosConstraints (campos_constraints);
      source->setLoadIntoMemory(false);
      source->setGenOrganized(true);
      source->setWindowSizeAndFocalLength(640, 480, 575.f);
      source->generate (training_dir_ourcvfh_);

      boost::shared_ptr<faat_pcl::rec_3d_framework::Source<pcl::PointXYZRGB> > cast_source;
      cast_source = boost::static_pointer_cast<faat_pcl::rec_3d_framework::PartialPCDSource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB> > (source);

      //configure normal estimator
      boost::shared_ptr<faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointT, pcl::Normal> > normal_estimator;
      normal_estimator.reset (new faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointT, pcl::Normal>);
      normal_estimator->setCMR (false);
      normal_estimator->setDoVoxelGrid (false);
      normal_estimator->setRemoveOutliers (false);
      normal_estimator->setValuesForCMRFalse (0.001f, 0.02f);
      normal_estimator->setForceUnorganized(true);

      //boost::shared_ptr<faat_pcl::rec_3d_framework::ColorOURCVFHEstimator<PointT, pcl::Histogram<1327> > > vfh_estimator;
      //vfh_estimator.reset (new faat_pcl::rec_3d_framework::ColorOURCVFHEstimator<PointT, pcl::Histogram<1327> >);

      boost::shared_ptr<faat_pcl::rec_3d_framework::OrganizedColorOURCVFHEstimator<PointT, pcl::Histogram<1327> > > vfh_estimator;
      vfh_estimator.reset (new faat_pcl::rec_3d_framework::OrganizedColorOURCVFHEstimator<PointT, pcl::Histogram<1327> >);
      vfh_estimator->setNormalEstimator (normal_estimator);
      vfh_estimator->setNormalizeBins(true);
      vfh_estimator->setUseRFForColor (true);
      //vfh_estimator->setRefineClustersParam (2.5f);
      vfh_estimator->setRefineClustersParam (100.f);
      vfh_estimator->setAdaptativeMLS (false);

      vfh_estimator->setAxisRatio (1.f);
      vfh_estimator->setMinAxisValue (1.f);

      {
          //segmentation parameters for training
          std::vector<float> eps_thresholds, cur_thresholds, clus_thresholds;
          eps_thresholds.push_back (0.15);
          cur_thresholds.push_back (0.015f);
          cur_thresholds.push_back (1.f);
          clus_thresholds.push_back (10.f);

          vfh_estimator->setClusterToleranceVector (clus_thresholds);
          vfh_estimator->setEpsAngleThresholdVector (eps_thresholds);
          vfh_estimator->setCurvatureThresholdVector (cur_thresholds);
      }

      std::string desc_name = "rf_our_cvfh_color_normalized";

      boost::shared_ptr<faat_pcl::rec_3d_framework::OURCVFHEstimator<pcl::PointXYZRGB, pcl::Histogram<1327> > > cast_estimator;
      cast_estimator = boost::dynamic_pointer_cast<faat_pcl::rec_3d_framework::OrganizedColorOURCVFHEstimator<pcl::PointXYZRGB, pcl::Histogram<1327> > > (vfh_estimator);

      boost::shared_ptr<faat_pcl::rec_3d_framework::GlobalNNCVFHRecognizer<faat_pcl::Metrics::HistIntersectionUnionDistance, PointT, pcl::Histogram<1327> > > rf_color_ourcvfh_global_;
      rf_color_ourcvfh_global_.reset(new faat_pcl::rec_3d_framework::GlobalNNCVFHRecognizer<faat_pcl::Metrics::HistIntersectionUnionDistance, PointT, pcl::Histogram<1327> >);
      rf_color_ourcvfh_global_->setDataSource (cast_source);
      rf_color_ourcvfh_global_->setTrainingDir (training_dir_ourcvfh_);
      rf_color_ourcvfh_global_->setDescriptorName (desc_name);
      rf_color_ourcvfh_global_->setFeatureEstimator (cast_estimator);
      rf_color_ourcvfh_global_->setNN (50);
      rf_color_ourcvfh_global_->setICPIterations (0);
      rf_color_ourcvfh_global_->setNoise (0.0f);
      rf_color_ourcvfh_global_->setUseCache (true);
      rf_color_ourcvfh_global_->setMaxHyp(15);
      rf_color_ourcvfh_global_->setMaxDescDistance(0.75f);
      rf_color_ourcvfh_global_->initialize (false);
      rf_color_ourcvfh_global_->setDebugLevel(2);
      {
          //segmentation parameters for recognition
          std::vector<float> eps_thresholds, cur_thresholds, clus_thresholds;
          eps_thresholds.push_back (0.15);
          cur_thresholds.push_back (0.015f);
          cur_thresholds.push_back (0.02f);
          cur_thresholds.push_back (1.f);
          clus_thresholds.push_back (10.f);

          vfh_estimator->setClusterToleranceVector (clus_thresholds);
          vfh_estimator->setEpsAngleThresholdVector (eps_thresholds);
          vfh_estimator->setCurvatureThresholdVector (cur_thresholds);

          vfh_estimator->setAxisRatio (0.8f);
          vfh_estimator->setMinAxisValue (0.8f);

          vfh_estimator->setAdaptativeMLS (false);
      }

      boost::shared_ptr < faat_pcl::rec_3d_framework::Recognizer<PointT> > cast_recog;
      cast_recog = boost::static_pointer_cast<faat_pcl::rec_3d_framework::GlobalNNCVFHRecognizer<faat_pcl::Metrics::HistIntersectionUnionDistance, PointT, pcl::Histogram<1327> > > (rf_color_ourcvfh_global_);
      multi_recog_->addRecognizer(cast_recog);
    }

    multi_recog_->setCGAlgorithm(gcg_alg);
    multi_recog_->setVoxelSizeICP(0.005f);
    multi_recog_->setICPType(1);
    multi_recog_->setICPIterations(icp_iterations_);
    multi_recog_->initialize();

    recognize_ = n_.advertiseService ("mp_recognition", &Recognizer::recognize, this);
    std::cout << "Ready to get service calls..." << std::endl;
    ros::spin ();
  }
};

int
main (int argc, char ** argv)
{
  ros::init (argc, argv, "recognition_service");

  Recognizer m;
  m.initialize (argc, argv);

  return 0;
}
