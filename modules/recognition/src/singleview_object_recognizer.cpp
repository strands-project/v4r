#include "v4r/recognition/singleview_object_recognizer.h"


#include <pcl/filters/passthrough.h>
#include <pcl/apps/dominant_plane_segmentation.h>

#include <v4r/recognition/local_recognizer.h>
#include <v4r/recognition/metrics.h>
#include <v4r/segmentation/multiplane_segmentation.h>
#include <v4r/features/organized_color_ourcvfh_estimator.h>
#include <v4r/features/ourcvfh_estimator.h>
#include <v4r/features/shot_local_estimator.h>
#include <v4r/features/shot_local_estimator_omp.h>
#include <v4r/recognition/registered_views_source.h>
#include <v4r/recognition/partial_pcd_source.h>
#include <v4r/recognition/global_nn_recognizer_cvfh.h>
#include <v4r/common/graph_geometric_consistency.h>
#include <v4r/recognition/ghv.h>
//#include <v4r/recognition/hv/hv_cuda_wrapper.h>
#include <v4r/common/visibility_reasoning.h>
#include <v4r/common/miscellaneous.h>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>

//#define USE_CUDA

#ifndef USE_SIFT_GPU
    #include <v4r/features/opencv_sift_local_estimator.h>
#endif

#ifdef USE_CUDA
    #include <v4r/recognition/include/cuda/ghv_cuda.h>
    #include <v4r/recognition/include/cuda/ghv_cuda_wrapper.h>
#endif


bool USE_SEGMENTATION_ = false;

namespace v4r
{
void SingleViewRecognizer::multiplaneSegmentation()
{
    //Multiplane segmentation
    v4r::MultiPlaneSegmentation<PointT> mps;
    mps.setInputCloud(pInputCloud_);
    mps.setMinPlaneInliers(1000);
    mps.setResolution(hv_params_.resolution_);
    mps.setNormals(pSceneNormals_);
    mps.setMergePlanes(true);
    mps.segment();
    planes_found_ = mps.getModels();
}

bool SingleViewRecognizer::retrain (const std::vector<std::string> &model_ids)
{
      //delete .idx files from recognizers
      { //sift flann idx
          bf::path file(idx_flann_fn_sift_);
           if(bf::exists(file))
              bf::remove(file);
      }

      { //shot flann idx
        bf::path file(idx_flann_fn_shot_);
         if(bf::exists(file))
            bf::remove(file);
      }

      std::cout << "Never finishes this..." << std::endl;

      if(model_ids.empty())
      {
          std::cout << "Number of model ids is zero. " << std::endl;
          multi_recog_->reinitialize();
      }
      else
      {
          std::cout << "Number of ids:" << model_ids.size() << std::endl;
          multi_recog_->reinitialize(model_ids);
      }

      return true;
}


void SingleViewRecognizer::constructHypotheses()
{
        multi_recog_->setSceneNormals(pSceneNormals_);
        multi_recog_->setInputCloud (pInputCloud_);
        multi_recog_->setSaveHypotheses(true);
        multi_recog_->recognize ();
        multi_recog_->getSavedHypotheses(hypotheses_);
        multi_recog_->getKeypointCloud(pKeypointsMultipipe_);
        multi_recog_->getKeypointIndices(keypointIndices_);

        models_ = multi_recog_->getModels ();
        transforms_ = multi_recog_->getTransforms ();
}

std::vector<bool> SingleViewRecognizer::hypothesesVerification()
{
    std::cout << "=================================================================" << std::endl <<
                 "Verifying hypotheses on CPU with following parameters: " << std::endl <<
                 "*** Resolution: " << hv_params_.resolution_ << std::endl <<
                 "*** Inlier Threshold: " << hv_params_.inlier_threshold_ << std::endl <<
                 "*** Radius clutter: " << hv_params_.radius_clutter_ << std::endl <<
                 "*** Regularizer: " << hv_params_.regularizer_ << std::endl <<
                 "*** Clutter regularizer: " << hv_params_.clutter_regularizer_ << std::endl <<
                 "*** Occlusion threshold: " << hv_params_.occlusion_threshold_ << std::endl <<
                 "*** Optimizer type: " << hv_params_.optimizer_type_ << std::endl <<
                 "*** Color sigma L / AB: " << hv_params_.color_sigma_l_ << " / " << hv_params_.color_sigma_ab_ << std::endl <<
                 "*** Use supervoxels: " << hv_params_.use_supervoxels_ << std::endl <<
                 "*** Use detect clutter: " << hv_params_.detect_clutter_ << std::endl <<
                 "*** Use ignore colors: " << hv_params_.ignore_color_ << std::endl <<
                 "=================================================================" << std::endl << std::endl;

    std::vector<bool> mask_hv (aligned_models_.size());
    pcl::PointCloud<PointT>::Ptr occlusion_cloud (new pcl::PointCloud<PointT>(*pInputCloud_));

    //initialize go
#ifdef USE_CUDA
    boost::shared_ptr<v4r::recognition::GHVCudaWrapper<PointT> > go (new v4r::recognition::GHVCudaWrapper<PointT>);
#else
    v4r::GHV<PointT, PointT>::ParameterGHV paramGHV;
    paramGHV.resolution_ = hv_params_.resolution_;
    paramGHV.occlusion_thres_ = hv_params_.occlusion_threshold_;
    paramGHV.inliers_threshold_ = hv_params_.inlier_threshold_;

    boost::shared_ptr<v4r::GHV<PointT, PointT> > go (new v4r::GHV<PointT, PointT>(paramGHV));
    //go->setRadiusNormals(0.03f);
    go->setDetectClutter (hv_params_.detect_clutter_);
    go->setOptimizerType (hv_params_.optimizer_type_);
    go->setUseReplaceMoves(hv_params_.use_replace_moves_);
    go->setRadiusNormals (hv_params_.radius_normals_);
    go->setRequiresNormals(hv_params_.requires_normals_);
    go->setInitialStatus(hv_params_.initial_status_);
    go->setIgnoreColor(hv_params_.ignore_color_);
    go->setHistogramSpecification(hv_params_.histogram_specification_);
    go->setSmoothSegParameters(hv_params_.smooth_seg_params_eps_,
                               hv_params_.smooth_seg_params_curv_t_,
                               hv_params_.smooth_seg_params_dist_t_,
                               hv_params_.smooth_seg_params_min_points_);//0.1, 0.035, 0.005);
    go->setVisualizeGoCues(0);
    go->setUseSuperVoxels(hv_params_.use_supervoxels_);
    go->setZBufferSelfOcclusionResolution (hv_params_.z_buffer_self_occlusion_resolution_);
    go->setHypPenalty (hv_params_.hyp_penalty_);
    go->setDuplicityCMWeight(hv_params_.duplicity_cm_weight_);
#endif

    assert(pSceneNormals_->points.size() == pInputCloud_->points.size());
    go->setRadiusClutter (hv_params_.radius_clutter_);
    go->setRegularizer (hv_params_.regularizer_ );
    go->setClutterRegularizer (hv_params_.clutter_regularizer_);
    go->setColorSigma (hv_params_.color_sigma_l_, hv_params_.color_sigma_ab_);
    go->setOcclusionCloud (occlusion_cloud);
    go->setSceneCloud (pInputCloud_);
    go->setNormalsForClutterTerm(pSceneNormals_);


#ifdef USE_CUDA
    std::vector<pcl::PointCloud<PointT>::ConstPtr> aligned_models;
    std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr> aligned_normals;
    std::vector<pcl::PointCloud<pcl::PointXYZL>::Ptr> aligned_smooth_faces;

    aligned_models.resize (models_->size ());
    aligned_smooth_faces.resize (models_->size ());
    aligned_normals.resize (models_->size ());

    std::map<std::string, int> id_to_model_clouds;
    std::map<std::string, int>::iterator it;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transformations;
    std::vector<int> transforms_to_models;
    transforms_to_models.resize(models_->size());
    transformations.resize(models_->size());

    int individual_models = 0;

    for (size_t kk = 0; kk < models_->size (); kk++)
    {

        int pos = 0;
        it = id_to_model_clouds.find(models_->at(kk)->id_);
        if(it == id_to_model_clouds.end())
        {
            //not included yet
            ConstPointInTPtr model_cloud = models_->at (kk)->getAssembled (hv_params_.resolution_);
            pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud = models_->at (kk)->getNormalsAssembled (hv_params_.resolution_);
            aligned_models[individual_models] = model_cloud;
            aligned_normals[individual_models] = normal_cloud;
            pos = individual_models;

            id_to_model_clouds.insert(std::make_pair(models_->at(kk)->id_, individual_models));

            individual_models++;
        }
        else
        {
            pos = it->second;
        }

        transformations[kk] = transforms_->at(kk);
        transforms_to_models[kk] = pos;
    }

    aligned_models.resize(individual_models);
    aligned_normals.resize(individual_models);
    std::cout << "aligned models size:" << aligned_models.size() << " " << models_->size() << std::endl;

    go->addModelNormals(aligned_normals);
    go->addModels(aligned_models, transformations, transforms_to_models);
#else
    go->addModels (aligned_models_, true);

    go->addNormalsClouds(aligned_normals_);
#endif

    //append planar models
    if(sv_params_.add_planes_)
    {
        multiplaneSegmentation();
        go->addPlanarModels(planes_found_);
        for(size_t kk=0; kk < planes_found_.size(); kk++)
        {
            std::stringstream plane_id;
            plane_id << "plane_" << kk;
        }
    }

    if(!models_.size())
    {
        std::cout << "No models to verify, returning... " << std::endl;
        std::cout << "Cancelling service request." << std::endl;
        return mask_hv;
    }

    go->verify ();

    std::vector<bool> mask_hv_with_planes;
    verified_planes_.clear();
    go->getMask (mask_hv_with_planes);

    std::vector<int> coming_from;
    coming_from.resize(aligned_models_.size() + planes_found_.size());

    for(size_t j=0; j < aligned_models_.size(); j++)
        coming_from[j] = 0;

    for(size_t j=0; j < planes_found_.size(); j++)
        coming_from[aligned_models_.size() + j] = 1;

    for (size_t j = 0; j < aligned_models_.size (); j++)
        mask_hv[j] = mask_hv_with_planes[j];

    for (size_t j = 0; j < planes_found_.size(); j++)
    {
        if(mask_hv_with_planes[aligned_models_.size () + j])
            verified_planes_.push_back(planes_found_[j].plane_cloud_);
    }

    return mask_hv;
}

/*
 * Correspondence grouping (clustering) for existing feature matches. If enough points (>cg_size) are
 * in the same cluster and vote for the same model, a hypothesis (with pose estimate) is constructed.
 */
void SingleViewRecognizer::constructHypothesesFromFeatureMatches(std::map < std::string,v4r::ObjectHypothesis<PointT> > hypothesesInput,
                                                           const pcl::PointCloud<PointT>::Ptr pKeypoints,
                                                           const pcl::PointCloud<pcl::Normal>::Ptr pKeypointNormals,
                                                           std::vector<Hypothesis<PointT> > &hypothesesOutput,
                                                           std::vector <pcl::Correspondences>  &corresp_clusters_hyp)
{
    std::cout << "=================================================================" << std::endl <<
                 "Start correspondence grouping with following parameters: " << std::endl <<
                 "Threshold: " << cg_params_.cg_size_threshold_ << std::endl <<
                 "cg_size_: " << cg_params_.cg_size_ << std::endl <<
                 "ransac_threshold_: " << cg_params_.ransac_threshold_ << std::endl <<
                 "dist_for_clutter_factor_: " << cg_params_.dist_for_clutter_factor_ << std::endl <<
                 "max_taken_: " << cg_params_.max_taken_ << std::endl <<
                 "max_taken_: " << cg_params_.max_taken_ << std::endl <<
                 "dot_distance_: " << cg_params_.dot_distance_ << std::endl <<
                 "=================================================================" << std::endl << std::endl;

    aligned_models_.clear();
    aligned_normals_.clear();
    transforms_.clear();
    models_.clear();
    models_verified_.clear();
    transforms_verified_.clear();

    hypothesesOutput.clear();
    std::map<std::string, v4r::ObjectHypothesis<PointT> >::iterator it_map;
    std::cout << "I have " << hypothesesInput.size() << " hypotheses. " << std::endl;

//#pragma omp parallel
    for (it_map = hypothesesInput.begin (); it_map != hypothesesInput.end (); it_map++)
    {
        const v4r::ObjectHypothesis<PointT> &oh = it_map->second;

        if((int)oh.model_scene_corresp_->size() < cg_params_.cg_size_threshold_)
            continue;

        std::vector <pcl::Correspondences> corresp_clusters;
        std::cout << oh.model_->id_ << ": " << oh.model_scene_corresp_->size() << std::endl;
        cast_cg_alg_->setSceneCloud (pKeypoints);
        cast_cg_alg_->setInputCloud (oh.model_->keypoints_);

        if(cast_cg_alg_->getRequiresNormals())
        {
            cast_cg_alg_->setInputAndSceneNormals(oh.model_->kp_normals_, oh.scene_normals_);
        }

        cast_cg_alg_->setModelSceneCorrespondences (oh.model_scene_corresp_);
        cast_cg_alg_->cluster (corresp_clusters);

        std::cout << "Instances:" << corresp_clusters.size () << " Total correspondences:" << oh.model_scene_corresp_->size () << " " << it_map->first << std::endl;
        for (size_t i = 0; i < corresp_clusters.size (); i++)
        {
            Eigen::Matrix4f best_trans;
            pcl::registration::TransformationEstimationSVD < PointT, PointT > t_est;
            t_est.estimateRigidTransformation (*oh.model_->keypoints_, *pKeypoints, corresp_clusters[i], best_trans);

            Hypothesis<PointT> ht_temp ( oh.model_, best_trans );
            hypothesesOutput.push_back(ht_temp);
            corresp_clusters_hyp.push_back( corresp_clusters[i] );
            models_.push_back( oh.model_ );
            transforms_.push_back( best_trans );

            ConstPointInTPtr model_cloud = oh.model_->getAssembled (hv_params_.resolution_);
            pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
            pcl::transformPointCloud (*model_cloud, *model_aligned, best_trans);
            aligned_models_.push_back(model_aligned);

            pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud_const = oh.model_->getNormalsAssembled (hv_params_.resolution_);
            pcl::PointCloud<pcl::Normal>::Ptr normal_cloud(new pcl::PointCloud<pcl::Normal>(*normal_cloud_const) );

            const Eigen::Matrix3f rot   = transforms_[i].block<3, 3> (0, 0);
            for(size_t jj=0; jj < normal_cloud->points.size(); jj++)
            {
                const pcl::Normal norm_pt = normal_cloud->points[jj];
                normal_cloud->points[jj].getNormalVector3fMap() = rot * norm_pt.getNormalVector3fMap();
            }
            aligned_normals_.push_back(normal_cloud);
        }
    }
}

void SingleViewRecognizer::preFilterWithFSV(const pcl::PointCloud<PointT>::ConstPtr scene_cloud, std::vector<float> &fsv)
{
    pcl::PointCloud<PointT>::Ptr occlusion_cloud (new pcl::PointCloud<PointT> (*scene_cloud));
    fsv.resize(models_.size());

    if(occlusion_cloud->isOrganized())
    {
        //compute FSV for the model and occlusion_cloud
        v4r::VisibilityReasoning<PointT> vr (525.f, 640, 480);
        vr.setThresholdTSS (0.01f);

        for(size_t i=0; i < models_.size(); i++)
        {
            pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud = models_[i]->getNormalsAssembled (hv_params_.resolution_);
            pcl::PointCloud<pcl::Normal>::Ptr normal_aligned (new pcl::PointCloud<pcl::Normal>);
            v4r::transformNormals(normal_cloud, normal_aligned, transforms_[i]);

            if(models_[i]->getFlipNormalsBasedOnVP())
            {
                const Eigen::Vector3f viewpoint = Eigen::Vector3f(0,0,0);

                for(size_t kk=0; kk < aligned_models_[i]->points.size(); kk++)
                {
                    Eigen::Vector3f n = normal_aligned->points[kk].getNormalVector3fMap();
                    n.normalize();
                    const Eigen::Vector3f p = aligned_models_[i]->points[kk].getVector3fMap();
                    Eigen::Vector3f d = viewpoint - p;
                    d.normalize();
                    if(n.dot(d) < 0)
                    {
                        normal_aligned->points[kk].getNormalVector3fMap() = normal_aligned->points[kk].getNormalVector3fMap() * -1;
                    }
                }
            }
            fsv[i] = vr.computeFSVWithNormals (occlusion_cloud, aligned_models_[i], normal_aligned);
        }
    }
    else
    {
        PCL_ERROR("Occlusion cloud is not organized. Cannot compute FSV.");
    }
}

bool SingleViewRecognizer::recognize ()
{
    transforms_.clear();
    transforms_verified_.clear();
    models_.clear();
    models_verified_.clear();
    aligned_models_.clear();

    if(pSceneNormals_->points.size() == 0) {
        std::cout << "No normals point cloud for scene given. Calculate normals of scene..." << std::endl;
        v4r::computeNormals<PointT>(pInputCloud_, pSceneNormals_, sv_params_.normal_computation_method_);
    }

    if( sv_params_.chop_at_z_ > 0) {
        pcl::PassThrough<PointT> pass;
        pass.setFilterLimits ( 0.f, sv_params_.chop_at_z_ );
        pass.setFilterFieldName ("z");
        pass.setInputCloud (pInputCloud_);
        pass.setKeepOrganized (true);
        pass.filter (*pInputCloud_);
        pcl::copyPointCloud(*pSceneNormals_, *pass.getIndices(), *pSceneNormals_);
    }

    bool have_a_finite_pt = false;
    for(size_t pt_id=0; pt_id<pInputCloud_->points.size(); pt_id++) {
        if ( pcl::isFinite (pInputCloud_->points[pt_id]) ) {
            have_a_finite_pt = true;
            break;
        }
    }

    if(!have_a_finite_pt)
        return false;

    constructHypotheses();
    setModelsAndTransforms(models_, transforms_);

    std::vector<bool> mask_hv = hypothesesVerification();

    for (size_t j = 0; j < mask_hv.size (); j++) {
        if( mask_hv[j] ) {
            models_verified_.push_back(models_[j]);
            transforms_verified_.push_back(transforms_[j]);
        }
    }

    std::cout << "Number of models: " << models_.size() << ", number of verified models: " << models_verified_.size() << std::endl;

    return true;
  }


void SingleViewRecognizer::printParams(std::ostream &ostr) const
{
    ostr      << "cg_size_thresh: " << cg_params_.cg_size_threshold_ << std::endl
              << "cg_size: " << cg_params_.cg_size_ << std::endl
              << "cg_ransac_threshold: " << cg_params_.ransac_threshold_ << std::endl
              << "cg_dist_for_clutter_factor: " << cg_params_.dist_for_clutter_factor_ << std::endl
              << "cg_max_taken: " << cg_params_.max_taken_ << std::endl
              << "cg_max_time_for_cliques_computation: " << cg_params_.max_time_for_cliques_computation_ << std::endl
              << "cg_dot_distance: " << cg_params_.dot_distance_ << std::endl
              << "cg_use_cg_graph: " << cg_params_.use_cg_graph_ << std::endl
              << "hv_resolution: " << hv_params_.resolution_ << std::endl
              << "hv_inlier_threshold: " << hv_params_.inlier_threshold_ << std::endl
              << "hv_radius_clutter: " << hv_params_.radius_clutter_ << std::endl
              << "hv_regularizer: " << hv_params_.regularizer_ << std::endl
              << "hv_clutter_regularizer: " << hv_params_.clutter_regularizer_ << std::endl
              << "hv_occlusion_threshold: " << hv_params_.occlusion_threshold_ << std::endl
              << "hv_optimizer_type: " << hv_params_.optimizer_type_ << std::endl
              << "hv_color_sigma_l: " << hv_params_.color_sigma_l_ << std::endl
              << "hv_color_sigma_ab: " << hv_params_.color_sigma_ab_ << std::endl
              << "hv_use_supervoxels: " << hv_params_.use_supervoxels_ << std::endl
              << "hv_detect_clutter: " << hv_params_.detect_clutter_ << std::endl
              << "hv_ignore_color: " << hv_params_.ignore_color_ << std::endl
              << "chop_z: " << sv_params_.chop_at_z_ << std::endl
              << "icp_iterations: " << sv_params_.icp_iterations_ << std::endl
              << "icp_type: " << sv_params_.icp_type_ << std::endl
              << "icp_voxel_size: " << hv_params_.resolution_ << std::endl
              << "do_sift: " << sv_params_.do_sift_ << std::endl
              << "do_shot: " << sv_params_.do_shot_ << std::endl
              << "do_ourcvfh: " << sv_params_.do_ourcvfh_ << std::endl
              << "====================" << std::endl << std::endl;
}

  void SingleViewRecognizer::initialize ()
  {
    boost::function<bool (const Eigen::Vector3f &)> campos_constraints;
    campos_constraints = camPosConstraints ();

    multi_recog_.reset (new v4r::MultiRecognitionPipeline<PointT>);

    boost::shared_ptr < v4r::GraphGeometricConsistencyGrouping<PointT, PointT> > gcg_alg (
                new v4r::GraphGeometricConsistencyGrouping<PointT, PointT>);
    gcg_alg->setGCThreshold (cg_params_.cg_size_threshold_);
    gcg_alg->setGCSize (cg_params_.cg_size_);
    gcg_alg->setRansacThreshold (cg_params_.ransac_threshold_);
    gcg_alg->setUseGraph (cg_params_.use_cg_graph_);
    gcg_alg->setDistForClusterFactor (cg_params_.dist_for_clutter_factor_);
    gcg_alg->setMaxTaken(cg_params_.max_taken_);
    gcg_alg->setMaxTimeForCliquesComputation(cg_params_.max_time_for_cliques_computation_);
    gcg_alg->setDotDistance (cg_params_.dot_distance_);

    cast_cg_alg_ = boost::static_pointer_cast<v4r::CorrespondenceGrouping<PointT, PointT> > (gcg_alg);

    if ( sv_params_.do_sift_ )
    {
      std::string desc_name = "sift";

      boost::shared_ptr < v4r::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT>
          > mesh_source (new v4r::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT>);
      mesh_source->setPath (models_dir_);
      mesh_source->setModelStructureDir (training_dir_);
      std::string foo;
      mesh_source->generate (foo);
      mesh_source->createVoxelGridAndDistanceTransform(hv_params_.resolution_);

      boost::shared_ptr < v4r::Source<PointT> > cast_source;
      cast_source = boost::static_pointer_cast<v4r::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT> > (mesh_source);

#ifdef USE_SIFT_GPU

      if(!sift_) { //--create a new SIFT-GPU context
          static char kw[][16] = {"-m", "-fo", "-1", "-s", "-v", "1", "-pack"};
          char * argvv[] = {kw[0], kw[1], kw[2], kw[3],kw[4],kw[5],kw[6], NULL};

          int argcc = sizeof(argvv) / sizeof(char*);
          sift_ = new SiftGPU ();
          sift_->ParseParam (argcc, argvv);

          //create an OpenGL context for computation
          if (sift_->CreateContextGL () != SiftGPU::SIFTGPU_FULL_SUPPORTED)
            throw std::runtime_error ("PSiftGPU::PSiftGPU: No GL support!");
      }

      boost::shared_ptr < v4r::SIFTLocalEstimation<PointT, FeatureT > > estimator;
      estimator.reset (new v4r::SIFTLocalEstimation<PointT, FeatureT >(sift_));

      boost::shared_ptr < v4r::LocalEstimator<PointT, FeatureT > > cast_estimator;
      cast_estimator = boost::dynamic_pointer_cast<v4r::SIFTLocalEstimation<PointT, FeatureT > > (estimator);
#else
      boost::shared_ptr < v4r::OpenCVSIFTLocalEstimation<PointT, FeatureT > > estimator;
      estimator.reset (new v4r::OpenCVSIFTLocalEstimation<PointT, FeatureT >);

      boost::shared_ptr < v4r::LocalEstimator<PointT, FeatureT > > cast_estimator;
      cast_estimator = boost::dynamic_pointer_cast<v4r::OpenCVSIFTLocalEstimation<PointT, FeatureT > > (estimator);
#endif

      boost::shared_ptr<v4r::LocalRecognitionPipeline<flann::L1, PointT, FeatureT > > new_sift_local;
      new_sift_local.reset (new v4r::LocalRecognitionPipeline<flann::L1, PointT, FeatureT > (idx_flann_fn_sift_));
      new_sift_local->setDataSource (cast_source);
      new_sift_local->setTrainingDir (training_dir_);
      new_sift_local->setDescriptorName (desc_name);
      new_sift_local->setICPIterations (sv_params_.icp_iterations_);
      new_sift_local->setFeatureEstimator (cast_estimator);
      new_sift_local->setUseCache (true);
      new_sift_local->setCGAlgorithm (cast_cg_alg_);
      new_sift_local->setKnn (sv_params_.knn_sift_);
      new_sift_local->setUseCache (true);
      new_sift_local->setSaveHypotheses(true);
      new_sift_local->initialize (false);

      boost::shared_ptr < v4r::Recognizer<PointT> > cast_recog;
      cast_recog = boost::static_pointer_cast<v4r::LocalRecognitionPipeline<flann::L1, PointT, FeatureT > > (
                                                                                                                                        new_sift_local);
      std::cout << "Feature Type: " << cast_recog->getFeatureType() << std::endl;
      multi_recog_->addRecognizer (cast_recog);
    }

    if(sv_params_.do_ourcvfh_ && USE_SEGMENTATION_)
    {
      boost::shared_ptr<v4r::PartialPCDSource<pcl::PointXYZRGBNormal, PointT> >
                          source ( new v4r::PartialPCDSource< pcl::PointXYZRGBNormal, PointT>);
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
      source->generate (training_dir_);
      source->createVoxelGridAndDistanceTransform(hv_params_.resolution_);

      boost::shared_ptr<v4r::Source<PointT> > cast_source;
      cast_source = boost::static_pointer_cast<v4r::PartialPCDSource<pcl::PointXYZRGBNormal, PointT> > (source);

      //configure normal estimator
      boost::shared_ptr<v4r::PreProcessorAndNormalEstimator<PointT, pcl::Normal> > normal_estimator;
      normal_estimator.reset (new v4r::PreProcessorAndNormalEstimator<PointT, pcl::Normal>);
      normal_estimator->setCMR (false);
      normal_estimator->setDoVoxelGrid (false);
      normal_estimator->setRemoveOutliers (false);
      normal_estimator->setValuesForCMRFalse (0.001f, 0.02f);
      normal_estimator->setForceUnorganized(true);

      //boost::shared_ptr<v4r::ColorOURCVFHEstimator<PointT, pcl::Histogram<1327> > > vfh_estimator;
      //vfh_estimator.reset (new v4r::ColorOURCVFHEstimator<PointT, pcl::Histogram<1327> >);

      boost::shared_ptr<v4r::OrganizedColorOURCVFHEstimator<PointT, pcl::Histogram<1327> > > vfh_estimator;
      vfh_estimator.reset (new v4r::OrganizedColorOURCVFHEstimator<PointT, pcl::Histogram<1327> >);
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

      boost::shared_ptr<v4r::OURCVFHEstimator<PointT, pcl::Histogram<1327> > > cast_estimator;
      cast_estimator = boost::dynamic_pointer_cast<v4r::OrganizedColorOURCVFHEstimator<PointT, pcl::Histogram<1327> > > (vfh_estimator);

      boost::shared_ptr<v4r::GlobalNNCVFHRecognizer<v4r::Metrics::HistIntersectionUnionDistance, PointT, pcl::Histogram<1327> > > rf_color_ourcvfh_global_;
      rf_color_ourcvfh_global_.reset(new v4r::GlobalNNCVFHRecognizer<v4r::Metrics::HistIntersectionUnionDistance, PointT, pcl::Histogram<1327> >);
      rf_color_ourcvfh_global_->setDataSource (cast_source);
      rf_color_ourcvfh_global_->setTrainingDir (training_dir_);
      rf_color_ourcvfh_global_->setDescriptorName (desc_name);
      rf_color_ourcvfh_global_->setFeatureEstimator (cast_estimator);
      rf_color_ourcvfh_global_->setNN (50);
      rf_color_ourcvfh_global_->setICPIterations ( sv_params_.icp_iterations_ );
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

      boost::shared_ptr < v4r::Recognizer<PointT> > cast_recog;
      cast_recog = boost::static_pointer_cast<v4r::GlobalNNCVFHRecognizer<v4r::Metrics::HistIntersectionUnionDistance, PointT, pcl::Histogram<1327> > > (rf_color_ourcvfh_global_);
      multi_recog_->addRecognizer(cast_recog);
    }

    if(sv_params_.do_shot_)
    {
        std::string desc_name = "shot";
        bool use_cache = true;
        float test_sampling_density = 0.01f;

        //configure mesh source
        boost::shared_ptr < v4r::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT>
                > mesh_source (new v4r::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT>);
        mesh_source->setPath (models_dir_);
        mesh_source->setModelStructureDir (training_dir_);
        mesh_source->setLoadViews(false);
        std::string foo;
        mesh_source->generate (foo);
        mesh_source->createVoxelGridAndDistanceTransform(hv_params_.resolution_);

        boost::shared_ptr < v4r::Source<PointT> > cast_source;
        cast_source = boost::static_pointer_cast<v4r::RegisteredViewsSource<pcl::PointXYZRGBNormal, PointT, PointT> > (mesh_source);

        boost::shared_ptr<v4r::UniformSamplingExtractor<PointT> > uniform_keypoint_extractor ( new v4r::UniformSamplingExtractor<PointT>);
        uniform_keypoint_extractor->setSamplingDensity (0.01f);
        uniform_keypoint_extractor->setFilterPlanar (true);
        uniform_keypoint_extractor->setMaxDistance( sv_params_.chop_at_z_ );
        uniform_keypoint_extractor->setThresholdPlanar(0.1);

        boost::shared_ptr<v4r::KeypointExtractor<PointT> > keypoint_extractor;
        keypoint_extractor = boost::static_pointer_cast<v4r::KeypointExtractor<PointT> > (uniform_keypoint_extractor);

        boost::shared_ptr<v4r::PreProcessorAndNormalEstimator<PointT, pcl::Normal> > normal_estimator;
        normal_estimator.reset (new v4r::PreProcessorAndNormalEstimator<PointT, pcl::Normal>);
        normal_estimator->setCMR (false);
        normal_estimator->setDoVoxelGrid (true);
        normal_estimator->setRemoveOutliers (false);
        normal_estimator->setValuesForCMRFalse (0.003f, 0.02f);

        boost::shared_ptr<v4r::SHOTLocalEstimationOMP<PointT, pcl::Histogram<352> > > estimator;
        estimator.reset (new v4r::SHOTLocalEstimationOMP<PointT, pcl::Histogram<352> >);
        estimator->setNormalEstimator (normal_estimator);
        estimator->addKeypointExtractor (keypoint_extractor);
        estimator->setSupportRadius (0.04f);
        estimator->setAdaptativeMLS (false);

        boost::shared_ptr<v4r::LocalEstimator<PointT, pcl::Histogram<352> > > cast_estimator;
        cast_estimator = boost::dynamic_pointer_cast<v4r::LocalEstimator<PointT, pcl::Histogram<352> > > (estimator);

        boost::shared_ptr<v4r::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<352> > > local;
        local.reset(new v4r::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<352> > (idx_flann_fn_shot_));
        local->setDataSource (cast_source);
        local->setTrainingDir(training_dir_);
        local->setDescriptorName (desc_name);
        local->setFeatureEstimator (cast_estimator);
//        local->setCGAlgorithm (cast_cg_alg_);
        local->setKnn(sv_params_.knn_shot_);
        local->setUseCache (use_cache);
        local->setThresholdAcceptHyp (1);
        uniform_keypoint_extractor->setSamplingDensity (test_sampling_density);
        local->setICPIterations ( sv_params_.icp_iterations_ );
        local->setKdtreeSplits (128);
        local->setSaveHypotheses(true);
        local->initialize (false);
        local->setMaxDescriptorDistance(std::numeric_limits<float>::infinity());

        boost::shared_ptr<v4r::Recognizer<PointT> > cast_recog;
        cast_recog = boost::static_pointer_cast<v4r::LocalRecognitionPipeline<flann::L1, PointT, pcl::Histogram<352> > > (local);
        multi_recog_->addRecognizer(cast_recog);
    }

//    multi_recog_->setSaveHypotheses(true);
    multi_recog_->setVoxelSizeICP(hv_params_.resolution_);
    multi_recog_->setICPType(sv_params_.icp_type_);
    multi_recog_->setCGAlgorithm(gcg_alg);
//    multi_recog_->setVoxelSizeICP(0.005f);
//    multi_recog_->setICPType(1);
    multi_recog_->setICPIterations(sv_params_.icp_iterations_);
    multi_recog_->initialize();
  }

  void SingleViewRecognizer::visualize()
  {
      if(!vis_) {
          vis_.reset(new pcl::visualization::PCLVisualizer("single-view recognition results"));
          vis_->createViewPort(0,0,1,0.33,vp1_);
          vis_->createViewPort(0,0.33,1,0.66,vp2_);
          vis_->createViewPort(0,0.66,1,1,vp3_);
          vis_->addText("input cloud", 10, 10, 20, 1, 1, 1, "input", vp1_);
          vis_->addText("generated hypotheses", 10, 10, 20, 0, 0, 0, "generated hypotheses", vp2_);
          vis_->addText("verified hypotheses", 10, 10, 20, 0, 0, 0, "verified hypotheses", vp3_);
      }

      vis_->removeAllPointClouds();
      vis_->removeAllPointClouds(vp1_);
      vis_->removeAllPointClouds(vp2_);
      vis_->removeAllPointClouds(vp3_);

      Eigen::Vector4f zero_origin; zero_origin[0] = zero_origin[1] = zero_origin[2] = zero_origin[3] = 0.f;
      pcl::PointCloud<PointT>::Ptr vis_cloud (new pcl::PointCloud<PointT>);
      pcl::copyPointCloud(*pInputCloud_, *vis_cloud);
      vis_cloud->sensor_origin_ = zero_origin;
      vis_cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
      vis_->addPointCloud(vis_cloud, "input cloud", vp1_);
      vis_->setBackgroundColor(.0f, .0f, .0f, vp2_);

      for(size_t i=0; i<models_.size(); i++)
      {
          ModelT &m = *models_[i];
          const std::string model_id = m.id_.substr(0, m.id_.length() - 4);
          std::stringstream model_label;
          model_label << model_id << "_" << i;
          pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT>() );
          pcl::PointCloud<PointT>::ConstPtr model_cloud = m.getAssembled( 0.003f );
          pcl::transformPointCloud( *model_cloud, *model_aligned, transforms_[i]);
          vis_->addPointCloud(model_aligned, model_label.str(), vp2_);
          vis_->setBackgroundColor(.5f, .5f, .5f, vp2_);
      }

      for(size_t i=0; i<models_verified_.size(); i++)
      {
          ModelT &m = *models_verified_[i];
          const std::string model_id = m.id_.substr(0, m.id_.length() - 4);
          std::stringstream model_label;
          model_label << model_id << "_v_" << i;
          pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT>() );
          pcl::PointCloud<PointT>::ConstPtr model_cloud = m.getAssembled( 0.003f );
          pcl::transformPointCloud( *model_cloud, *model_aligned, transforms_verified_[i]);
          vis_->addPointCloud(model_aligned, model_label.str(), vp3_);
          vis_->setBackgroundColor(1.f, 1.f, 1.f, vp3_);
      }
      vis_->spin();
  }
}
