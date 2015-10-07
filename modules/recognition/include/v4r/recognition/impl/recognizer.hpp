#include <v4r/recognition/recognizer.h>
#include <v4r/segmentation/multiplane_segmentation.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

namespace v4r
{

template <typename PointT>
void
ObjectHypothesis<PointT>::visualize() const
{
    std::cerr << "This function is not implemented for this point cloud type!" << std::endl;
}

template <>
void
ObjectHypothesis<pcl::PointXYZRGB>::visualize() const
{
    if(!vis_)
    {
        vis_.reset(new pcl::visualization::PCLVisualizer("correspondences for hypothesis"));
    }

    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr model_cloud = model_->getAssembled( 0.003f );
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_aligned ( new pcl::PointCloud<pcl::PointXYZRGB>() );
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_vis ( new pcl::PointCloud<pcl::PointXYZRGB>() );
    Eigen::Vector4f zero_origin; zero_origin[0] = zero_origin[1] = zero_origin[2] = zero_origin[3] = 0.f;
    pcl::copyPointCloud( *scene_, *scene_vis);
    scene_vis->sensor_origin_ = zero_origin;
    scene_vis->sensor_orientation_ = Eigen::Quaternionf::Identity();
    pcl::copyPointCloud( *model_cloud, *model_aligned);
    vis_->addPointCloud(scene_vis, "scene");
    vis_->addPointCloud(model_aligned, "model_aligned");
    vis_->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (model_->keypoints_, model_->kp_normals_, 10, 0.05, "normals_model");

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr kp_colored_scene ( new pcl::PointCloud<pcl::PointXYZRGB>() );
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr kp_colored_model ( new pcl::PointCloud<pcl::PointXYZRGB>() );
    kp_colored_scene->points.resize(model_scene_corresp_->size());
    kp_colored_model->points.resize(model_scene_corresp_->size());
    for (size_t i=0; i<model_scene_corresp_->size(); i++)
    {
        const pcl::Correspondence &c = model_scene_corresp_->at(i);
        pcl::PointXYZRGB kp_m = model_->keypoints_->points[c.index_query];
        pcl::PointXYZRGB kp_s = scene_->points[c.index_match];

        const float r = kp_m.r = kp_s.r = 100 + rand() % 155;
        const float g = kp_m.g = kp_s.g = 100 + rand() % 155;
        const float b = kp_m.b = kp_s.b = 100 + rand() % 155;
        kp_colored_scene->points[i] = kp_s;
        kp_colored_model->points[i] = kp_m;

        std::stringstream ss; ss << "correspondence " << i;
        vis_->addLine(kp_s, kp_m, r/255, g/255, b/255, ss.str());
        vis_->addSphere(kp_s, 2, r/255, g/255, b/255, ss.str() + "kp_s", vp1_);
        vis_->addSphere(kp_m, 2, r/255, g/255, b/255, ss.str() + "kp_m", vp1_);
    }

    vis_->addPointCloud(kp_colored_scene, "kps_s");
    vis_->addPointCloud(kp_colored_model, "kps_m");

    vis_->spin();
}

template<typename PointT>
void
Recognizer<PointT>::hypothesisVerification ()
{
  std::vector<typename pcl::PointCloud<PointT>::ConstPtr> aligned_models (models_.size ());
  std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr> aligned_normals (models_.size ());
  models_verified_.clear();
  transforms_verified_.clear();

  if(!models_.size())
  {
      std::cout << "No models to verify, returning... " << std::endl;
      std::cout << "Cancelling service request." << std::endl;
      return;
  }

  for(size_t i=0; i<models_.size(); i++)
  {
      ConstPointTPtr model_cloud = models_[i]->getAssembled (hv_algorithm_->getResolution());
      typename pcl::PointCloud<PointT>::Ptr model_aligned (new pcl::PointCloud<PointT>);
      pcl::transformPointCloud (*model_cloud, *model_aligned, transforms_[i]);
      aligned_models[i] = model_aligned;

      pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud_const = models_[i]->getNormalsAssembled (hv_algorithm_->getResolution());
      pcl::PointCloud<pcl::Normal>::Ptr normal_cloud(new pcl::PointCloud<pcl::Normal>(*normal_cloud_const) );

      const Eigen::Matrix3f rot   = transforms_[i].block<3, 3> (0, 0);
      for(size_t jj=0; jj < normal_cloud->points.size(); jj++)
      {
          const pcl::Normal norm_pt = normal_cloud->points[jj];
          normal_cloud->points[jj].getNormalVector3fMap() = rot * norm_pt.getNormalVector3fMap();
      }
      aligned_normals[i] = normal_cloud;
  }

  typename pcl::PointCloud<PointT>::Ptr occlusion_cloud (new pcl::PointCloud<PointT>(*scene_));
  hv_algorithm_->setRequiresNormals(false);
  hv_algorithm_->setOcclusionCloud (occlusion_cloud);
  hv_algorithm_->setSceneCloud (scene_);
  hv_algorithm_->setNormalsForClutterTerm(scene_normals_);
  hv_algorithm_->addModels (aligned_models, true);
  hv_algorithm_->addNormalsClouds(aligned_normals);

  std::vector<bool> mask_hv;
  if (hv_algorithm_->getRequiresNormals ())
  {
    hv_algorithm_->addNormalsClouds (aligned_normals);
  }

  hv_algorithm_->addModels (aligned_models, true);

  std::vector<v4r::PlaneModel<PointT> > planes_found;
  if(hv_algorithm_->param_.add_planes_ && hv_algorithm_->add_planes_is_posssible())
  {
      //Multiplane segmentation
      v4r::MultiPlaneSegmentation<PointT> mps;
      mps.setInputCloud(scene_);
      mps.setMinPlaneInliers(1000);
      mps.setResolution(hv_algorithm_->getResolution());
      mps.setNormals(scene_normals_);
      mps.setMergePlanes(true);
      mps.segment();
      planes_found = mps.getModels();

      hv_algorithm_->addPlanarModels(planes_found);
      for(size_t kk=0; kk < planes_found.size(); kk++)
      {
          std::stringstream plane_id;
          plane_id << "plane_" << kk;
      }
  }

  hv_algorithm_->verify ();
  hv_algorithm_->getMask (mask_hv);

  std::vector<bool> mask_hv_with_planes;

  hv_algorithm_->getMask (mask_hv_with_planes);

  std::vector<int> coming_from (aligned_models.size() + planes_found.size());

  for(size_t j=0; j < aligned_models.size(); j++)
      coming_from[j] = 0;

  for(size_t j=0; j < planes_found.size(); j++)
      coming_from[aligned_models.size() + j] = 1;

  for (size_t j = 0; j < aligned_models.size (); j++)
  {
      mask_hv[j] = mask_hv_with_planes[j];

      if(mask_hv[j]) {
          models_verified_.push_back (models_[j]);
          transforms_verified_.push_back (transforms_[j]);
      }
  }

  for (size_t j = 0; j < planes_found.size(); j++)
  {
      if(mask_hv_with_planes[aligned_models.size () + j])
          verified_planes_.push_back(planes_found[j].plane_cloud_);
  }

}


template<typename PointT>
void
Recognizer<PointT>::poseRefinement()
{
  PointTPtr scene_voxelized (new pcl::PointCloud<PointT> ());
  pcl::VoxelGrid<PointT> voxel_grid_icp;
  voxel_grid_icp.setInputCloud (scene_);
  if(icp_scene_indices_ && icp_scene_indices_->indices.size() > 0)
  {
    voxel_grid_icp.setIndices(icp_scene_indices_);
  }
  voxel_grid_icp.setLeafSize (param_.voxel_size_icp_, param_.voxel_size_icp_, param_.voxel_size_icp_);
  voxel_grid_icp.filter (*scene_voxelized);

  switch (param_.icp_type_)
  {
    case 0:
    {
#pragma omp parallel for schedule(dynamic,1) num_threads(omp_get_num_procs())
      for (int i = 0; i < static_cast<int> (models_.size ()); i++)
      {
        ConstPointTPtr model_cloud;
        PointTPtr model_aligned (new pcl::PointCloud<PointT>);
        model_cloud = models_[i]->getAssembled (param_.voxel_size_icp_);
        pcl::transformPointCloud (*model_cloud, *model_aligned, transforms_[i]);

        typename pcl::registration::CorrespondenceRejectorSampleConsensus<PointT>::Ptr
                                rej (new pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> ());

        rej->setInputTarget (scene_voxelized);
        rej->setMaximumIterations (1000);
        rej->setInlierThreshold (0.005f);
        rej->setInputSource (model_aligned);

        pcl::IterativeClosestPoint<PointT, PointT> reg;
        reg.addCorrespondenceRejector (rej);
        reg.setInputTarget (scene_voxelized);
        reg.setInputSource (model_aligned);
        reg.setMaximumIterations (param_.icp_iterations_);
        reg.setMaxCorrespondenceDistance (param_.max_corr_distance_);

        typename pcl::PointCloud<PointT>::Ptr output_ (new pcl::PointCloud<PointT> ());
        reg.align (*output_);

        Eigen::Matrix4f icp_trans = reg.getFinalTransformation ();
        transforms_[i] = icp_trans * transforms_[i];
      }
    }
      break;
    default:
    {
      #pragma omp parallel for schedule(dynamic,1) num_threads(omp_get_num_procs())
      for (int i = 0; i < static_cast<int> (models_.size ()); i++)
      {
        typename VoxelBasedCorrespondenceEstimation<PointT, PointT>::Ptr
                    est (new VoxelBasedCorrespondenceEstimation<PointT, PointT> ());

        typename pcl::registration::CorrespondenceRejectorSampleConsensus<PointT>::Ptr
                    rej (new pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> ());

        Eigen::Matrix4f scene_to_model_trans = transforms_[i].inverse ();
        boost::shared_ptr<distance_field::PropagationDistanceField<PointT> > dt;
        models_[i]->getVGDT (dt);

        PointTPtr model_aligned (new pcl::PointCloud<PointT>);
        PointTPtr cloud_voxelized_icp_cropped (new pcl::PointCloud<PointT>);
        typename pcl::PointCloud<PointT>::ConstPtr cloud;
        dt->getInputCloud(cloud);
        model_aligned.reset(new pcl::PointCloud<PointT>(*cloud));

        pcl::transformPointCloud (*scene_voxelized, *cloud_voxelized_icp_cropped, scene_to_model_trans);

        PointT minPoint, maxPoint;
        pcl::getMinMax3D(*cloud, minPoint, maxPoint);
        minPoint.x -= param_.max_corr_distance_;
        minPoint.y -= param_.max_corr_distance_;
        minPoint.z -= param_.max_corr_distance_;

        maxPoint.x += param_.max_corr_distance_;
        maxPoint.y += param_.max_corr_distance_;
        maxPoint.z += param_.max_corr_distance_;

        pcl::CropBox<PointT> cropFilter;
        cropFilter.setInputCloud (cloud_voxelized_icp_cropped);
        cropFilter.setMin(minPoint.getVector4fMap());
        cropFilter.setMax(maxPoint.getVector4fMap());
        cropFilter.filter (*cloud_voxelized_icp_cropped);

        est->setVoxelRepresentationTarget (dt);
        est->setInputSource (cloud_voxelized_icp_cropped);
        est->setInputTarget (model_aligned);
        est->setMaxCorrespondenceDistance (param_.max_corr_distance_);

        rej->setInputTarget (model_aligned);
        rej->setMaximumIterations (1000);
        rej->setInlierThreshold (0.005f);
        rej->setInputSource (cloud_voxelized_icp_cropped);

        pcl::IterativeClosestPoint<PointT, PointT, float> reg;
        reg.setCorrespondenceEstimation (est);
        reg.addCorrespondenceRejector (rej);
        reg.setInputTarget (model_aligned); //model
        reg.setInputSource (cloud_voxelized_icp_cropped); //scene
        reg.setMaximumIterations (param_.icp_iterations_);
        reg.setEuclideanFitnessEpsilon(1e-5);
        reg.setTransformationEpsilon(0.001f * 0.001f);

        pcl::registration::DefaultConvergenceCriteria<float>::Ptr convergence_criteria;
        convergence_criteria = reg.getConvergeCriteria();
        convergence_criteria->setAbsoluteMSE(1e-12);
        convergence_criteria->setMaximumIterationsSimilarTransforms(15);
        convergence_criteria->setFailureAfterMaximumIterations(false);

        typename pcl::PointCloud<PointT>::Ptr output_ (new pcl::PointCloud<PointT> ());
        reg.align (*output_);

        Eigen::Matrix4f icp_trans;
        icp_trans = reg.getFinalTransformation () * scene_to_model_trans;
        transforms_[i] = icp_trans.inverse ();
      }
    }
  }
}

template<typename PointT>
void
Recognizer<PointT>::visualize() const
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
    typename pcl::PointCloud<PointT>::Ptr vis_cloud (new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*scene_, *vis_cloud);
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
        typename pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT>() );
        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m.getAssembled( 0.003f );
        pcl::transformPointCloud( *model_cloud, *model_aligned, transforms_[i]);
        vis_->addPointCloud(model_aligned, model_label.str(), vp2_);
    }
    vis_->setBackgroundColor(.5f, .5f, .5f, vp2_);

    for(size_t i=0; i<models_verified_.size(); i++)
    {
        ModelT &m = *models_verified_[i];
        const std::string model_id = m.id_.substr(0, m.id_.length() - 4);
        std::stringstream model_label;
        model_label << model_id << "_v_" << i;
        typename pcl::PointCloud<PointT>::Ptr model_aligned ( new pcl::PointCloud<PointT>() );
        typename pcl::PointCloud<PointT>::ConstPtr model_cloud = m.getAssembled( 0.003f );
        pcl::transformPointCloud( *model_cloud, *model_aligned, transforms_verified_[i]);
        vis_->addPointCloud(model_aligned, model_label.str(), vp3_);
    }

    visualizePlanes();
    vis_->setBackgroundColor(1.f, 1.f, 1.f, vp3_);
    vis_->spin();
}

template<typename PointT>
void
Recognizer<PointT>::visualizePlanes() const
{
    Eigen::Vector4f zero_origin; zero_origin[0] = zero_origin[1] = zero_origin[2] = zero_origin[3] = 0.f;
    for(size_t plane_id=0; plane_id < verified_planes_.size(); plane_id++)
    {
        std::stringstream plane_name;
        plane_name << "plane_" << plane_id;
        verified_planes_[plane_id]->sensor_origin_ = zero_origin;
        verified_planes_[plane_id]->sensor_orientation_ = Eigen::Quaternionf::Identity();
        vis_->addPointCloud<PointT> ( verified_planes_[plane_id], plane_name.str (), vp3_ );
    }
}

//template <>
//void
//Recognizer<pcl::PointXYZRGB>::visualizePlanes() const
//{
//    Eigen::Vector4f zero_origin; zero_origin[0] = zero_origin[1] = zero_origin[2] = zero_origin[3] = 0.f;
//    for(size_t plane_id=0; plane_id < verified_planes_.size(); plane_id++)
//    {
//        std::stringstream plane_name;
//        plane_name << "plane_" << plane_id;
//        verified_planes_[plane_id]->sensor_origin_ = zero_origin;
//        verified_planes_[plane_id]->sensor_orientation_ = Eigen::Quaternionf::Identity();
//        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_handler ( verified_planes_[plane_id] );
//        vis_->addPointCloud<pcl::PointXYZRGB> ( verified_planes_[plane_id], rgb_handler, plane_name.str (), vp3_ );
//    }
//}


}
