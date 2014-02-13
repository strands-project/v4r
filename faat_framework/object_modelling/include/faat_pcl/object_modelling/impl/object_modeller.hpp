/*
 * object_modeller.hpp
 *
 *  Created on: Mar 15, 2013
 *      Author: aitor
 */

#ifndef OBJECT_MODELLER_HPP_
#define OBJECT_MODELLER_HPP_

#include <faat_pcl/object_modelling/object_modeller.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/shot_local_estimator_omp.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/global/ourcvfh_estimator.h>
#include <faat_pcl/recognition/cg/graph_geometric_consistency.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/lum.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_estimation_normal_shooting.h>
#include <faat_pcl/object_modelling/impl/prim.hpp>
#include <faat_pcl/registration/visibility_reasoning.h>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/copy.hpp>
#include <numeric>
#include <pcl/surface/convex_hull.h>
#include <faat_pcl/registration/icp_with_gc.h>
#include <pcl/filters/passthrough.h>
#include <boost/filesystem.hpp>

namespace bf = boost::filesystem;

/*template<template<class > class Distance, typename PointT, typename PointTNormal>
  void
  faat_pcl::object_modelling::ObjectModeller<Distance, PointT, PointTNormal>::computeFeatures ()
  {

    boost::shared_ptr<faat_pcl::rec_3d_framework::UniformSamplingExtractor<PointT> >
                                                                                     uniform_keypoint_extractor (
                                                                                                                 new faat_pcl::rec_3d_framework::UniformSamplingExtractor<
                                                                                                                     PointT>);

    uniform_keypoint_extractor->setSamplingDensity (0.005f);
    uniform_keypoint_extractor->setFilterPlanar (true);
    boost::shared_ptr<faat_pcl::rec_3d_framework::KeypointExtractor<PointT> > keypoint_extractor;
    keypoint_extractor = boost::static_pointer_cast<faat_pcl::rec_3d_framework::KeypointExtractor<PointT> > (uniform_keypoint_extractor);

    boost::shared_ptr<faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointT, pcl::Normal> > normal_estimator;
    normal_estimator.reset (new faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointT, pcl::Normal>);
    normal_estimator->setCMR (false);
    normal_estimator->setDoVoxelGrid (true);
    normal_estimator->setRemoveOutliers (true);
    normal_estimator->setMinNRadius (27);
    normal_estimator->setValuesForCMRFalse (0.003f, 0.018f);

    boost::shared_ptr<faat_pcl::rec_3d_framework::SHOTLocalEstimationOMP<PointT, pcl::Histogram<352> > > estimator;
    estimator.reset (new faat_pcl::rec_3d_framework::SHOTLocalEstimationOMP<PointT, pcl::Histogram<352> >);
    estimator->setNormalEstimator (normal_estimator);
    estimator->addKeypointExtractor (keypoint_extractor);
    estimator->setSupportRadius (0.04);

    clouds_normals_.resize (clouds_.size ());
    clouds_normals_at_keypoints_.resize (clouds_.size ());
    processed_xyz_normals_clouds_.resize (clouds_.size ());

    keypoint_clouds_.resize (clouds_.size ());
    processed_clouds_.resize (clouds_.size ());
    clouds_signatures_.resize (clouds_.size ());
    clouds_flann_signatures_.resize (clouds_.size ());
    signatures_flann_data_.resize (clouds_.size ());
    flann_index_.resize (clouds_.size ());
    //clouds_codebooks_.resize(clouds_.size());

    boost::shared_ptr<faat_pcl::rec_3d_framework::OURCVFHEstimator<PointT, pcl::VFHSignature308 > > ourcvfh_estimator;
    ourcvfh_estimator.reset (new faat_pcl::rec_3d_framework::OURCVFHEstimator<PointT, pcl::VFHSignature308 >);
    ourcvfh_estimator->setNormalEstimator (normal_estimator);
    ourcvfh_estimator->setRefineClustersParam (2.5f);
    ourcvfh_estimator->setCVFHParams (0.15f, 0.025f, 2.f);
    ourcvfh_estimator->setNormalizeBins(true);
    ourcvfh_estimator->setAdaptativeMLS (false);

    for (size_t i = 0; i < clouds_.size (); i++)
    {
      PointCloudPtr processed (new pcl::PointCloud<PointT>);
      pcl::PointCloud<pcl::Histogram<352> >::Ptr signatures (new pcl::PointCloud<pcl::Histogram<352> > ());
      PointCloudPtr keypoints_pointcloud;

      bool success = estimator->estimate (clouds_[i], processed, keypoints_pointcloud, signatures);

      clouds_normals_[i].reset (new pcl::PointCloud<pcl::Normal>);
      clouds_normals_at_keypoints_[i].reset (new pcl::PointCloud<pcl::Normal>);

      pcl::PointCloud<pcl::Normal>::Ptr est_normals;
      estimator->getNormals (est_normals);

      pcl::copyPointCloud (*est_normals, *clouds_normals_[i]);

      std::vector<int> correct_indices;
      getIndicesFromCloud (processed, keypoints_pointcloud, correct_indices);
      pcl::copyPointCloud (*clouds_normals_[i], correct_indices, *clouds_normals_at_keypoints_[i]);

      keypoint_clouds_[i] = keypoints_pointcloud;
      processed_clouds_[i] = processed;
      clouds_signatures_[i] = signatures;

      int size_feat = sizeof(signatures->points[0].histogram) / sizeof(float);

      for (size_t s = 0; s < signatures->points.size (); s++)
      {
        flann_model descr_model;
        descr_model.keypoint_id = s;
        descr_model.descr.resize (size_feat);
        memcpy (&descr_model.descr[0], &signatures->points[s].histogram[0], size_feat * sizeof(float));
        clouds_flann_signatures_[i].push_back (descr_model);
      }

      //Build flann structure...
      convertToFLANN<flann_model> (clouds_flann_signatures_[i], signatures_flann_data_[i]);
      flann_index_[i] = new flann::Index<DistT> (signatures_flann_data_[i], flann::KDTreeIndexParams (4));
      flann_index_[i]->buildIndex ();

    }

    for (size_t i = 0; i < processed_xyz_normals_clouds_.size (); i++)
    {
      processed_xyz_normals_clouds_[i].reset (new pcl::PointCloud<pcl::PointNormal>);
      pcl::copyPointCloud (*processed_clouds_[i], *processed_xyz_normals_clouds_[i]);
      pcl::copyPointCloud (*clouds_normals_[i], *processed_xyz_normals_clouds_[i]);
    }

    pairwise_poses_.resize (processed_clouds_.size ());
    for (size_t i = 0; i < processed_clouds_.size (); i++)
      pairwise_poses_[i].resize (processed_clouds_.size ());

  }*/

template<template<class > class Distance, typename PointT, typename PointTNormal>
void
faat_pcl::object_modelling::ObjectModeller<Distance, PointT, PointTNormal>::processClouds ()
{

  boost::shared_ptr<faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointT, pcl::Normal> > normal_estimator;
  normal_estimator.reset (new faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointT, pcl::Normal>);
  normal_estimator->setCMR (false);
  normal_estimator->setDoVoxelGrid (false);
  normal_estimator->setRemoveOutliers (false);
  normal_estimator->setMinNRadius (27);
  normal_estimator->setValuesForCMRFalse (0.001f, 0.018f);

  clouds_normals_.resize (clouds_.size ());
  clouds_normals_at_keypoints_.resize (clouds_.size ());
  processed_xyz_normals_clouds_.resize (clouds_.size ());
  processed_clouds_.resize (clouds_.size ());

  for (size_t i = 0; i < clouds_.size (); i++)
  {
    PointCloudPtr processed (new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr est_normals(new pcl::PointCloud<pcl::Normal>);

    normal_estimator->estimate (clouds_[i], processed, est_normals);
    clouds_normals_[i].reset (new pcl::PointCloud<pcl::Normal>);
    pcl::copyPointCloud (*est_normals, *clouds_normals_[i]);
    processed_clouds_[i] = processed;
  }

  for (size_t i = 0; i < processed_xyz_normals_clouds_.size (); i++)
  {
    processed_xyz_normals_clouds_[i].reset (new pcl::PointCloud<PointTNormal>);
    pcl::copyPointCloud (*processed_clouds_[i], *processed_xyz_normals_clouds_[i]);
    pcl::copyPointCloud (*clouds_normals_[i], *processed_xyz_normals_clouds_[i]);
  }

  pairwise_poses_.resize (processed_clouds_.size ());
  for (size_t i = 0; i < processed_clouds_.size (); i++)
    pairwise_poses_[i].resize (processed_clouds_.size ());

  is_candidate_.resize(processed_clouds_.size());
  for(size_t i=0; i < processed_clouds_.size(); i++)
  {
    std::vector<bool> tt(processed_clouds_.size(), true);
    is_candidate_[i] = tt;
  }
}

/*template<template<class > class Distance, typename PointT, typename PointTNormal>
  void
  faat_pcl::object_modelling::ObjectModeller<Distance, PointT, PointTNormal>::computePairWisePosesBruteFroce ()
  {
    //Do geometric consistency to compute poses
    faat_pcl::GraphGeometricConsistencyGrouping<PointT, PointT> gcg_alg;
    gcg_alg.setGCThreshold (5);
    gcg_alg.setGCSize (0.01f);
    gcg_alg.setRansacThreshold (0.01f);
    gcg_alg.setUseGraph (true);

    std::vector<pcl::Correspondences> corresp_clusters;

    pairwise_corresp_clusters_.resize (processed_clouds_.size ());
    pairwise_registration_.resize (processed_clouds_.size ());
    for (size_t i = 0; i < processed_clouds_.size (); i++)
    {
      pairwise_corresp_clusters_[i].resize (processed_clouds_.size ());
      pairwise_registration_[i].resize (processed_clouds_.size ());

      for (size_t j = (i + 1); j < processed_clouds_.size (); j++)
      {
        gcg_alg.setSceneCloud (keypoint_clouds_[i]);
        gcg_alg.setInputCloud (keypoint_clouds_[j]);
        gcg_alg.setModelSceneCorrespondences (pairwise_correspondences_[i][j]);
        gcg_alg.setInputAndSceneNormals (clouds_normals_at_keypoints_[j], clouds_normals_at_keypoints_[i]);
        gcg_alg.cluster (corresp_clusters);

        if (corresp_clusters.size () > 0)
        {
          size_t max_cluster_id;
          size_t max_cluster_size = 0;
          for (size_t kk = 0; kk < corresp_clusters.size (); kk++)
          {
            if (corresp_clusters[kk].size () > max_cluster_size)
            {
              max_cluster_id = kk;
              max_cluster_size = corresp_clusters[kk].size ();
            }
          }

          if (use_max_cluster_only_)
          {
            Eigen::Matrix4f best_trans;
            typename pcl::registration::TransformationEstimationSVD<PointT, PointT> t_est;
            t_est.estimateRigidTransformation (*keypoint_clouds_[j], *keypoint_clouds_[i], corresp_clusters[max_cluster_id], best_trans);
            pairwise_poses_[i][j].push_back (best_trans);
            pairwise_corresp_clusters_[i][j].push_back (corresp_clusters[max_cluster_id]);

            if (refine_features_poses_)
            {
              if (use_gc_icp_)
                refineRelativePosesICPWithGC (i, j);
              else
                refineRelativePoses (i, j);
            }

            selectBestRelativePose (i, j);
            std::cout << "max correspondences (i,j,#,reg_quality, overlap, loop closure prob.):" << i << " " << j << " "
                << corresp_clusters[max_cluster_id].size () << " " << pairwise_registration_[i][j][0].reg_error_ << " "
                << pairwise_registration_[i][j][0].overlap_ << " " << pairwise_registration_[i][j][0].loop_closure_probability_ << std::endl;
          }
          else
          {
            std::cout << "Instances:" << corresp_clusters.size () << " Total correspondences:" << pairwise_correspondences_[i][j]->size ()
                << std::endl;

            float threshold_accept_model_hypothesis_ = 0.8f;
            std::vector<int> corresp_sizes;

            for (size_t kk = 0; kk < corresp_clusters.size (); kk++)
            {
              if (static_cast<float> ((corresp_clusters[kk]).size ()) < (threshold_accept_model_hypothesis_ * static_cast<float> (max_cluster_size)))
                continue;

              Eigen::Matrix4f best_trans;
              typename pcl::registration::TransformationEstimationSVD<PointT, PointT> t_est;
              t_est.estimateRigidTransformation (*keypoint_clouds_[j], *keypoint_clouds_[i], corresp_clusters[kk], best_trans);
              pairwise_poses_[i][j].push_back (best_trans);
              pairwise_corresp_clusters_[i][j].push_back (corresp_clusters[kk]);
              corresp_sizes.push_back (corresp_clusters[kk].size ());
            }

            std::cout << "Number of instances on which ICP will be done:" << pairwise_poses_[i][j].size () << std::endl;

            //which is the cluster that gives a better registration, should we decide this after ICP?
            selectBestRelativePose (i, j);

            size_t best = 0;
            float best_reg_error_ = (std::numeric_limits<float>::max () - 0.01f) * -1.f;
            for (size_t k = 0; k < pairwise_poses_[i][j].size (); k++)
            {
              float reg_error = (1.f - pairwise_registration_[i][j][k].fsv_fraction_) * corresp_sizes[k];
              //(pairwise_registration_[i][j][k].overlap_ / 10.f); // + (1.f / (pairwise_registration_[i][j][k].reg_error_));
              //+ (pairwise_registration_[i][j][k].loop_closure_probability_ * 0.05f);

              if (reg_error > best_reg_error_)
              {
                best = k;
                best_reg_error_ = reg_error;
              }
            }

            if (pairwise_poses_[i][j].size () > 0)
            {
              std::cout << pairwise_poses_[i][j].size () << " " << best << std::endl;
              pairwise_poses_[i][j][0] = pairwise_poses_[i][j][best];
              pairwise_corresp_clusters_[i][j][0] = pairwise_corresp_clusters_[i][j][best];
              pairwise_poses_[i][j].resize (1);
              pairwise_corresp_clusters_[i][j].resize (1);

              std::cout << "max correspondences (i,j,#,reg_quality, overlap, loop closure prob.):" << i << " " << j << " "
                  << corresp_clusters[max_cluster_id].size () << " " << pairwise_registration_[i][j][0].reg_error_ << " "
                  << pairwise_registration_[i][j][0].overlap_ << " " << pairwise_registration_[i][j][0].loop_closure_probability_ << std::endl;
            }

            if (refine_features_poses_)
            {
              if (use_gc_icp_)
                refineRelativePosesICPWithGC (i, j);
              else
                refineRelativePoses (i, j);
            }
          }
        }
      }
    }
  }*/

/*template<template<class > class Distance, typename PointT, typename PointTNormal>
  void
  faat_pcl::object_modelling::ObjectModeller<Distance, PointT, PointTNormal>::computePairWisePosesSelective ()
  {
    //Do geometric consistency to compute poses
    faat_pcl::GraphGeometricConsistencyGrouping<PointT, PointT> gcg_alg;
    gcg_alg.setGCThreshold (5);
    gcg_alg.setGCSize (0.01f);
    gcg_alg.setRansacThreshold (0.01f);
    gcg_alg.setUseGraph (true);

    //get view with most points
    size_t max_points = 0;
    size_t id_max_points_view = 0;
    for (size_t i = 0; i < processed_clouds_.size (); i++)
    {
      if (processed_clouds_[i]->points.size () > max_points)
      {
        max_points = processed_clouds_[i]->points.size ();
        id_max_points_view = i;
      }
    }

    std::cout << max_points << " " << id_max_points_view << std::endl;
    std::vector<size_t> reference;
    reference.push_back (id_max_points_view);
    std::deque<size_t> to_search;
    for (size_t i = 0; i < processed_clouds_.size (); i++)
    {
      if (i != id_max_points_view)
      {
        to_search.push_front (i);
      }
    }

    pairwise_corresp_clusters_.resize (processed_clouds_.size ());
    pairwise_registration_.resize (processed_clouds_.size ());

    Eigen::MatrixXf matched_vs = Eigen::MatrixXf::Zero (processed_clouds_.size (), processed_clouds_.size ());
    while (!to_search.empty ())
    {

      std::cout << "To search size:" << to_search.size () << std::endl;
      std::cout << "Reference size:" << reference.size () << std::endl;
      for (size_t i = 0; i < reference.size (); i++)
      {
        std::cout << reference[i] << " ";
      }

      std::cout << std::endl;

      //iterate over to_search and match against reference
      size_t kkk = 0;
      std::vector<size_t> keep_in_search;
      std::vector<size_t> add_to_reference;
      while (kkk < to_search.size ())
      {
        size_t ts = to_search[kkk];
        bool converged_to_reference = false;

        //match view against those in reference and mark the matrix
        for (size_t r = 0; r < reference.size (); r++)
        {

          if (matched_vs (ts, reference[r]) > 0.5)
          {
            continue;
          }

          int i, j;
          i = static_cast<int> (std::min (ts, reference[r]));
          j = static_cast<int> (std::max (ts, reference[r]));

          pairwise_corresp_clusters_[i].resize (processed_clouds_.size ());
          pairwise_registration_[i].resize (processed_clouds_.size ());

          gcg_alg.setSceneCloud (keypoint_clouds_[i]);
          gcg_alg.setInputCloud (keypoint_clouds_[j]);
          gcg_alg.setModelSceneCorrespondences (pairwise_correspondences_[i][j]);
          gcg_alg.setInputAndSceneNormals (clouds_normals_at_keypoints_[j], clouds_normals_at_keypoints_[i]);

          std::vector<pcl::Correspondences> corresp_clusters;
          gcg_alg.cluster (corresp_clusters);

          if (corresp_clusters.size () > 0)
          {
            size_t max_cluster_id;
            size_t max_cluster_size = 0;
            for (size_t kk = 0; kk < corresp_clusters.size (); kk++)
            {
              if (corresp_clusters[kk].size () > max_cluster_size)
              {
                max_cluster_id = kk;
                max_cluster_size = corresp_clusters[kk].size ();
              }
            }

            if (use_max_cluster_only_)
            {
              Eigen::Matrix4f best_trans;
              typename pcl::registration::TransformationEstimationSVD<PointT, PointT> t_est;
              t_est.estimateRigidTransformation (*keypoint_clouds_[j], *keypoint_clouds_[i], corresp_clusters[max_cluster_id], best_trans);
              pairwise_poses_[i][j].push_back (best_trans);
              pairwise_corresp_clusters_[i][j].push_back (corresp_clusters[max_cluster_id]);

              if (refine_features_poses_)
              {
                if (use_gc_icp_)
                  refineRelativePosesICPWithGC (i, j);
                else
                  refineRelativePoses (i, j);
              }

              selectBestRelativePose (i, j);
              std::cout << "max correspondences (i,j,#,reg_quality, overlap, loop closure prob.):" << i << " " << j << " "
                  << corresp_clusters[max_cluster_id].size () << " " << pairwise_registration_[i][j][0].reg_error_ << " "
                  << pairwise_registration_[i][j][0].overlap_ << " " << pairwise_registration_[i][j][0].loop_closure_probability_ << " "
                  << pairwise_registration_[i][j][0].fsv_fraction_ << std::endl;
            }
            else
            {
              std::cout << "Instances:" << corresp_clusters.size () << " Total correspondences:" << pairwise_correspondences_[i][j]->size ()
                  << std::endl;

              float threshold_accept_model_hypothesis_ = 0.8f;
              for (size_t kk = 0; kk < corresp_clusters.size (); kk++)
              {
                if (static_cast<float> ((corresp_clusters[kk]).size ())
                    < (threshold_accept_model_hypothesis_ * static_cast<float> (max_cluster_size)))
                  continue;

                Eigen::Matrix4f best_trans;
                typename pcl::registration::TransformationEstimationSVD<PointT, PointT> t_est;
                t_est.estimateRigidTransformation (*keypoint_clouds_[j], *keypoint_clouds_[i], corresp_clusters[kk], best_trans);
                pairwise_poses_[i][j].push_back (best_trans);
                pairwise_corresp_clusters_[i][j].push_back (corresp_clusters[kk]);
              }

              if (refine_features_poses_)
              {
                if (use_gc_icp_)
                  refineRelativePosesICPWithGC (i, j);
                else
                  refineRelativePoses (i, j);
              }

              //which is the cluster that gives a better registration, should we decide this after ICP?
              selectBestRelativePose (i, j);

              size_t best = 0;
              float best_reg_error_ = std::numeric_limits<float>::max ();
              for (size_t k = 0; k < pairwise_poses_[i][j].size (); k++)
              {
                float reg_error = pairwise_registration_[i][j][k].fsv_fraction_ + (1.f / (pairwise_registration_[i][j][k].reg_error_));
                //+ (pairwise_registration_[i][j][k].loop_closure_probability_ * 0.05f);

                if (reg_error < best_reg_error_)
                {
                  best = k;
                  best_reg_error_ = reg_error;
                }
              }

              if (pairwise_poses_[i][j].size () > 0)
              {
                std::cout << pairwise_poses_[i][j].size () << " " << best << std::endl;
                pairwise_poses_[i][j][0] = pairwise_poses_[i][j][best];
                pairwise_corresp_clusters_[i][j][0] = pairwise_corresp_clusters_[i][j][best];
                pairwise_poses_[i][j].resize (1);
                pairwise_corresp_clusters_[i][j].resize (1);

                std::cout << "max correspondences (i,j,#,reg_quality, overlap, loop closure prob.):" << i << " " << j << " "
                    << corresp_clusters[max_cluster_id].size () << " " << pairwise_registration_[i][j][0].reg_error_ << " "
                    << pairwise_registration_[i][j][0].overlap_ << " " << pairwise_registration_[i][j][0].loop_closure_probability_ << std::endl;
              }

            }
          }

          matched_vs (i, j) = 1.f;
          matched_vs (j, i) = 1.f;

          if (pairwise_registration_[i][j].size () > 0 && pairwise_registration_[i][j][0].overlap_ > 100
              && pairwise_registration_[i][j][0].reg_error_ > 0.8f && pcl_isfinite(pairwise_registration_[i][j][0].fsv_fraction_)
              && pairwise_registration_[i][j][0].fsv_fraction_ < 0.05f)
          {
            converged_to_reference = true;
          }
        }

        //if view ts converged to any of the reference views, do not add it to new_to_search and put it into reference
        if (converged_to_reference)
        {
          add_to_reference.push_back (ts);
        }
        else
        {
          keep_in_search.push_back (ts);
        }
        //to_search.pop_front ();
        //new_to_search.push_front
        kkk++;
      }

      for (size_t k = 0; k < add_to_reference.size (); k++)
        reference.push_back (add_to_reference[k]);

      to_search.clear ();
      for (size_t k = 0; k < keep_in_search.size (); k++)
        to_search.push_back (keep_in_search[k]);

      if (!to_search.empty ())
      {
        bool all_tried = true;
        for (size_t i = 0; i < processed_clouds_.size (); i++)
        {
          for (size_t j = (i + 1); j < processed_clouds_.size (); j++)
          {
            if (matched_vs (i, j) < 0.5)
            {
              all_tried = false;
            }
          }
        }

        if (all_tried)
          break;
      }

    }

//    for (size_t i = 0; i < processed_clouds_.size (); i++)
//     {
//     pairwise_corresp_clusters_[i].resize (processed_clouds_.size ());
//     pairwise_registration_[i].resize (processed_clouds_.size ());
//
//     for (size_t j = (i + 1); j < processed_clouds_.size (); j++)
//     {
//     gcg_alg.setSceneCloud (keypoint_clouds_[i]);
//     gcg_alg.setInputCloud (keypoint_clouds_[j]);
//     gcg_alg.setModelSceneCorrespondences (pairwise_correspondences_[i][j]);
//     gcg_alg.setInputAndSceneNormals (clouds_normals_at_keypoints_[j], clouds_normals_at_keypoints_[i]);
//     gcg_alg.cluster (corresp_clusters);
//     }
//     }
  }*/

/*template<template<class > class Distance, typename PointT, typename PointTNormal>
  void
  faat_pcl::object_modelling::ObjectModeller<Distance, PointT, PointTNormal>::computePairWiseRelativePosesWithFeatures ()
  {
    //Match features between different views (all against all) and create correspondence matrix
    //TODO: We could force here that correspondence should be reciprocal between i and j.

    flann::Matrix<int> indices;
    flann::Matrix<float> distances;
    int k = 1;
    distances = flann::Matrix<float> (new float[k], 1, k);
    indices = flann::Matrix<int> (new int[k], 1, k);
    int size_feat = sizeof(clouds_signatures_[0]->points[0].histogram) / sizeof(float);

    pairwise_correspondences_.resize (processed_clouds_.size ());
    for (size_t i = 0; i < processed_clouds_.size (); i++)
    {
      pairwise_correspondences_[i].resize (processed_clouds_.size ());
      for (size_t j = (i + 1); j < processed_clouds_.size (); j++)
      {
        //correspondences between i and j
        pairwise_correspondences_[i][j].reset (new pcl::Correspondences);
        for (size_t s = 0; s < clouds_signatures_[i]->points.size (); s++)
        {
          nearestKSearch (flann_index_[j], clouds_signatures_[i]->points[s].histogram, size_feat, k, indices, distances);
          std::vector<int> flann_models_indices;
          std::vector<int> flann_models_distances;
          for (size_t l = 0; l < k; l++)
          {
            flann_models_indices.push_back (indices[0][l]);
            flann_models_distances.push_back (distances[0][l]);
          }

          for (size_t ii = 0; ii < flann_models_indices.size (); ii++)
          {
            //correspondences between flann_models_indices[ii] belonging to view j and s belonging to view i
            pairwise_correspondences_[i][j]->push_back (pcl::Correspondence (flann_models_indices[ii], s, flann_models_distances[i]));
          }
        }
      }
    }

    if (range_images_.size () != processed_clouds_.size ())
    {
      //pw_reg.fsv_fraction_ = 1.f / pw_reg.reg_error_;
      PCL_ERROR("No range images provided..., WE SHOULD COMPUTE THEM!\n");
      float min_fl = std::numeric_limits<float>::max ();
      float max_fl = -1;
      int cx, cy;
      cx = cy = 150;
      for (size_t i = 0; i < clouds_.size (); i++)
      {
        faat_pcl::registration::VisibilityReasoning<PointT> vr (0, 0, 0);
        float fl = vr.computeFocalLength (cx, cy, clouds_[i]);
        std::cout << "Focal length:" << fl << std::endl;
        if (fl > max_fl)
          max_fl = fl;

        if (fl < min_fl)
          min_fl = fl;
      }

      std::cout << "Min Focal length:" << min_fl << std::endl;
      std::cout << "Max Focal length:" << max_fl << std::endl;

      for (size_t i = 0; i < clouds_.size (); i++)
      {
        faat_pcl::registration::VisibilityReasoning<PointT> vr (0, 0, 0);
        PointCloudPtr range_image;
        vr.computeRangeImage (cx, cy, min_fl, clouds_[i], range_image);
        range_images_.push_back (range_image);
      }

      focal_length_ = min_fl;
      cx_ = static_cast<float> (cx);
      cy_ = static_cast<float> (cy);
    }

    if (bf_pairwise_)
    {
      computePairWisePosesBruteFroce ();
    }
    else
    {
      computePairWisePosesSelective ();
    }
  }*/

/*template<template<class > class Distance, typename PointT, typename PointTNormal>
  void
  faat_pcl::object_modelling::ObjectModeller<Distance, PointT, PointTNormal>::selectBestRelativePose (int i, int j)
  {
    //build octree to compute inliers...
    pcl::octree::OctreePointCloudSearch<PointTNormal> octree (0.003);
    octree.setInputCloud (processed_xyz_normals_clouds_[i]);
    octree.addPointsFromInputCloud ();

    std::vector<int> pointIdxNKNSearch;
    std::vector<float> pointNKNSquaredDistance;

    float inliers_threshold_ = 0.01f;
    float color_sigma = 32.f;
    color_sigma *= color_sigma;
    pairwise_registration_[i][j].resize (pairwise_poses_[i][j].size ());
    for (size_t k = 0; k < pairwise_poses_[i][j].size (); k++)
    {

      pair_wise_registration pw_reg;
      pw_reg.reg_error_ = -std::numeric_limits<float>::quiet_NaN ();
      pw_reg.overlap_ = -std::numeric_limits<float>::quiet_NaN ();

      if (pairwise_poses_[i][j][k] == Eigen::Matrix4f::Zero ())
      {
        pairwise_registration_[i][j][k] = pw_reg;
        continue;
      }

      pw_reg.reg_error_ = 0.f;
      pw_reg.overlap_ = 0;

      typename pcl::PointCloud<PointTNormal>::Ptr pp_in (new pcl::PointCloud<PointTNormal>);
      pcl::transformPointCloudWithNormals (*processed_xyz_normals_clouds_[j], *pp_in, pairwise_poses_[i][j][k]);

      std::vector<float> color_weights;

      for (size_t kk = 0; kk < pp_in->points.size (); kk++)
      {
        if (octree.nearestKSearch (pp_in->points[kk], 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
          float d = sqrt (pointNKNSquaredDistance[0]);

          if (d < inliers_threshold_)
          {
            float d_weight = -(d * d / (inliers_threshold_)) + 1;
            Eigen::Vector3f scene_p_normal = processed_xyz_normals_clouds_[i]->points[pointIdxNKNSearch[0]].getNormalVector3fMap ();
            Eigen::Vector3f model_p_normal = pp_in->points[kk].getNormalVector3fMap ();
            scene_p_normal.normalize ();
            model_p_normal.normalize ();
            float dotp = scene_p_normal.dot (model_p_normal);

            //using color
            float rgb_i, rgb_j;
            bool exists_i;
            bool exists_j;

            typedef pcl::PointCloud<PointT> CloudM;
            typedef typename pcl::traits::fieldList<typename CloudM::PointType>::type FieldListM;

            pcl::for_each_type<FieldListM> (
                                            pcl::CopyIfFieldExists<typename CloudM::PointType, float> (
                                                                                                       processed_clouds_[i]->points[pointIdxNKNSearch[0]],
                                                                                                       "rgb", exists_i, rgb_i));

            pcl::for_each_type<FieldListM> (
                                            pcl::CopyIfFieldExists<typename CloudM::PointType, float> (processed_clouds_[j]->points[kk], "rgb",
                                                                                                       exists_j, rgb_j));

            float color_weight = 1.f;
            if (exists_i && exists_j)
            {
              uint32_t rgb = *reinterpret_cast<int*> (&rgb_i);
              uint8_t ri = (rgb >> 16) & 0x0000ff;
              uint8_t gi = (rgb >> 8) & 0x0000ff;
              uint8_t bi = (rgb) & 0x0000ff;

              rgb = *reinterpret_cast<int*> (&rgb_j);
              uint8_t rj = (rgb >> 16) & 0x0000ff;
              uint8_t gj = (rgb >> 8) & 0x0000ff;
              uint8_t bj = (rgb) & 0x0000ff;

              color_weight = std::exp ((-0.5f * (ri - rj) * (ri - rj)) / (color_sigma));
              color_weight *= std::exp ((-0.5f * (gi - gj) * (gi - gj)) / (color_sigma));
              color_weight *= std::exp ((-0.5f * (bi - bj) * (bi - bj)) / (color_sigma));
              color_weights.push_back (color_weight);
            }

            //pw_reg.reg_error_ += d_weight * dotp * color_weight;
            pw_reg.reg_error_ += d_weight; // * dotp; // * color_weight;
            pw_reg.overlap_++;
          }
        }
      }

      pw_reg.reg_error_ /= pw_reg.overlap_;
      if (color_weights.size () > 0)
      {
        std::sort (color_weights.begin (), color_weights.end ());
        float median = color_weights[color_weights.size () / 2];
        float mean = std::accumulate (color_weights.begin (), color_weights.end (), 0.f) / static_cast<float> (color_weights.size ());
        pw_reg.reg_error_ *= mean;
      }

      //compute visibility consistency measures...
      //construct depth images or take them from range_images if available
      //compute differences whenever the two range_images are defined...
      //do it first from Ci and then from Cj
      if (range_images_.size () == processed_clouds_.size ())
      {
        faat_pcl::registration::VisibilityReasoning<PointT> vr (focal_length_, cx_, cy_);
        float fsv_ij = vr.computeFSV (range_images_[i], clouds_[j], pairwise_poses_[i][j][k]);
        float fsv_ji = vr.computeFSV (range_images_[j], clouds_[i], pairwise_poses_[i][j][k].inverse ());
        pw_reg.fsv_fraction_ = std::max (fsv_ij, fsv_ji);
        //std::cout << "FSV fraction:" << pw_reg.fsv_fraction_ << std::endl;
      }
      else
      {
        //pw_reg.fsv_fraction_ = 1.f / pw_reg.reg_error_;
        PCL_ERROR("No range images provided...\n");
      }

      //loop closure probability...
      assert(pw_reg.overlap_ <= std::max(static_cast<int>(processed_clouds_[i]->points.size()), static_cast<int>(processed_clouds_[j]->points.size())));
      pw_reg.loop_closure_probability_ = pw_reg.overlap_ / std::max (static_cast<float> (processed_clouds_[i]->points.size ()),
                                                                     static_cast<float> (processed_clouds_[j]->points.size ()));
      //* (pw_reg.overlap_ / static_cast<float>(processed_clouds_[j]->points.size()));

      pairwise_registration_[i][j][k] = pw_reg;
    }
  }*/

template<template<class > class Distance, typename PointT, typename PointTNormal>
  void
  faat_pcl::object_modelling::ObjectModeller<Distance, PointT, PointTNormal>::refineRelativePosesICPWithGC (int i, int j)
  {
    typedef PointTNormal PointTInternal;

    typename pcl::registration::TransformationEstimationPointToPlaneLLS<PointTInternal, PointTInternal>::Ptr
                                                                                                             trans_lls (
                                                                                                                        new pcl::registration::TransformationEstimationPointToPlaneLLS<
                                                                                                                            PointTInternal,
                                                                                                                            PointTInternal>);

    typename pcl::registration::CorrespondenceRejectorSampleConsensus<PointTInternal>::Ptr
                                                                                           rej (
                                                                                                new pcl::registration::CorrespondenceRejectorSampleConsensus<
                                                                                                    PointTInternal> ());

    rej->setMaximumIterations (1000);
    rej->setInlierThreshold (0.01f);

    typename pcl::registration::CorrespondenceEstimationNormalShooting<PointTInternal, PointTInternal, PointTInternal>::Ptr
                                                                                                                   cens (
                                                                                                                         new pcl::registration::CorrespondenceEstimationNormalShooting<
                                                                                                                             PointTInternal,
                                                                                                                             PointTInternal,
                                                                                                                             PointTInternal>);

    float max_corresp_dist = 0.25f;
    faat_pcl::IterativeClosestPointWithGC<PointTInternal, PointTInternal> icp;
    icp.setTransformationEpsilon (0.000001 * 0.000001);
    icp.setMinNumCorrespondences (3);
    icp.setMaxCorrespondenceDistance (max_corresp_dist);
    icp.setUseCG (true);
    icp.setSurvivalOfTheFittest (icp_with_gc_survival_of_the_fittest_);
    icp.setMaximumIterations(15);
    icp.setOverlapPercentage(ov_percentage_);
    icp.setVisFinal(VIS_FINAL_GC_ICP_);
    icp.setDtVxSize(dt_vx_size_);

    //typedef PointT RangeImageT;
    icp.template setRangeImages<PointT>(range_images_[j], range_images_[i], focal_length_, cx_, cy_);
    pcl::registration::DefaultConvergenceCriteria<float>::Ptr convergence_criteria;
    convergence_criteria = icp.getConvergeCriteria ();
    convergence_criteria->setAbsoluteMSE (1e-9);
    convergence_criteria->setMaximumIterationsSimilarTransforms (15);
    convergence_criteria->setFailureAfterMaximumIterations (false);

    icp.setInputTarget (processed_xyz_normals_clouds_[i]);
    rej->setInputTarget (processed_xyz_normals_clouds_[i]);
    cens->setInputTarget (processed_xyz_normals_clouds_[i]);

    typename pcl::PointCloud<PointTInternal> pp_out;
    typename pcl::PointCloud<PointTInternal>::Ptr pp_in (new pcl::PointCloud<PointTInternal>);
    pcl::transformPointCloudWithNormals (*processed_xyz_normals_clouds_[j], *pp_in, pairwise_poses_[i][j][0]);

    icp.setInputSource (pp_in);
    rej->setInputSource (pp_in);
    cens->setSourceNormals (pp_in);
    cens->setInputSource (pp_in);
    icp.align (pp_out);
    std::vector<std::pair<float, Eigen::Matrix4f> > res;
    icp.getResults(res);

    pairwise_poses_[i][j].resize(res.size());
    //TODO: Can filter according to res[k].first, for now keep all
    for (size_t k = 0; k < res.size (); k++)
    {
      pairwise_poses_[i][j][k] = res[k].second;
    }
  }

template<template<class > class Distance, typename PointT, typename PointTNormal>
void
faat_pcl::object_modelling::ObjectModeller<Distance, PointT, PointTNormal>::saveGraph(std::string & directory)
{
  bf::path dir = directory;
  if(!bf::exists(dir))
  {
    bf::create_directory(dir);
    for(size_t i=0; i < clouds_.size(); i++)
    {
      std::stringstream save_to;
      save_to << directory << "/" << i << ".pcd";
      pcl::io::savePCDFileBinary(save_to.str(), *clouds_[i]);

      std::stringstream save_to_range_image;
      save_to_range_image << directory << "/range_image_" << i << ".pcd";
      pcl::io::savePCDFileBinary(save_to_range_image.str(), *range_images_[i]);
    }

    std::stringstream graph_file;
    graph_file << directory << "/graph.txt";
    std::ofstream myfile;
    myfile.open (graph_file.str().c_str());
    for (size_t i = 0; i < processed_xyz_normals_clouds_.size (); i++)
    {
      for (size_t j = (i + 1); j < processed_xyz_normals_clouds_.size (); j++)
      {
        std::cout << "View " << i << " " << j << std::endl;
        if(pairwise_poses_[i][j].size() > 0)
        {
          myfile << i << " " << j << " " << pairwise_poses_[i][j].size() << std::endl;
          for(size_t k=0; k < pairwise_poses_[i][j].size(); k++)
          {
            //write pose matrix
            for(size_t ii=0; ii < 4; ii++)
            {
              for(size_t jj=0; jj < 4; jj++)
                myfile << pairwise_poses_[i][j][k](ii,jj) << " ";
            }

            myfile << std::endl;
          }
        }
      }
    }

    myfile.close();
  }
}

template<template<class > class Distance, typename PointT, typename PointTNormal>
  void
  faat_pcl::object_modelling::ObjectModeller<Distance, PointT, PointTNormal>::refineRelativePoses (int i, int j)
  {
    typedef PointTNormal PointTInternal;

    typename pcl::registration::TransformationEstimationPointToPlaneLLS<PointTInternal, PointTInternal>::Ptr
                                                                                                             trans_lls (
                                                                                                                        new pcl::registration::TransformationEstimationPointToPlaneLLS<
                                                                                                                            PointTInternal,
                                                                                                                            PointTInternal>);

    typename pcl::registration::CorrespondenceRejectorSampleConsensus<PointTInternal>::Ptr
                                                                                           rej (
                                                                                                new pcl::registration::CorrespondenceRejectorSampleConsensus<
                                                                                                    PointTInternal> ());

    rej->setMaximumIterations (1000);
    rej->setInlierThreshold (0.01f);

    typename pcl::registration::CorrespondenceEstimationNormalShooting<PointTInternal, PointTInternal, PointTInternal>::Ptr
                                                                                                                   cens (
                                                                                                                         new pcl::registration::CorrespondenceEstimationNormalShooting<
                                                                                                                             PointTInternal,
                                                                                                                             PointTInternal,
                                                                                                                             PointTInternal>);

    typename pcl::IterativeClosestPoint<PointTInternal, PointTInternal> icp;
    icp.setMaximumIterations (30);
    icp.setMaxCorrespondenceDistance (0.01);
    //icp.setTransformationEstimation (trans_lls);
    icp.setTransformationEpsilon (0.001 * 0.001);
    icp.setUseReciprocalCorrespondences (false);
    //icp.addCorrespondenceRejector (rej);
    //icp.setCorrespondenceEstimation (cens);

    pcl::registration::DefaultConvergenceCriteria<float>::Ptr convergence_criteria;
    convergence_criteria = icp.getConvergeCriteria ();
    convergence_criteria->setAbsoluteMSE (1e-9);
    convergence_criteria->setMaximumIterationsSimilarTransforms (5);
    convergence_criteria->setFailureAfterMaximumIterations (false);

    icp.setInputTarget (processed_xyz_normals_clouds_[i]);
    rej->setInputTarget (processed_xyz_normals_clouds_[i]);
    cens->setInputTarget (processed_xyz_normals_clouds_[i]);

    for (size_t k = 0; k < pairwise_poses_[i][j].size (); k++)
    {
      typename pcl::PointCloud<PointTInternal> pp_out;
      typename pcl::PointCloud<PointTInternal>::Ptr pp_in (new pcl::PointCloud<PointTInternal>);
      pcl::transformPointCloudWithNormals (*processed_xyz_normals_clouds_[j], *pp_in, pairwise_poses_[i][j][k]);

      icp.setInputSource (pp_in);
      rej->setInputSource (pp_in);
      cens->setSourceNormals (pp_in);
      cens->setInputSource (pp_in);
      icp.align (pp_out);

      pcl::registration::DefaultConvergenceCriteria<float>::ConvergenceState conv_state;
      conv_state = convergence_criteria->getConvergenceState ();

      if (conv_state != pcl::registration::DefaultConvergenceCriteria<float>::CONVERGENCE_CRITERIA_ITERATIONS && conv_state
          != pcl::registration::DefaultConvergenceCriteria<float>::CONVERGENCE_CRITERIA_NOT_CONVERGED && conv_state
          != pcl::registration::DefaultConvergenceCriteria<float>::CONVERGENCE_CRITERIA_NO_CORRESPONDENCES)
      {
        //PCL_INFO("icp converged... %d %d transform: %d conv state: %d\n", i, j, k, conv_state);
        pairwise_poses_[i][j][k] = icp.getFinalTransformation () * pairwise_poses_[i][j][k];
      }
      else
      {
        //PCL_WARN("icp did not converge... %d %d transform: %d conv state: %d\n", i, j, k, conv_state);
        //pairwise_poses_[i][j][k] = Eigen::Matrix4f::Zero();
        pairwise_poses_[i][j][k] = icp.getFinalTransformation () * pairwise_poses_[i][j][k];
      }
    }
  }

template<template<class > class Distance, typename PointT, typename PointTNormal>
  void
  faat_pcl::object_modelling::ObjectModeller<Distance, PointT, PointTNormal>::computeRelativePosesWithICP ()
  {

    if (range_images_.size () != processed_clouds_.size ())
    {
      //pw_reg.fsv_fraction_ = 1.f / pw_reg.reg_error_;
      PCL_ERROR("No range images provided..., WE SHOULD COMPUTE THEM!\n");
      float min_fl = std::numeric_limits<float>::max ();
      float max_fl = -1;
      int cx, cy;
      cx = cy = 150;
      for (size_t i = 0; i < clouds_.size (); i++)
      {
        faat_pcl::registration::VisibilityReasoning<PointT> vr (0, 0, 0);
        float fl = vr.computeFocalLength (cx, cy, clouds_[i]);
        std::cout << "Focal length:" << fl << std::endl;
        if (fl > max_fl)
          max_fl = fl;

        if (fl < min_fl)
          min_fl = fl;
      }

      std::cout << "Min Focal length:" << min_fl << std::endl;
      std::cout << "Max Focal length:" << max_fl << std::endl;

      for (size_t i = 0; i < clouds_.size (); i++)
      {
        faat_pcl::registration::VisibilityReasoning<PointT> vr (0, 0, 0);
        PointCloudPtr range_image;
        vr.computeRangeImage (cx, cy, min_fl, clouds_[i], range_image);
        range_images_.push_back (range_image);
      }

      focal_length_ = min_fl;
      cx_ = static_cast<float> (cx);
      cy_ = static_cast<float> (cy);

      /*pcl::visualization::PCLVisualizer vis("range images...");
       for (size_t i = 0; i < range_images_.size (); i++)
       {
       vis.addPointCloud(range_images_[i]);
       vis.spin();
       vis.removeAllPointClouds();
       }*/
    }

    pairwise_registration_.resize (processed_clouds_.size ());
    for (size_t i = 0; i < processed_xyz_normals_clouds_.size (); i++)
    {
      pairwise_registration_[i].resize (processed_clouds_.size ());
      for (size_t j = (i + 1); j < processed_xyz_normals_clouds_.size (); j++)
      {
        pairwise_poses_[i][j].push_back (Eigen::Matrix4f::Identity ());

        if(is_candidate_[i][j])
        {
          std::cout << "View " << i << " " << j << std::endl;

          if (use_gc_icp_)
            refineRelativePosesICPWithGC (i, j);
          else
            refineRelativePoses (i, j);

          //refineRelativePosesICPWithGC (i, j);
          //selectBestRelativePose (i, j);
        }
      }
    }
  }

/*template<template<class > class Distance, typename PointT, typename PointTNormal>
  void
  faat_pcl::object_modelling::ObjectModeller<Distance, PointT, PointTNormal>::computeAbsolutePosesRecursive (
                                                                                               DirectedGraph & mst,
                                                                                               boost::graph_traits<DirectedGraph>::vertex_descriptor & start,
                                                                                               Eigen::Matrix4f accum)

  {
    if (boost::out_degree (start, mst) == 0)
      return;

    std::cout << "out degree:" << boost::out_degree (start, mst) << " vertex:" << start << std::endl;

    //iterate over edges and call recursive
    //TODO: Sort this edges if out_degree > 1
    boost::graph_traits<DirectedGraph>::out_edge_iterator ei, ei_end;
    std::vector<std::pair<boost::graph_traits<DirectedGraph>::out_edge_iterator, float> > sorted_edges;
    boost::property_map<DirectedGraph, boost::edge_weight_t>::type EdgeWeightMap = get (boost::edge_weight_t (), mst);

    for (boost::tie (ei, ei_end) = boost::out_edges (start, mst); ei != ei_end; ++ei)
    {
      std::pair<boost::graph_traits<DirectedGraph>::out_edge_iterator, float> p;
      p.first = ei;
      p.second = static_cast<float> (get (EdgeWeightMap, *ei));
      sorted_edges.push_back (p);
    }

    std::sort (
               sorted_edges.begin (),
               sorted_edges.end (),
               boost::bind (&std::pair<boost::graph_traits<DirectedGraph>::out_edge_iterator, float>::second, _1)
                   < boost::bind (&std::pair<boost::graph_traits<DirectedGraph>::out_edge_iterator, float>::second, _2));

    //for (boost::tie(ei, ei_end) = boost::out_edges(start, mst); ei != ei_end; ++ei)
    for (size_t i = 0; i < sorted_edges.size (); i++)
    {
      Eigen::Matrix4f internal_accum;
      boost::graph_traits<DirectedGraph>::vertex_descriptor dst = target (*(sorted_edges[i].first), mst);
      std::cout << start << "-" << dst << std::endl;
      if (dst < start)
      {
        //std::cout << "(Inverse...) Number of correspondences:" << pairwise_corresp_clusters_[dst][start][0].size() << std::endl;
        internal_accum = accum * pairwise_poses_[dst][start][0].inverse ();
        absolute_poses_[dst] = internal_accum;
      }
      else
      {
        //std::cout << "Number of correspondences:" << pairwise_corresp_clusters_[start][dst][0].size() << std::endl;
        internal_accum = accum * pairwise_poses_[start][dst][0];
        absolute_poses_[dst] = internal_accum;
      }

      computeAbsolutePosesRecursive (mst, dst, internal_accum);
    }
  }

template<template<class > class Distance, typename PointT, typename PointTNormal>
  void
  faat_pcl::object_modelling::ObjectModeller<Distance, PointT, PointTNormal>::computeAbsolutePosesFromMST (
                                                                                             DirectedGraph & mst,
                                                                                             boost::graph_traits<DirectedGraph>::vertex_descriptor & root)
  {

    absolute_poses_.clear ();
    absolute_poses_.resize (processed_clouds_.size ());
    absolute_poses_[root] = Eigen::Matrix4f::Identity ();

    std::cout << "root vertex:" << root << std::endl;
    std::cout << absolute_poses_[root] << std::endl;

    computeAbsolutePosesRecursive (mst, root, absolute_poses_[root]);
  }*/

/*template<template<class > class Distance, typename PointT, typename PointTNormal>
  bool
  faat_pcl::object_modelling::ObjectModeller<Distance, PointT, PointTNormal>::graphGloballyConsistent ()
  {

  }

template<template<class > class Distance, typename PointT, typename PointTNormal>
  void
  faat_pcl::object_modelling::ObjectModeller<Distance, PointT, PointTNormal>::globalSelectionWithMST (bool refine_global_pose)
  {
    const int num_nodes = (static_cast<int> (processed_clouds_.size ()));

    Graph G (num_nodes);
    for (size_t i = 0; i < processed_clouds_.size (); i++)
    {
      for (size_t j = (i + 1); j < processed_clouds_.size (); j++)
      {
        std::cout << "Number of poses..." << pairwise_poses_[i][j].size () << " ... (" << i << "," << j << ")" << std::endl;

        if (pairwise_poses_[i][j].size () == 0 || pairwise_registration_[i][j].size() == 0)
          continue;

        size_t best = 0;
        float best_reg_error_ = 0;
        std::cout << pairwise_poses_[i][j].size () << " " << pairwise_registration_[i][j].size() << std::endl;

        for (size_t k = 0; k < pairwise_poses_[i][j].size (); k++)
        {
          if (pairwise_registration_[i][j][k].reg_error_ > best_reg_error_)
          {
            best = k;
            best_reg_error_ = pairwise_registration_[i][j][k].reg_error_;
          }
        }

        size_t kk = best;
        if (pairwise_registration_[i][j][kk].overlap_ <= 200 || pairwise_registration_[i][j][kk].reg_error_ < 0.f
            || !pcl_isfinite(pairwise_registration_[i][j][kk].fsv_fraction_))
        {
          //std::cout << "This matrix should be Zero" << std::endl << pairwise_poses_[i][j][kk]  << std::endl;
        }
        else
        {
          //EdgeWeightProperty w = 1.f / (pairwise_registration_[i][j][kk].overlap_ * pairwise_registration_[i][j][kk].reg_error_);
          //EdgeWeightProperty w = 1.f / (pairwise_registration_[i][j][kk].reg_error_);
          //EdgeWeightProperty w = 1.f / pairwise_corresp_clusters_[i][j][kk].size();
          EdgeWeightProperty w = pairwise_registration_[i][j][kk].fsv_fraction_; // + (1.f / (pairwise_registration_[i][j][kk].reg_error_));
          //+ (pairwise_registration_[i][j][kk].loop_closure_probability_ * 0.05f);

          Eigen::Matrix4f tmp = pairwise_poses_[i][j][0];
          pairwise_poses_[i][j][0] = pairwise_poses_[i][j][kk];
          pairwise_poses_[i][j][kk] = tmp;

          add_edge (static_cast<int> (i), static_cast<int> (j), w, G);
        }
      }
    }

    boost::vector_property_map<int> components (num_vertices (G));
    int n_cc = static_cast<int> (boost::connected_components (G, &components[0]));
    std::cout << "Number of connected components..." << n_cc << std::endl;

    std::vector<graph_traits<Graph>::vertex_descriptor> p (num_vertices (G));
    prim_minimum_spanning_tree (G, &p[0]);

    DirectedGraph MST (num_vertices (G));
    graph_traits<Graph>::vertex_descriptor root = p[0];

    boost::property_map<Graph, boost::edge_weight_t>::type EdgeWeightMap = get (boost::edge_weight_t (), G);

    for (size_t i = 0; i < p.size (); i++)
    {
      if (p[i] == i)
      {
        root = p[i];
      }
      else
      {
        EdgeWeightProperty w = get (EdgeWeightMap, boost::edge (std::min (p[i], i), std::max (p[i], i), G).first);
        boost::add_edge (p[i], i, w, MST);
      }
    }

    computeAbsolutePosesFromMST (MST, root);

    if (refine_global_pose)
      refinePosesGlobally (MST, root);

    boost::copy_graph (MST, ST_graph_);
    boost::copy_graph (G, LR_graph_);
  }*/

template<template<class > class Distance, typename PointT, typename PointTNormal>
  void
  faat_pcl::object_modelling::ObjectModeller<Distance, PointT, PointTNormal>::getCorrespondences (PointCloudPtr i, PointCloudPtr j,
                                                                                    pcl::CorrespondencesPtr & corresp_i_j)
  {
    pcl::octree::OctreePointCloudSearch<PointT> octree (0.003);
    octree.setInputCloud (j);
    octree.addPointsFromInputCloud ();

    std::vector<int> pointIdxNKNSearch;
    std::vector<float> pointNKNSquaredDistance;

    for (size_t jj = 0; jj < i->points.size (); jj++)
    {
      if (octree.nearestKSearch (i->points[jj], 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
      {
        pcl::Correspondence corr;
        corr.index_query = jj;
        corr.index_match = pointIdxNKNSearch[0];
        corr.distance = sqrt (pointNKNSquaredDistance[0]);

        if (corr.distance < 0.02f)
          corresp_i_j->push_back (corr);
      }
    }
  }

template<template<class > class Distance, typename PointT, typename PointTNormal>
  void
  faat_pcl::object_modelling::ObjectModeller<Distance, PointT, PointTNormal>::refinePosesGlobally (DirectedGraph & mst,
                                                                                     boost::graph_traits<DirectedGraph>::vertex_descriptor & root)
  {
    //add all point clouds in the absolute coordinate system
    //add correspondences between transformed point clouds only between those clouds that are connected in the MST
    pcl::registration::LUM<PointT> lum;
    boost::graph_traits<DirectedGraph>::vertex_descriptor current = root;

    for (size_t i = 0; i < processed_clouds_.size (); i++)
    {

      PointCloudPtr no_nans (new pcl::PointCloud<PointT>);
      pcl::PassThrough<PointT> pass_;
      pass_.setFilterLimits (-100.f, 100.f);
      pass_.setFilterFieldName ("z");
      pass_.setInputCloud (clouds_[i]);
      pass_.filter (*no_nans);

      std::stringstream cloud_name;
      cloud_name << "cloud_" << i << ".pcd";

      PointCloudPtr pp (new pcl::PointCloud<PointT>);
      pcl::transformPointCloud (*no_nans, *pp, absolute_poses_[i]);

      pcl::io::savePCDFileBinary(cloud_name.str(), *pp);

      lum.addPointCloud (pp);
    }

    for (size_t i = 0; i < processed_clouds_.size (); i++)
    {
      for (size_t j = (i + 1); j < processed_clouds_.size (); j++)
      {
        if(pairwise_poses_[i][j].size() > 0)
        {
          pcl::CorrespondencesPtr corresp_i_j (new pcl::Correspondences);
          getCorrespondences (lum.getPointCloud (static_cast<int> (i)), lum.getPointCloud (static_cast<int> (j)), corresp_i_j);
          lum.setCorrespondences (static_cast<int> (i), static_cast<int> (j), corresp_i_j);
        }
      }
    }

    lum.setMaxIterations (200);
    lum.setConvergenceThreshold (0.0);
    lum.compute ();

    PointCloudPtr cloud_out (new pcl::PointCloud<PointT>);
    cloud_out = lum.getConcatenatedCloud ();

    pcl::visualization::PCLVisualizer vis ("lum refined");
    //pcl::visualization::PointCloudColorHandlerRGBField<PointT> handler_rgb (cloud_out);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> handler_rgb (cloud_out, 0, 255, 0);
    vis.addPointCloud<PointT> (cloud_out, handler_rgb, "refined pose");
    vis.spin ();
  }

template<template<class > class Distance, typename PointT, typename PointTNormal>
  void
  faat_pcl::object_modelling::ObjectModeller<Distance, PointT, PointTNormal>::visualizePairWiseAlignment ()
  {
    pcl::visualization::PCLVisualizer vis ("");
    for (size_t i = 0; i < processed_clouds_.size (); i++)
    {
      std::stringstream cloud_name;
      cloud_name << "cloud_" << i;

      pcl::visualization::PointCloudColorHandlerCustom<PointT> handler_rgb (processed_clouds_[i], 0, 0, 255);
      vis.addPointCloud<PointT> (processed_clouds_[i], handler_rgb, cloud_name.str ());

      for (size_t j = (i + 1); j < processed_clouds_.size (); j++)
      {
        std::stringstream cloud_name;
        cloud_name << "cloud_" << j;

        std::cout << "Number of transformations for " << i << " to " << j << ": " << pairwise_poses_[i][j].size () << std::endl;
        for (size_t kk = 0; kk < pairwise_poses_[i][j].size (); kk++)
        {
          if (pairwise_poses_[i][j][kk] == Eigen::Matrix4f::Zero ())
            continue;

          if (pairwise_registration_[i][j][kk].overlap_ < 0.9)
            continue;

          std::cout << pairwise_registration_[i][j][kk].overlap_ << " " << pairwise_registration_[i][j][kk].reg_error_ << std::endl;

          std::cout << kk << std::endl;
          PointCloudPtr pp (new pcl::PointCloud<PointT>);
          pcl::transformPointCloud (*processed_clouds_[j], *pp, pairwise_poses_[i][j][kk]);

          pcl::visualization::PointCloudColorHandlerCustom<PointT> handler_rgb (pp, 255, 0, 0);
          vis.addPointCloud<PointT> (pp, handler_rgb, cloud_name.str ());

          //pcl::visualization::PointCloudColorHandlerRGBField<PointT> handler_rgb (pp);
          //vis.addPointCloud<PointT> (pp, handler_rgb, cloud_name.str ());

          vis.spin ();
          vis.removePointCloud (cloud_name.str ());
        }
      }

      vis.removePointCloud (cloud_name.str ());

    }
  }

template<template<class > class Distance, typename PointT, typename PointTNormal>
  void
  faat_pcl::object_modelling::ObjectModeller<Distance, PointT, PointTNormal>::visualizeProcessed ()
  {
    pcl::visualization::PCLVisualizer vis ("");
    for (size_t i = 0; i < processed_clouds_.size (); i++)
    {
      std::stringstream cloud_name;
      cloud_name << "cloud_" << i;
      //pcl::visualization::PointCloudColorHandlerRGBField<PointT> handler_rgb (processed_clouds_[i]);
      pcl::visualization::PointCloudColorHandlerRandom<PointT> handler_rgb (processed_clouds_[i]);
      vis.addPointCloud<PointT> (processed_clouds_[i], handler_rgb, cloud_name.str ());
    }
    vis.spin ();
  }

template<template<class > class Distance, typename PointT, typename PointTNormal>
  void
  faat_pcl::object_modelling::ObjectModeller<Distance, PointT, PointTNormal>::visualizeGlobalAlignment (bool visualize_cameras)
  {
    pcl::visualization::PCLVisualizer vis ("visualizeGlobalAlignment");
    pcl::PointCloud<pcl::PointXYZ>::Ptr camera_positions (new pcl::PointCloud<pcl::PointXYZ>);
    int v1, v2, v3;
    vis.createViewPort (0, 0, 0.33, 1, v1);
    vis.createViewPort (0.33, 0, 0.66, 1, v2);
    vis.createViewPort (0.66, 0, 1, 1, v3);

    for (size_t i = 0; i < processed_clouds_.size (); i++)
    {
      std::stringstream cloud_name;
      cloud_name << "cloud_" << i;

      PointCloudPtr pp (new pcl::PointCloud<PointT>);
      pcl::transformPointCloud (*processed_clouds_[i], *pp, absolute_poses_[i]);
      //pcl::visualization::PointCloudColorHandlerRGBField<PointT> handler_rgb (pp);
      pcl::visualization::PointCloudColorHandlerRandom<PointT> handler_rgb (pp);
      vis.addPointCloud<PointT> (pp, handler_rgb, cloud_name.str ());

      if (visualize_cameras)
      {
        Eigen::Vector4f cam (0, 0, 0, 1);
        cam = absolute_poses_[i] * cam;

        std::stringstream camera_text;
        camera_text << "c" << i << std::endl;
        pcl::PointXYZ ptext;
        ptext.getVector4fMap () = cam;

        std::string text = camera_text.str ();
        vis.addText3D (text, ptext, 0.02, 0.0, 1.0, 0.0, camera_text.str ());

        camera_positions->push_back (ptext);
      }
    }

    if (visualize_cameras)
    {
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler_rgb (camera_positions, 255, 0, 0);
      vis.addPointCloud<pcl::PointXYZ> (camera_positions, handler_rgb, "camera positions");

      pcl::ConvexHull<pcl::PointXYZ> convex_hull;
      convex_hull.setInputCloud (camera_positions);
      convex_hull.setDimension (3);
      convex_hull.setComputeAreaVolume (true);
      pcl::PolygonMesh mesh_out;
      convex_hull.reconstruct (mesh_out);
      std::cout << "CHull Volume:" << convex_hull.getTotalVolume () << std::endl;
      vis.addPolygonMesh (mesh_out, "camera hull", v1);

      //visualize graph using connected lines
      int e = 0;
      for (size_t i = 0; i < processed_clouds_.size (); i++)
      {
        for (size_t j = (i + 1); j < processed_clouds_.size (); j++)
        {
          if ((boost::edge (j, i, ST_graph_).second) || (boost::edge (i, j, ST_graph_).second))
          {
            std::stringstream line_name;
            line_name << "line_" << e;
            pcl::PointXYZ p1, p2;
            p1 = camera_positions->points[i];
            p2 = camera_positions->points[j];
            vis.addLine<pcl::PointXYZ, pcl::PointXYZ> (p1, p2, line_name.str (), v2);
            e++;
          }
        }
      }

      //visualize LR graph...
      {
        boost::property_map<Graph, boost::edge_weight_t>::type EdgeWeightMap = get (boost::edge_weight_t (), LR_graph_);
        int e = 0;
        for (size_t i = 0; i < processed_clouds_.size (); i++)
        {
          for (size_t j = (i + 1); j < processed_clouds_.size (); j++)
          {
            if ((boost::edge (j, i, LR_graph_).second) || (boost::edge (i, j, LR_graph_).second))
            {
              std::stringstream line_name;
              line_name << "line_whole_graph_" << e;
              pcl::PointXYZ p1, p2;
              p1 = camera_positions->points[i];
              p2 = camera_positions->points[j];
              vis.addLine<pcl::PointXYZ, pcl::PointXYZ> (p1, p2, line_name.str (), v3);

              pcl::PointXYZ mid_point;
              mid_point.getVector3fMap () = (p1.getVector3fMap () + p2.getVector3fMap ()) / 2.f;

              {
                std::stringstream camera_name, edge_weight;
                camera_name << "w_" << e << std::endl;
                std::string text = boost::lexical_cast<std::string> (get (EdgeWeightMap, boost::edge (i, j, LR_graph_).first));
                vis.addText3D (text, mid_point, 0.005, 0.0, 1.0, 0.0, camera_name.str (), v3 + 1);
              }
              e++;
            }
          }
        }
      }
    }

    vis.spin ();
  }

#endif /* OBJECT_MODELLER_HPP_ */
