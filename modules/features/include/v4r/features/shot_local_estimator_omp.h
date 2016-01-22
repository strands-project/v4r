/*
 * shot_local_estimator.h
 *
 *  Created on: Mar 24, 2012
 *      Author: aitor
 */

#ifndef FAAT_PCL_REC_FRAMEWORK_SHOT_LOCAL_ESTIMATOR_OMP_H_
#define FAAT_PCL_REC_FRAMEWORK_SHOT_LOCAL_ESTIMATOR_OMP_H_

#include "local_estimator.h"
#include <pcl/features/shot_omp.h>
#include <pcl/io/pcd_io.h>
#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/common/normals.h>

namespace v4r
{
    template<typename PointT, typename FeatureT>
      class V4R_EXPORTS SHOTLocalEstimationOMP : public LocalEstimator<PointT, FeatureT>
      {
        typedef typename pcl::PointCloud<PointT>::Ptr PointInTPtr;
        typedef typename pcl::PointCloud<FeatureT>::Ptr FeatureTPtr;
        typedef pcl::PointCloud<pcl::PointXYZ> KeypointCloud;

        using LocalEstimator<PointT, FeatureT>::keypoint_extractor_;
        using LocalEstimator<PointT, FeatureT>::normals_;
        using LocalEstimator<PointT, FeatureT>::keypoint_indices_;

        pcl::PointIndices indices_;

      public:
        class Parameter : public LocalEstimator<PointT, FeatureT>::Parameter
        {
        public:
            using LocalEstimator<PointT, FeatureT>::Parameter::adaptative_MLS_;
            using LocalEstimator<PointT, FeatureT>::Parameter::normal_computation_method_;
            using LocalEstimator<PointT, FeatureT>::Parameter::support_radius_;

            Parameter():LocalEstimator<PointT, FeatureT>::Parameter()
            {}
        }param_;

        SHOTLocalEstimationOMP (const Parameter &p = Parameter()) : LocalEstimator<PointT, FeatureT>(p)
        {
            this->descr_name_ = "shot_omp";
            param_ = p;
        }

        void
        setIndices (const pcl::PointIndices & p_indices)
        {
          indices_ = p_indices;
        }

        size_t getFeatureType() const
        {
            return SHOT;
        }

        void
        setIndices(const std::vector<int> & p_indices)
        {
          indices_.indices = p_indices;
        }

        bool acceptsIndices() const
        {
          return true;
        }

        bool
        estimate (const pcl::PointCloud<PointT> & in, pcl::PointCloud<PointT> & processed, pcl::PointCloud<PointT> & keypoints, pcl::PointCloud<FeatureT> & signatures)
        {
            (void) processed;

            if ( keypoint_extractor_.empty() || in.points.empty())
                throw std::runtime_error("SHOTLocalEstimationOMP :: This feature needs a keypoint extractor and a non-empty input point cloud... please provide one");


//          if( !indices_.indices.empty() )
//          {
//              pcl::copyPointCloud(*in, indices_, *in);
//          }

//          processed = in;

          //pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
//          if (param_.adaptative_MLS_)
//          {
//            typename pcl::search::KdTree<PointInT>::Ptr tree;
//            Eigen::Vector4f centroid_cluster;
//            pcl::compute3DCentroid (*processed, centroid_cluster);
//            float dist_to_sensor = centroid_cluster.norm ();
//            float sigma = dist_to_sensor * 0.01f;

//            pcl::MovingLeastSquares<PointInT, PointInT> mls;
//            mls.setSearchMethod (tree);
//            mls.setSearchRadius (sigma);
//            mls.setUpsamplingMethod (mls.SAMPLE_LOCAL_PLANE);
//            mls.setUpsamplingRadius (0.002);
//            mls.setUpsamplingStepSize (0.001);
//            mls.setInputCloud (processed);

//            PointInTPtr filtered (new pcl::PointCloud<PointInT>);
//            mls.process (*filtered);
//            filtered->is_dense = false;
//            processed = filtered;
//          }

          if ( !normals_ || normals_->points.size() != keypoints.points.size() )
          {
              normals_.reset(new pcl::PointCloud<pcl::Normal>());
              computeNormals<PointT>(in.makeShared(), normals_, param_.normal_computation_method_);
          }

          this->computeKeypoints(in, keypoints, normals_);

          if( keypoints.points.empty() )
          {
              PCL_WARN("SHOTLocalEstimationOMP :: No keypoints were found\n");
              return false;
          }

      //            uniform_kp_extractor->setMaxDistance( 1000.0 ); // for training we want to consider all points (except nan values)

          // filter inite points and normals before describing keypoint

          pcl::PointCloud<pcl::Normal>::Ptr normals_filtered (new pcl::PointCloud<pcl::Normal>);
          PointInTPtr cloud_filtered (new pcl::PointCloud<PointT>);
          cloud_filtered->points.resize(keypoints.points.size());
          normals_filtered->points.resize(normals_->points.size());
          size_t kept=0;
          for(size_t i=0; i<keypoints.points.size(); i++)
          {
              if( pcl::isFinite(keypoints.points[i]) && pcl::isFinite(normals_->points[i]))
              {
                  cloud_filtered->points[kept] = keypoints.points[i];
                  normals_filtered->points[kept] = normals_->points[i];
                  kept++;
              }
          }
          cloud_filtered->points.resize(kept);
          normals_filtered->points.resize(kept);
          cloud_filtered->height = normals_filtered->height = 1;
          cloud_filtered->width = normals_filtered->width = kept;

          //compute signatures
          typedef typename pcl::SHOTEstimationOMP<PointT, pcl::Normal, pcl::SHOT352> SHOTEstimator;
          typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
          tree->setInputCloud (cloud_filtered);

          pcl::PointCloud<pcl::SHOT352>::Ptr shots (new pcl::PointCloud<pcl::SHOT352>);
          SHOTEstimator shot_estimate;
          shot_estimate.setNumberOfThreads (8);
          shot_estimate.setSearchMethod (tree);
          shot_estimate.setInputCloud (keypoints.makeShared());
          shot_estimate.setSearchSurface(cloud_filtered);
          shot_estimate.setInputNormals (normals_filtered);
          shot_estimate.setRadiusSearch (param_.support_radius_);
          shot_estimate.compute (*shots);

          signatures.resize (shots->points.size ());
          signatures.width = static_cast<int> (shots->points.size ());
          signatures.height = 1;

          int size_feat = sizeof(signatures.points[0].histogram) / sizeof(float);

          kept = 0;
          for (size_t k = 0; k < shots->points.size (); k++)
          {

            bool is_nan = false;
            for (int i = 0; i < size_feat; i++)
            {
              if (!pcl_isfinite(shots->points[k].descriptor[i]))
              {
                is_nan = true;
                break;
              }
            }

            if (!is_nan)
            {
              for (int i = 0; i < size_feat; i++)
                signatures.points[kept].histogram[i] = shots->points[k].descriptor[i];

              keypoints.points[kept] = keypoints.points[k];
              keypoint_indices_[kept] = keypoint_indices_[k];

              kept++;
            }
          }

          keypoint_indices_.resize(kept);
          keypoints.points.resize(kept);
          keypoints.width = kept;
          signatures.points.resize (kept);
          signatures.width = kept;
          normals_.reset();

          return true;
        }

        bool
        needNormals ()
        {
          return true;
        }
      };
}

#endif /* REC_FRAMEWORK_SHOT_LOCAL_ESTIMATOR_OMP_H_ */
