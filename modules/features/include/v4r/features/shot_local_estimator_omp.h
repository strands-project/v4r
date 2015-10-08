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
#include <v4r/common/miscellaneous.h>

namespace v4r
{
    template<typename PointInT, typename FeatureT>
      class V4R_EXPORTS SHOTLocalEstimationOMP : public LocalEstimator<PointInT, FeatureT>
      {

        typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;
        typedef typename pcl::PointCloud<FeatureT>::Ptr FeatureTPtr;
        typedef pcl::PointCloud<pcl::PointXYZ> KeypointCloud;

        using LocalEstimator<PointInT, FeatureT>::keypoint_extractor_;
        using LocalEstimator<PointInT, FeatureT>::neighborhood_indices_;
        using LocalEstimator<PointInT, FeatureT>::neighborhood_dist_;
        using LocalEstimator<PointInT, FeatureT>::normals_;
        using LocalEstimator<PointInT, FeatureT>::keypoint_indices_;

        pcl::PointIndices indices_;

      public:
        class Parameter : public LocalEstimator<PointInT, FeatureT>::Parameter
        {
        public:
            using LocalEstimator<PointInT, FeatureT>::Parameter::adaptative_MLS_;
            using LocalEstimator<PointInT, FeatureT>::Parameter::normal_computation_method_;
            using LocalEstimator<PointInT, FeatureT>::Parameter::support_radius_;

            Parameter():LocalEstimator<PointInT, FeatureT>::Parameter()
            {}
        }param_;

        SHOTLocalEstimationOMP (const Parameter &p = Parameter()) :
            LocalEstimator<PointInT, FeatureT>(p)
        {
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

        std::string getFeatureDescriptorName() const
        {
            return "shot_omp";
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
        estimate (const PointInTPtr & in, PointInTPtr & processed, PointInTPtr & keypoints, FeatureTPtr & signatures)
        {
          if (!keypoint_extractor_.size() || !in || !in->points.size())
            throw std::runtime_error("SHOTLocalEstimationOMP :: This feature needs a keypoint extractor and a non-empty input point cloud... please provide one");

          if(indices_.indices.size())
              pcl::copyPointCloud(*in, indices_, *in);

          processed = in;

          //pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
          pcl::MovingLeastSquares<PointInT, PointInT> mls;
          if (param_.adaptative_MLS_)
          {
            typename pcl::search::KdTree<PointInT>::Ptr tree;
            Eigen::Vector4f centroid_cluster;
            pcl::compute3DCentroid (*processed, centroid_cluster);
            float dist_to_sensor = centroid_cluster.norm ();
            float sigma = dist_to_sensor * 0.01f;
            mls.setSearchMethod (tree);
            mls.setSearchRadius (sigma);
            mls.setUpsamplingMethod (mls.SAMPLE_LOCAL_PLANE);
            mls.setUpsamplingRadius (0.002);
            mls.setUpsamplingStepSize (0.001);
            mls.setInputCloud (processed);

            PointInTPtr filtered (new pcl::PointCloud<PointInT>);
            mls.process (*filtered);
            filtered->is_dense = false;
            processed = filtered;
          }


          normals_.reset (new pcl::PointCloud<pcl::Normal>);
          v4r::computeNormals<PointInT>(processed, normals_, param_.normal_computation_method_);

          this->computeKeypoints(processed, keypoints, normals_);

          if (keypoints->points.size () == 0)
          {
            PCL_WARN("SHOTLocalEstimationOMP :: No keypoints were found\n");
            return false;
          }

          //compute signatures
          typedef typename pcl::SHOTEstimationOMP<PointInT, pcl::Normal, pcl::SHOT352> SHOTEstimator;
          typename pcl::search::KdTree<PointInT>::Ptr tree (new pcl::search::KdTree<PointInT>);
          tree->setInputCloud (processed);

          pcl::PointCloud<pcl::SHOT352>::Ptr shots (new pcl::PointCloud<pcl::SHOT352>);
          SHOTEstimator shot_estimate;
          shot_estimate.setNumberOfThreads (8);
          shot_estimate.setSearchMethod (tree);
          shot_estimate.setInputCloud (keypoints);
          shot_estimate.setSearchSurface(processed);
          shot_estimate.setInputNormals (normals_);
          shot_estimate.setRadiusSearch (param_.support_radius_);
          shot_estimate.compute (*shots);

          signatures->resize (shots->points.size ());
          signatures->width = static_cast<int> (shots->points.size ());
          signatures->height = 1;

          int size_feat = sizeof(signatures->points[0].histogram) / sizeof(float);

          size_t kept = 0;
          for (size_t k = 0; k < shots->points.size (); k++)
          {

            size_t NaNs = 0;
            for (int i = 0; i < size_feat; i++)
            {
              if (!pcl_isfinite(shots->points[k].descriptor[i]))
                NaNs++;
            }

            if (NaNs)
            {
              for (int i = 0; i < size_feat; i++)
                signatures->points[kept].histogram[i] = shots->points[k].descriptor[i];

              keypoints->points[kept] = keypoints->points[k];
              keypoint_indices_.indices[kept] = keypoint_indices_.indices[k];

              kept++;
            }
          }

          keypoint_indices_.indices.resize(kept);
          keypoints->points.resize(kept);
          keypoints->width = kept;
          signatures->points.resize (kept);
          signatures->width = kept;
          indices_.indices.clear ();
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
