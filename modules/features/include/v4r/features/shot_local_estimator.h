/*
 * shot_local_estimator.h
 *
 *  Created on: Mar 24, 2012
 *      Author: aitor
 */

#ifndef FAAT_PCL_REC_FRAMEWORK_SHOT_LOCAL_ESTIMATOR_H_
#define FAAT_PCL_REC_FRAMEWORK_SHOT_LOCAL_ESTIMATOR_H_

#include "local_estimator.h"
#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/common/normals.h>
#include <pcl/features/shot.h>
#include <pcl/io/pcd_io.h>

namespace v4r
{
    template<typename PointInT, typename FeatureT>
      class V4R_EXPORTS SHOTLocalEstimation : public LocalEstimator<PointInT, FeatureT>
      {

        typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;
        typedef typename pcl::PointCloud<FeatureT>::Ptr FeatureTPtr;

        using LocalEstimator<PointInT, FeatureT>::keypoint_extractor_;
        using LocalEstimator<PointInT, FeatureT>::normals_;

      public:
      public:
        class Parameter : public LocalEstimator<PointInT, FeatureT>::Parameter
        {
            using LocalEstimator<PointInT, FeatureT>::Parameter::adaptative_MLS_;
            using LocalEstimator<PointInT, FeatureT>::Parameter::normal_computation_method_;
            using LocalEstimator<PointInT, FeatureT>::Parameter::support_radius_;

            Parameter():LocalEstimator<PointInT, FeatureT>::Parameter()
            {}
        }param_;

        SHOTLocalEstimation (const Parameter &p = Parameter()) : LocalEstimator<PointInT, FeatureT>(p)
        {
            param_ = p;
        }

        size_t getFeatureType() const
        {
          return SHOT;
        }

        std::string getFeatureDescriptorName() const
        {
            return "shot";
        }

        bool estimate (const PointInTPtr & in, PointInTPtr & processed, PointInTPtr & keypoints, FeatureTPtr & signatures)
        {
            if (!keypoint_extractor_.size() || !in || !in->points.size())
              throw std::runtime_error("SHOTLocalEstimationOMP :: This feature needs a keypoint extractor and a non-empty input point cloud... please provide one");


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
          typedef typename pcl::SHOTEstimation<PointInT, pcl::Normal, pcl::SHOT352> SHOTEstimator;
          typename pcl::search::KdTree<PointInT>::Ptr tree (new pcl::search::KdTree<PointInT>);

          pcl::PointCloud<pcl::SHOT352>::Ptr shots (new pcl::PointCloud<pcl::SHOT352>);
          SHOTEstimator shot_estimate;
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

          for (size_t k = 0; k < shots->points.size (); k++)
            for (int i = 0; i < size_feat; i++)
              signatures->points[k].histogram[i] = shots->points[k].descriptor[i];

          return true;

        }

        bool
        needNormals ()
        {
          return true;
        }
      };
}

#endif /* REC_FRAMEWORK_SHOT_LOCAL_ESTIMATOR_H_ */
