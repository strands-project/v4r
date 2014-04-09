/*
 * shot_local_estimator.h
 *
 *  Created on: Mar 24, 2012
 *      Author: aitor
 */

#ifndef FAAT_PCL_REC_FRAMEWORK_COLORSHOT_LOCAL_ESTIMATOR_H_
#define FAAT_PCL_REC_FRAMEWORK_COLORSHOT_LOCAL_ESTIMATOR_H_

#include <faat_pcl/3d_rec_framework/defines/faat_3d_rec_framework_defines.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/local/local_estimator.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/normal_estimator.h>
#include <pcl/features/shot.h>
#include <pcl/io/pcd_io.h>

namespace faat_pcl
{
  namespace rec_3d_framework
  {
    template<typename PointInT, typename FeatureT>
      class FAAT_3D_FRAMEWORK_API ColorSHOTLocalEstimation : public LocalEstimator<PointInT, FeatureT>
      {

        typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;
        typedef typename pcl::PointCloud<FeatureT>::Ptr FeatureTPtr;

        using LocalEstimator<PointInT, FeatureT>::support_radius_;
        using LocalEstimator<PointInT, FeatureT>::normal_estimator_;
        using LocalEstimator<PointInT, FeatureT>::keypoint_extractor_;

      public:
        bool
        estimate (const PointInTPtr & in, PointInTPtr & processed, PointInTPtr & keypoints, FeatureTPtr & signatures)
        {

          if (!normal_estimator_)
          {
            PCL_ERROR("SHOTLocalEstimation :: This feature needs normals... please provide a normal estimator\n");
            return false;
          }

          if (keypoint_extractor_.size() == 0)
          {
            PCL_ERROR("SHOTLocalEstimation :: This feature needs a keypoint extractor... please provide one\n");
            return false;
          }

          pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
          normals.reset (new pcl::PointCloud<pcl::Normal>);
          {
            pcl::ScopeTime t ("Compute normals");
            normal_estimator_->estimate (in, processed, normals);
          }

          //compute keypoints
          computeKeypoints(processed, keypoints, normals);

          if (keypoints->points.size () == 0)
          {
            PCL_WARN("ColorSHOTLocalEstimation :: No keypoints were found\n");
            return false;
          }

          //compute signatures
          typedef typename pcl::SHOTColorEstimation<PointInT, pcl::Normal, pcl::SHOT1344> SHOTEstimator;
          typename pcl::search::KdTree<PointInT>::Ptr tree (new pcl::search::KdTree<PointInT>);

          pcl::PointCloud<pcl::SHOT1344>::Ptr shots (new pcl::PointCloud<pcl::SHOT1344>);
          SHOTEstimator shot_estimate;
          shot_estimate.setSearchMethod (tree);
          shot_estimate.setInputNormals (normals);
          shot_estimate.setInputCloud (keypoints);
          shot_estimate.setSearchSurface(processed);
          shot_estimate.setRadiusSearch (support_radius_);
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

      };
  }
}

#endif /* REC_FRAMEWORK_COLORSHOT_LOCAL_ESTIMATOR_H_ */
