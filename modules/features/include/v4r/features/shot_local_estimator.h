/******************************************************************************
 * Copyright (c) 2012 Aitor Aldoma
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/


#ifndef V4R_SHOT_LOCAL_ESTIMATOR_H_
#define V4R_FRAMEWORK_SHOT_LOCAL_ESTIMATOR_H_

#include "local_estimator.h"
#include <v4r/features/types.h>
#include <v4r/common/normals.h>
#include <pcl/features/shot.h>
#include <pcl/io/pcd_io.h>

namespace v4r
{
    template<typename PointT>
      class V4R_EXPORTS SHOTLocalEstimation : public LocalEstimator<PointT>
      {

        typedef typename pcl::PointCloud<PointT>::Ptr PointInTPtr;

        using LocalEstimator<PointT>::keypoint_extractor_;
        using LocalEstimator<PointT>::normals_;

      public:
      public:
        class Parameter : public LocalEstimator<PointT>::Parameter
        {
        public:
            using LocalEstimator<PointT>::Parameter::adaptative_MLS_;
            using LocalEstimator<PointT>::Parameter::normal_computation_method_;
            using LocalEstimator<PointT>::Parameter::support_radius_;

            Parameter():LocalEstimator<PointT>::Parameter()
            {}
        }param_;

        SHOTLocalEstimation (const Parameter &p = Parameter()) : LocalEstimator<PointT>(p)
        {
            this->descr_name_ = "shot";
            param_ = p;
        }

        size_t getFeatureType() const
        {
          return SHOT;
        }

        bool estimate (const pcl::PointCloud<PointT> & in, pcl::PointCloud<PointT> & processed, pcl::PointCloud<PointT> & keypoints, std::vector<std::vector<float> > & signatures)
        {
            if (!keypoint_extractor_.size() || in.points.empty())
              throw std::runtime_error("SHOTLocalEstimationOMP :: This feature needs a keypoint extractor and a non-empty input point cloud... please provide one");

            pcl::MovingLeastSquares<PointT, PointT> mls;
            if (param_.adaptative_MLS_)
            {
              typename pcl::search::KdTree<PointT>::Ptr tree;
              Eigen::Vector4f centroid_cluster;
              pcl::compute3DCentroid (processed, centroid_cluster);
              float dist_to_sensor = centroid_cluster.norm ();
              float sigma = dist_to_sensor * 0.01f;
              mls.setSearchMethod (tree);
              mls.setSearchRadius (sigma);
              mls.setUpsamplingMethod (mls.SAMPLE_LOCAL_PLANE);
              mls.setUpsamplingRadius (0.002);
              mls.setUpsamplingStepSize (0.001);
              mls.setInputCloud (processed.makeShared());

              pcl::PointCloud<PointT> filtered;
              mls.process (filtered);
              filtered.is_dense = false;
              processed = filtered;
            }

            normals_.reset (new pcl::PointCloud<pcl::Normal>);
            computeNormals<PointT>(processed.makeShared(), normals_, param_.normal_computation_method_);

            this->computeKeypoints(processed, keypoints, normals_);

            if (keypoints.points.empty() == 0)
            {
              PCL_WARN("SHOTLocalEstimationOMP :: No keypoints were found\n");
              return false;
            }

          //compute signatures
          typedef typename pcl::SHOTEstimation<PointT, pcl::Normal, pcl::SHOT352> SHOTEstimator;
          typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);

          pcl::PointCloud<pcl::SHOT352>::Ptr shots (new pcl::PointCloud<pcl::SHOT352>);
          SHOTEstimator shot_estimate;
          shot_estimate.setSearchMethod (tree);
          shot_estimate.setInputCloud (keypoints.makeShared());
          shot_estimate.setSearchSurface(processed.makeShared());
          shot_estimate.setInputNormals (normals_);
          shot_estimate.setRadiusSearch (param_.support_radius_);
          shot_estimate.compute (*shots);
          signatures.resize (shots->points.size (), std::vector<float>(352));

          int size_feat = 352;

          for (size_t k = 0; k < shots->points.size (); k++)
            for (int i = 0; i < size_feat; i++)
              signatures[k][i] = shots->points[k].descriptor[i];

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
