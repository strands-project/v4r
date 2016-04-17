/******************************************************************************
 * Copyright (c) 2012 Aitor Aldoma, Thomas Faeulhammer
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

#ifndef V4R_OURCVFH_ESTIMATOR_H_
#define V4R_OURCVFH_ESTIMATOR_H_

#include <v4r/core/macros.h>
#include <v4r/features/global_estimator.h>
#include <v4r/features/types.h>

#include <v4r/features/pcl_ourcvfh.h>
#include <pcl/search/kdtree.h>
#include <glog/logging.h>

namespace v4r
{
    template<typename PointT>
      class V4R_EXPORTS OURCVFHEstimator : public GlobalEstimator<PointT>
      {
      private:
          using GlobalEstimator<PointT>::indices_;
          using GlobalEstimator<PointT>::cloud_;
          using GlobalEstimator<PointT>::normals_;
          using GlobalEstimator<PointT>::descr_name_;
          using GlobalEstimator<PointT>::descr_type_;
          using GlobalEstimator<PointT>::feature_dimensions_;

      public:
          OURCVFHEstimator()
          {
              descr_name_ = "ourcvfh";
              descr_type_ = FeatureType::OURCVFH;
              feature_dimensions_ = 308;
          }

          bool
          compute (Eigen::MatrixXf &signature)
          {
              CHECK(cloud_ && !cloud_->points.empty() && normals_);
              pcl::PointCloud<pcl::VFHSignature308> descriptors;

              typename pcl::search::KdTree<PointT>::Ptr kdtree (new pcl::search::KdTree<PointT>);

              v4r::OURCVFHEstimation<PointT, pcl::Normal, pcl::VFHSignature308> ourcvfh;
              if(!indices_.empty())   /// NOTE: setIndices does not seem to work for ESF
              {
                  typename pcl::PointCloud<PointT>::Ptr cloud_roi (new pcl::PointCloud<PointT>);
                  typename pcl::PointCloud<pcl::Normal>::Ptr normals_roi (new pcl::PointCloud<pcl::Normal>);
                  pcl::copyPointCloud(*cloud_, indices_, *cloud_roi);
                  pcl::copyPointCloud(*normals_, indices_, *normals_roi);
                  ourcvfh.setInputCloud(cloud_roi);
                  ourcvfh.setInputNormals(normals_roi);
              }
              else
              {
                  ourcvfh.setInputCloud (cloud_);
                  ourcvfh.setInputNormals(normals_);
              }

              ourcvfh.setSearchMethod(kdtree);
              ourcvfh.setEPSAngleThreshold(5.0 / 180.0 * M_PI); // 5 degrees.
              ourcvfh.setCurvatureThreshold(1.0);
              ourcvfh.setNormalizeBins(false);
              // Set the minimum axis ratio between the SGURF axes. At the disambiguation phase,
              // this will decide if additional Reference Frames need to be created, if ambiguous.
              ourcvfh.setAxisRatio(0.8);
              try{
                  ourcvfh.compute(descriptors);
              }
              catch (std::exception &e)
              {
                  std::cerr << "Could not compute descriptor. " << e.what() << std::endl;
                  return false;
              }

              signature.resize(descriptors.points.size(), feature_dimensions_);

              for(size_t pt=0; pt<descriptors.points.size(); pt++)
              {
                  for(size_t i=0; i<feature_dimensions_; i++)
                    signature(pt, i) = descriptors.points[pt].histogram[i];
              }

              indices_.clear();
              return true;
          }

          bool
          needNormals() const
          {
              return true;
          }
      };
}

#endif
