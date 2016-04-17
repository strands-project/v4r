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

#ifndef V4R_ESF_ESTIMATOR_H_
#define V4R_ESF_ESTIMATOR_H_

#include <v4r/core/macros.h>
#include <v4r/features/global_estimator.h>
#include <v4r/features/types.h>

#include <pcl/features/esf.h>
#include <glog/logging.h>

namespace v4r
{
template<typename PointT>
class V4R_EXPORTS ESFEstimation : public GlobalEstimator<PointT>
{
private:
    using GlobalEstimator<PointT>::indices_;
    using GlobalEstimator<PointT>::cloud_;
    using GlobalEstimator<PointT>::descr_name_;
    using GlobalEstimator<PointT>::descr_type_;
    using GlobalEstimator<PointT>::feature_dimensions_;

public:
    ESFEstimation()
    {
        descr_name_ = "esf";
        descr_type_ = FeatureType::ESF;
        feature_dimensions_ = 640;
    }

    bool
    compute (Eigen::MatrixXf &signature)
    {
        CHECK(cloud_ && !cloud_->points.empty());
        typename pcl::ESFEstimation<PointT, pcl::ESFSignature640> esf;
        pcl::PointCloud<pcl::ESFSignature640> ESF_signature;

        if(!indices_.empty())   /// NOTE: setIndices does not seem to work for ESF
        {
            typename pcl::PointCloud<PointT>::Ptr cloud_roi (new pcl::PointCloud<PointT>);
            pcl::copyPointCloud(*cloud_, indices_, *cloud_roi);
            esf.setInputCloud(cloud_roi);
        }
        else
        {
            esf.setInputCloud (cloud_);
        }

        esf.compute (ESF_signature);
        signature.resize(ESF_signature.points.size(), feature_dimensions_);

        for(size_t pt=0; pt<ESF_signature.points.size(); pt++)
        {
            for(size_t i=0; i<feature_dimensions_; i++)
                signature(pt, i) = ESF_signature.points[pt].histogram[i];
        }

        indices_.clear();

        return true;
    }

    bool needNormals() const
    {
        return false;
    }
};
}

#endif
