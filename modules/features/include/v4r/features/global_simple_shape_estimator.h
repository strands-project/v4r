/******************************************************************************
 * Copyright (c) 2017 Thomas Faeulhammer
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

#pragma once

#include <v4r/core/macros.h>
#include <v4r/features/global_estimator.h>
#include <v4r/features/types.h>

#include <glog/logging.h>

namespace v4r
{
/**
 * @brief The SimpleShapeEstimator class implements a simple global description
 * in terms of elongation of the point cloud along the principal components
 * @author Thomas Faeulhammer
 */
template<typename PointT>
class V4R_EXPORTS SimpleShapeEstimator : public GlobalEstimator<PointT>
{
private:
    using GlobalEstimator<PointT>::indices_;
    using GlobalEstimator<PointT>::cloud_;
    using GlobalEstimator<PointT>::descr_name_;
    using GlobalEstimator<PointT>::descr_type_;
    using GlobalEstimator<PointT>::feature_dimensions_;

public:
    SimpleShapeEstimator()
        : GlobalEstimator<PointT>("simple_shape", FeatureType::SIMPLE_SHAPE, 3)
    {}

    bool compute (Eigen::MatrixXf &signature);

    bool needNormals() const { return false; }

    typedef boost::shared_ptr< SimpleShapeEstimator<PointT> > Ptr;
    typedef boost::shared_ptr< SimpleShapeEstimator<PointT> const> ConstPtr;
};
}
