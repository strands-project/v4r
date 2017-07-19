/******************************************************************************
 * Copyright (c) 2012 Aitor Aldoma
 * Copyright (c) 2016 Thomas Faeulhammer
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

namespace v4r
{
class V4R_EXPORTS OURCVFHEstimatorParameter
{
public:
    std::vector<float> eps_angle_threshold_vector_;
    std::vector<float> curvature_threshold_vector_;
    std::vector<float> cluster_tolerance_vector_;
    float refine_factor_;
    bool normalize_bins_;
    size_t min_points_;
    float axis_ratio_;
    float min_axis_value_;


    OURCVFHEstimatorParameter() :
        eps_angle_threshold_vector_ ( { 10.f*M_PI/180.f } ),
        curvature_threshold_vector_ ( {0.04} ),
        cluster_tolerance_vector_ ( {0.02f} ), //3.f, 0.015f
        refine_factor_ (1.f),
        normalize_bins_ (false),
        min_points_(50),
        axis_ratio_ (0.8f),
        min_axis_value_(0.925f)
    {}
};

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
    using GlobalEstimator<PointT>::transforms_;

    OURCVFHEstimatorParameter param_;

public:
    OURCVFHEstimator(const OURCVFHEstimatorParameter &p = OURCVFHEstimatorParameter() )
        :
          GlobalEstimator<PointT>("ourcvfh", FeatureType::OURCVFH, 308),
          param_(p)
    { }

    bool
    compute (Eigen::MatrixXf &signature);

    bool
    needNormals() const
    {
        return true;
    }
};
}
