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

#ifndef V4R_SIFT_ESTIMATOR_H_
#define V4R_SIFT_ESTIMATOR_H_
#include <v4r_config.h>

#include <v4r/features/local_estimator.h>
#include <v4r/features/types.h>
#include <pcl/point_cloud.h>
#include <opencv2/opencv.hpp>

#ifdef HAVE_SIFTGPU
#include <SiftGPU/SiftGPU.h>
#else
#include <opencv2/nonfree/features2d.hpp> // requires OpenCV non-free module
#endif

namespace v4r
{

template<typename PointT>
class V4R_EXPORTS SIFTLocalEstimation : public LocalEstimator<PointT>
{
    using LocalEstimator<PointT>::keypoint_indices_;
    using LocalEstimator<PointT>::keypoints_;
    using LocalEstimator<PointT>::cloud_;
    using LocalEstimator<PointT>::indices_;
    using LocalEstimator<PointT>::processed_;
    using LocalEstimator<PointT>::descr_name_;
    using LocalEstimator<PointT>::descr_type_;
    using LocalEstimator<PointT>::descr_dims_;
    Eigen::VectorXf scales_;
    float max_distance_;

#ifdef HAVE_SIFTGPU
    boost::shared_ptr<SiftGPU> sift_;
#else
    boost::shared_ptr<cv::SIFT> sift_;
#endif

public:
    class Parameter
    {
    public:
        bool dense_extraction_;
        int stride_;    /// @brief is dense_extraction, this will define the stride in pixel for extracting SIFT keypoints
        Parameter
        (
                bool dense_extraction = false,
                int stride  = 20
        ):
            dense_extraction_ ( dense_extraction ),
            stride_ ( stride )
        {}
    }param_;

#ifdef HAVE_SIFTGPU
    SIFTLocalEstimation (const boost::shared_ptr<SiftGPU> &sift) : sift_(sift)
    {
        descr_name_ = "sift";
        descr_type_ = FeatureType::SIFT_GPU;
        descr_dims_ = 128;
    }

    SIFTLocalEstimation ()
    {
        descr_name_ = "sift";
        descr_type_ = FeatureType::SIFT_GPU;
        descr_dims_ = 128;

        //init sift
        static char kw[][16] = {"-m", "-fo", "-1", "-s", "-v", "0", "-pack"};
        char * argv[] = {kw[0], kw[1], kw[2], kw[3],kw[4],kw[5],kw[6], NULL};
        int argc = sizeof(argv) / sizeof(char*);
        sift_.reset(new SiftGPU());
        sift_->ParseParam (argc, argv);

        //create an OpenGL context for computation
        if (sift_->CreateContextGL () != SiftGPU::SIFTGPU_FULL_SUPPORTED)
            throw std::runtime_error ("PSiftGPU::PSiftGPU: No GL support!");
    }

#else
    SIFTLocalEstimation (double threshold = 0.03, double edge_threshold = 10.0)
    {
      this->descr_name_ = "sift_opencv";
        this->descr_type_ = FeatureType::SIFT_OPENCV;
      sift_.reset(new cv::SIFT(0, 3, threshold, edge_threshold));
    }
#endif

    void
    compute (const cv::Mat_<cv::Vec3b> &colorImage, Eigen::Matrix2Xf &keypoints, std::vector<std::vector<float> > &signatures);

    void
    compute (std::vector<std::vector<float> > & signatures);

    bool 
	acceptsIndices() const
    {
        return true;
    }

    void
    setMaxDistance(float max_distance)
    {
        max_distance_ = max_distance;
    }

    bool
    needNormals() const
    {
        return false;
    }


    typedef boost::shared_ptr< SIFTLocalEstimation<PointT> > Ptr;
    typedef boost::shared_ptr< SIFTLocalEstimation<PointT> const> ConstPtr;
};
}

#endif /* REC_FRAMEWORK_SHOT_LOCAL_ESTIMATOR_H_ */
