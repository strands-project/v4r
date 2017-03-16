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

//This stuff is needed to be able to make the SIFT histograms persistent
POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::Histogram<128>, (float[128], histogram, histogramSIFT) )

namespace v4r
{

template<typename PointT>
class V4R_EXPORTS SIFTLocalEstimation : public LocalEstimator<PointT>
{
    using LocalEstimator<PointT>::keypoint_indices_;
    using LocalEstimator<PointT>::cloud_;
    using LocalEstimator<PointT>::indices_;
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
        bool use_rootSIFT_; ///< enables RootSIFT as described in Arandjelovic and Zisserman, Three things everyone should know to improve object retrieval (CVPR, 2012)
        int stride_;    ///< is dense_extraction, this will define the stride in pixel for extracting SIFT keypoints
        Parameter ( ):
            dense_extraction_ ( false ),
            use_rootSIFT_( true ),
            stride_ ( 20 )
        {}
    }param_;

#ifdef HAVE_SIFTGPU
    SIFTLocalEstimation (const boost::shared_ptr<SiftGPU> &sift) :
      max_distance_ (std::numeric_limits<float>::max()),
      sift_(sift)
    {
        descr_name_ = "sift";
        descr_type_ = FeatureType::SIFT_GPU;
        descr_dims_ = 128;
    }

    SIFTLocalEstimation ()
        : max_distance_ (std::numeric_limits<float>::max())
    {
        descr_name_ = "sift";
        descr_type_ = FeatureType::SIFT_GPU;
        descr_dims_ = 128;

        //init sift
//        const char *argv[] = {"-m", "-fo","-1", "-s", "-v", "0", "-pack", "-maxd", "8192"};   // for Kinect v2
        const char *argv[] = {"-m", "-fo","-1", "-s", "-v", "0", "-pack"};  // for Kinect v1
        int argc = sizeof(argv) / sizeof(char*);
        sift_.reset(new SiftGPU());
        sift_->ParseParam (argc, (char **)argv);

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

    /**
     * @brief setMaxDistance sets the maximum distance in meter for a keypoint to be valid
     * @param max_distance in meters
     */
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


#ifdef HAVE_SIFTGPU
    /**
     * @brief matchSIFT matches two sets of SIFT descriptors
     * @param desc1 descriptor 1
     * @param desc2 descriptor 2
     * @return indices of the matching descriptors
     */
    std::vector<std::pair<int, int> >
    matchSIFT( const std::vector<std::vector<float> >& desc1, const std::vector<std::vector<float> >& desc2);
#endif

    typedef boost::shared_ptr< SIFTLocalEstimation<PointT> > Ptr;
    typedef boost::shared_ptr< SIFTLocalEstimation<PointT> const> ConstPtr;
};
}
