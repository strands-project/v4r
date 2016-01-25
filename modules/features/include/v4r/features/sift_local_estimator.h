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
#include <SiftGPU/SiftGPU.h>
#include <opencv2/opencv.hpp>

namespace v4r
{

template<typename PointT>
class V4R_EXPORTS SIFTLocalEstimation : public LocalEstimator<PointT>
{
    using LocalEstimator<PointT>::keypoint_indices_;

    std::vector<int> indices_;
    boost::shared_ptr<SiftGPU> sift_;


public:
    SIFTLocalEstimation (const boost::shared_ptr<SiftGPU> &sift) : sift_(sift)
    {
        this->descr_name_ = "sift";
    }

    SIFTLocalEstimation ()
    {
        this->descr_name_ = "sift";

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

    bool
    estimate (const pcl::PointCloud<PointT> & in, pcl::PointCloud<PointT> & keypoints, std::vector<std::vector<float> > & signatures, std::vector<float> & scales);

    bool
    estimate (const cv::Mat_<cv::Vec3b> &colorImage, std::vector<SiftGPU::SiftKeypoint> & ks, std::vector<std::vector<float> > &signatures, std::vector<float> & scales);

    bool
    estimate(const pcl::PointCloud<PointT> & in, std::vector<std::vector<float> > & signatures);

    bool
    estimate (const pcl::PointCloud<PointT> & in, pcl::PointCloud<PointT> & processed, pcl::PointCloud<PointT> & keypoints, std::vector<std::vector<float> > & signatures);

    void
    setIndices (const std::vector<int> & indices)
    {
        indices_ = indices;
    }

    bool acceptsIndices() const
    {
        return true;
    }

    size_t getFeatureType() const
    {
        return SIFT_GPU;
    }
};
}

#endif /* REC_FRAMEWORK_SHOT_LOCAL_ESTIMATOR_H_ */
