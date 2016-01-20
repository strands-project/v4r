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

#include "local_estimator.h"
#include <pcl/point_cloud.h>
#include <SiftGPU/SiftGPU.h>
#include <GL/glut.h>
#include <opencv2/opencv.hpp>
#include <v4r/common/faat_3d_rec_framework_defines.h>

namespace v4r
{

template<typename PointInT, typename FeatureT>
class V4R_EXPORTS SIFTLocalEstimation : public LocalEstimator<PointInT, FeatureT>
{
    typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;
    typedef typename pcl::PointCloud<FeatureT>::Ptr FeatureTPtr;

    using LocalEstimator<PointInT, FeatureT>::keypoint_indices_;

    std::vector<int> indices_;
    boost::shared_ptr<SiftGPU> sift_;

public:
    SIFTLocalEstimation (boost::shared_ptr<SiftGPU> sift = boost::shared_ptr<SiftGPU>())
    {
        this->descr_name_ = "sift";

        if (!sift)
        {
            //init sift

            static char kw[][16] = {"-m", "-fo", "-1", "-s", "-v", "1", "-pack"};
            char * argv[] = {kw[0], kw[1], kw[2], kw[3],kw[4],kw[5],kw[6], NULL};

            int argc = sizeof(argv) / sizeof(char*);
            sift_.reset( new SiftGPU () );
            sift_->ParseParam (argc, argv);

            //create an OpenGL context for computation
            if (sift_->CreateContextGL () != SiftGPU::SIFTGPU_FULL_SUPPORTED)
                throw std::runtime_error ("PSiftGPU::PSiftGPU: No GL support!");
        }
        else
            sift_ = sift;
    }

    bool
    estimate (const PointInTPtr & in, PointInTPtr & keypoints, FeatureTPtr & signatures, std::vector<float> & scales);

    bool
    estimate (const cv::Mat_ < cv::Vec3b > colorImage, std::vector<SiftGPU::SiftKeypoint> & ks, FeatureTPtr & signatures, std::vector<float> & scales);

    bool
    estimate(const PointInTPtr & in, FeatureTPtr & signatures);

    bool
    estimate (const PointInTPtr & in, PointInTPtr & processed, PointInTPtr & keypoints, FeatureTPtr & signatures);

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
        return SIFT;
    }
};
}

#endif /* REC_FRAMEWORK_SHOT_LOCAL_ESTIMATOR_H_ */
