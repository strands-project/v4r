#include <v4r_config.h>

#ifdef HAVE_SIFTGPU
#include <v4r/features/sift_local_estimator.h>
#include <v4r/common/pcl_opencv.h>
#include <v4r/common/miscellaneous.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <GL/glut.h>

namespace v4r
{

template<typename PointT>
void
SIFTLocalEstimation<PointT>::compute (std::vector<std::vector<float> > &signatures)
{
    cv::Mat colorImage = ConvertPCLCloud2Image(*cloud_);
    std::vector<SiftGPU::SiftKeypoint> ks;
    std::vector<float> scales;
    compute(colorImage, ks, signatures, scales);
    keypoint_indices_.resize(ks.size());

    std::vector<bool> obj_mask;
    if(indices_.empty())
        obj_mask.resize(cloud_->width * cloud_->height, true);
    else
        obj_mask = createMaskFromIndices(indices_, cloud_->width * cloud_->height);

    size_t kept = 0;
    for(size_t i=0; i < ks.size(); i++)
    {
        const int v = (int)(ks[i].y);
        const int u = (int)(ks[i].x);
        const int idx = v * cloud_->width + u;

        if(u >= 0 && v >= 0 && u < cloud_->width && v < cloud_->height && pcl::isFinite(cloud_->points[idx]) && obj_mask[idx])
        {
            keypoint_indices_[kept] = idx;
            scales[kept] = scales[i];
            signatures[kept] = signatures[i];
            kept++;
        }
    }

//    keypoints.points.resize(kept);
    scales.resize(kept);
    keypoint_indices_.resize(kept);
    signatures.resize(kept);
    indices_.clear();
    keypoints_.reset( new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*cloud_, keypoint_indices_, *keypoints_);
    processed_ = cloud_;
}


template<typename PointT>
void
SIFTLocalEstimation<PointT>::compute (const cv::Mat_ < cv::Vec3b > &colorImage, std::vector<SiftGPU::SiftKeypoint> & ks, std::vector<std::vector<float> > &signatures, std::vector<float> & scales)
{
    cv::Mat grayImage;
    cv::cvtColor (colorImage, grayImage, CV_BGR2GRAY);

    cv::Mat descriptors;

    if (sift_->CreateContextGL () != SiftGPU::SIFTGPU_FULL_SUPPORTED)
        throw std::runtime_error ("SiftGPU: No GL support!");

    sift_->VerifyContextGL();
    if (sift_->RunSIFT (grayImage.cols, grayImage.rows, grayImage.ptr<uchar> (0), GL_LUMINANCE, GL_UNSIGNED_BYTE))
    {
        int num = sift_->GetFeatureNum ();
        if (num > 0)
        {
            ks.resize(num);
            descriptors = cv::Mat(num,128,CV_32F);
            sift_->GetFeatureVector(&ks[0], descriptors.ptr<float>(0));
        }
        else std::cout<<"No SIFT found"<< std::endl;
    }
    else
        throw std::runtime_error ("SiftGPU:::Detect: SiftGPU Error!");

    //use indices_ to check if the keypoints and feature should be saved
    //compute SIFT keypoints and SIFT features
    //backproject sift keypoints to 3D and save in keypoints
    //save signatures

    scales.resize(ks.size());
    signatures.resize (ks.size (), std::vector<float>(128));

    for(size_t i=0; i < ks.size(); i++)
    {
        for (int k = 0; k < 128; k++)
            signatures[i][k] = descriptors.at<float>(i,k);

        scales[i] = ks[i].s;
    }
}

template class V4R_EXPORTS SIFTLocalEstimation<pcl::PointXYZRGB>;
}

#endif
