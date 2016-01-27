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
bool
SIFTLocalEstimation<PointT>::estimate(const pcl::PointCloud<PointT> & in, std::vector<std::vector<float> > &signatures)
{
    pcl::PointCloud<PointT> keypoints;
    std::vector<float> scales;
    return estimate(in, keypoints, signatures, scales);
}

template<typename PointT>
bool
SIFTLocalEstimation<PointT>::estimate (const pcl::PointCloud<PointT> & in, pcl::PointCloud<PointT> &processed, pcl::PointCloud<PointT> &keypoints, std::vector<std::vector<float> > &signatures)
{
    processed = in;
    std::vector<float> scales;
    return estimate(in, keypoints, signatures, scales);
}

template<typename PointT>
bool
SIFTLocalEstimation<PointT>::estimate (const pcl::PointCloud<PointT> & in, pcl::PointCloud<PointT> &keypoints, std::vector<std::vector<float> > &signatures, std::vector<float> & scales)
{
    cv::Mat colorImage = ConvertPCLCloud2Image(in);
    std::vector<SiftGPU::SiftKeypoint> ks;
    estimate(colorImage, ks, signatures, scales);

    keypoints.points.resize(ks.size());
    keypoint_indices_.resize(ks.size());

    std::vector<bool> obj_mask;
    if(indices_.empty())
        obj_mask.resize(in.width * in.height, true);
    else
        obj_mask = createMaskFromIndices(indices_, in.width * in.height);

    size_t kept = 0;
    for(size_t i=0; i < ks.size(); i++)
    {
        const int v = (int)(ks[i].y+.5);
        const int u = (int)(ks[i].x+.5);
        const int idx = v * in.width + u;

        if(u >= 0 && v >= 0 && u < in.width && v < in.height && pcl::isFinite(in.points[idx]) && obj_mask[idx])
        {
            keypoints.points[kept] = in.points[idx];
            keypoint_indices_[kept] = idx;
            scales[kept] = scales[i];
            signatures[kept] = signatures[i];
            kept++;
        }
    }

    keypoints.points.resize(kept);
    scales.resize(kept);
    keypoint_indices_.resize(kept);
    signatures.resize(kept);

    std::cout << "Number of SIFT features:" << kept << std::endl;
    indices_.clear();

    return true;
}


template<typename PointT>
bool
SIFTLocalEstimation<PointT>::estimate (const cv::Mat_ < cv::Vec3b > &colorImage, std::vector<SiftGPU::SiftKeypoint> & ks, std::vector<std::vector<float> > &signatures, std::vector<float> & scales)
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

    std::cout << "Number of SIFT features:" << ks.size() << std::endl;
    return true;
}

template class V4R_EXPORTS SIFTLocalEstimation<pcl::PointXYZRGB>;
}

#endif
