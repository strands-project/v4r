#include <v4r_config.h>

#include <v4r/common/miscellaneous.h>
#include <v4r/common/pcl_opencv.h>
#include <glog/logging.h>

#ifdef HAVE_SIFTGPU
#include <v4r/features/sift_local_estimator.h>
#include <GL/glut.h>
#else
#include <v4r/features/opencv_sift_local_estimator.h>
#endif

namespace v4r
{

template<typename PointT>
void
SIFTLocalEstimation<PointT>::compute (std::vector<std::vector<float> > &signatures)
{
    CHECK( cloud_ && cloud_->isOrganized() );

    cv::Mat colorImage = ConvertPCLCloud2Image(*cloud_);
    Eigen::Matrix2Xf keypoints2d;
    compute(colorImage, keypoints2d, signatures);
    keypoint_indices_.resize(keypoints2d.cols());

    std::vector<bool> obj_mask;
    if(indices_.empty())
        obj_mask.resize(cloud_->width * cloud_->height, true);
    else
        obj_mask = createMaskFromIndices(indices_, cloud_->width * cloud_->height);

    size_t kept = 0;
    for(int i=0; i < keypoints2d.cols(); i++)
    {
        int u = std::max<int>(0, std::min<int>((int)cloud_->width  -1, keypoints2d(0,i) ) );
        int v = std::max<int>(0, std::min<int>((int)cloud_->height -1, keypoints2d(1,i) ) );

        int idx = v * cloud_->width + u;

        if(!obj_mask[idx]) // keypoint does not belong to given object mask
            continue;

        if( pcl::isFinite(cloud_->points[idx]) && cloud_->points[idx].z < max_distance_)
        {
            signatures[kept] = signatures[i];
            keypoint_indices_[kept] = idx;
            kept++;
        }
    }

    signatures.resize(kept);
    keypoint_indices_.resize(kept);
    indices_.clear();
    keypoints_.reset( new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*cloud_, keypoint_indices_, *keypoints_);
    processed_ = cloud_;
}


template<typename PointT>
void
SIFTLocalEstimation<PointT>::compute (const cv::Mat_ < cv::Vec3b > &colorImage, Eigen::Matrix2Xf &keypoints, std::vector<std::vector<float> > &signatures)
{
    cv::Mat grayImage;
    cv::cvtColor (colorImage, grayImage, CV_BGR2GRAY);

    cv::Mat descriptors;

    #ifdef HAVE_SIFTGPU
    if (sift_->CreateContextGL () != SiftGPU::SIFTGPU_FULL_SUPPORTED)
        throw std::runtime_error ("SiftGPU: No GL support!");

    sift_->VerifyContextGL();

    if(param_.dense_extraction_)
    {
        std::vector<SiftGPU::SiftKeypoint> dense_ks;
        for(int u=0; u<colorImage.cols; u++)
        {
            for(int v=0; v<colorImage.rows; v++)
            {
                if( u%param_.stride_ == 0 && v%param_.stride_ == 0 )
                {
                    SiftGPU::SiftKeypoint kp;
                    kp.x = u;
                    kp.y = v;
                    dense_ks.push_back(kp);
                }
            }
        }
        SiftGPU::SiftKeypoint *ksp = new SiftGPU::SiftKeypoint[ dense_ks.size() ];
        for(size_t i=0; i<dense_ks.size(); i++)
            ksp[i] = dense_ks[i];
        sift_->SetKeypointList(dense_ks.size(), ksp, false);
        delete[] ksp;
    }

    if (sift_->RunSIFT (grayImage.cols, grayImage.rows, grayImage.ptr<uchar> (0), GL_LUMINANCE, GL_UNSIGNED_BYTE))
    {
        int num = sift_->GetFeatureNum();
        if (num>0)
        {
            std::vector<SiftGPU::SiftKeypoint> ks (num);
            descriptors = cv::Mat(num,128,CV_32F);
            sift_->GetFeatureVector(&ks[0], descriptors.ptr<float>(0));
            keypoints = Eigen::Matrix2Xf(2, ks.size());
    		signatures.resize (ks.size (), std::vector<float>(128));

    		for(size_t i=0; i < ks.size(); i++)
            {
                const SiftGPU::SiftKeypoint &kp = ks[i];

                for (int k = 0; k < 128; k++)
                    signatures[i][k] = descriptors.at<float>(i,k);

                keypoints(0,i) = kp.x;
                keypoints(1,i) = kp.y;
    		}
        }
        else
            std::cout << "No SIFT features found!" << std::endl;
    }
    else
    {
        throw std::runtime_error ("SiftGPU:::Detect: SiftGPU Error!");
        return;
    }
#else
    std::vector<cv::KeyPoint> ks;
    sift_->operator ()(grayImage, cv::Mat(), ks, descriptors, false);

    int num = ks.size();

    if(num>0)
    {
        keypoints = Eigen::Matrix2Xf(2, ks.size());
        signatures.resize (ks.size (), std::vector<float>(128));

        for(size_t i=0; i < ks.size(); i++)
        {
            for (int k = 0; k < 128; k++)
                signatures[i][k] = descriptors.at<float>(i,k);

          keypoints(0,i) = ks[i].pt.x;
          keypoints(1,i) = ks[i].pt.y;
        }
    }
    else
        std::cout << "No SIFT features found!" << std::endl;
#endif
}

template class V4R_EXPORTS SIFTLocalEstimation<pcl::PointXYZRGB>;
}
