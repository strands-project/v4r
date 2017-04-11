#include <v4r_config.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/common/pcl_opencv.h>
#include <pcl/common/io.h>
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

    PCLOpenCVConverter<PointT> pcl_opencv_converter;
    pcl_opencv_converter.setInputCloud(cloud_);
    const cv::Mat colorImage = pcl_opencv_converter.getRGBImage();
    Eigen::Matrix2Xf keypoints2d;
    compute(colorImage, keypoints2d, signatures);

    size_t kept = 0;
    for(size_t i=0; i < keypoint_indices_.size(); i++)
    {
        int idx = keypoint_indices_[i];

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
}


template<typename PointT>
void
SIFTLocalEstimation<PointT>::compute (const cv::Mat_ < cv::Vec3b > &colorImage, Eigen::Matrix2Xf &keypoints, std::vector<std::vector<float> > &signatures)
{
    cv::Mat grayImage;
    cv::cvtColor (colorImage, grayImage, CV_BGR2GRAY);

    boost::dynamic_bitset<> obj_mask;
    if(indices_.empty())
    {
        obj_mask.resize(colorImage.cols * colorImage.rows);
        obj_mask.set();
    }
    else
        obj_mask = createMaskFromIndices(indices_, colorImage.cols * colorImage.rows);


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
            cv::Mat descriptors(num,128,CV_32F);
            sift_->GetFeatureVector(&ks[0], descriptors.ptr<float>(0));
            keypoints = Eigen::Matrix2Xf(2, ks.size());
            signatures.resize (ks.size (), std::vector<float>(128));

            keypoint_indices_.resize( ks.size() );
            size_t kept = 0;
            for(size_t i=0; i < ks.size(); i++)
            {
                const SiftGPU::SiftKeypoint &kp = ks[i];
                int u = std::min<int>( colorImage.cols -1, kp.x+0.5f );
                int v = std::min<int>( colorImage.rows -1, kp.y+0.5f );
                int idx = v * colorImage.cols + u;

                if( obj_mask[idx] ) // keypoint does not belong to given object mask
                {
                    if( param_.use_rootSIFT_ )
                    {
                        double norm_L1 = cv::norm(descriptors.row(i), cv::NORM_L1);

                        for (size_t k = 0; k < 128; k++)
                            descriptors.at<float>(i,k) = sqrt( descriptors.at<float>(i,k) / norm_L1 );

//                        double norm_L2 = cv::norm( descriptors.row(i) );
//                        descriptors.row(i) /= norm_L2;
                    }

                    for (size_t k = 0; k < 128; k++)
                        signatures[kept][k] = descriptors.at<float>(i,k);

                    keypoints(0,kept) = kp.x;
                    keypoints(1,kept) = kp.y;
                    keypoint_indices_[kept] = idx;
                    kept++;
                }
            }
            signatures.resize(kept);
            keypoints.conservativeResize(2, kept);
            keypoint_indices_.resize( kept );
        }
        else
        {
            LOG(WARNING) << "No SIFT features found!";
            keypoint_indices_.clear();
        }
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
    if (num>0)
    {
        std::vector<SiftGPU::SiftKeypoint> ks (num);
        cv::Mat descriptors(num,128,CV_32F);
        sift_->GetFeatureVector(&ks[0], descriptors.ptr<float>(0));
        keypoints = Eigen::Matrix2Xf(2, ks.size());
        signatures.resize (ks.size (), std::vector<float>(128));

        keypoint_indices_.resize( ks.size() );
        size_t kept = 0;
        for(size_t i=0; i < ks.size(); i++)
        {
            int u = std::min<int>( colorImage.cols -1, ks[i].pt.x+0.5f );
            int v = std::min<int>( colorImage.rows -1, ks[i].pt.y+0.5f );
            int idx = v * colorImage.cols + u;

            if( obj_mask[idx] ) // keypoint does not belong to given object mask
            {
                if( param_.use_rootSIFT_ )
                {
                    double norm_L1 = cv::norm(descriptors.row(i), cv::NORM_L1);

                    for (size_t k = 0; k < 128; k++)
                        descriptors.at<float>(i,k) = sqrt( descriptors.at<float>(i,k) / norm_L1 );

//                        double norm_L2 = cv::norm( descriptors.row(i) );
//                        descriptors.row(i) /= norm_L2;
                }

                for (size_t k = 0; k < 128; k++)
                    signatures[kept][k] = descriptors.at<float>(i,k);

                keypoints(0,kept) = ks[i].pt.x;
                keypoints(1,kept) = ks[i].pt.y;
                keypoint_indices_[kept] = idx;
                kept++;
            }
        }
        signatures.resize(kept);
        keypoints.conservativeResize(2, kept);
        keypoint_indices_.resize( kept );
    }
#endif
}


#ifdef HAVE_SIFTGPU
template<typename PointT>
std::vector<std::pair<int, int> >
SIFTLocalEstimation<PointT>::matchSIFT( const std::vector<std::vector<float> >& desc1, const std::vector<std::vector<float> >& desc2)
{
    SiftMatchGPU matcher(4096 * 4);
    matcher.VerifyContextGL();

    std::vector<float> desc_1 ( desc1.size() * 128);
    std::vector<float> desc_2 ( desc2.size() * 128);
    for(size_t i=0; i<desc1.size(); i++)
    {
        for(size_t j=0; j<128; j++)
            desc_1[i*128+j] = desc1[i][j];
    }

    for(size_t i=0; i<desc2.size(); i++)
    {
        for(size_t j=0; j<128; j++)
            desc_2[i*128+j] = desc2[i][j];
    }

    matcher.SetDescriptors(0, desc1.size(), &desc_1[0]); //image 1
    matcher.SetDescriptors(1, desc2.size(), &desc_2[0]); //image 2

    //match and get result.
    int (*match_buf)[2] = new int[desc1.size()][2];

//        int num_match = matcher->GetSiftMatch(num1, match_buf,0.75, 0.8, 1);
    int num_match = matcher.GetSiftMatch(desc1.size(), match_buf, 0.5, 0.95, 1);

    std::vector<std::pair<int, int> > matches(num_match);
    for(int j = 0; j < num_match; j++)
        matches[j] = std::pair<int, int>( match_buf[j][0], match_buf[j][1] );

    delete[] match_buf;
    return matches;
}
#endif

template class V4R_EXPORTS SIFTLocalEstimation<pcl::PointXYZRGB>;
}
