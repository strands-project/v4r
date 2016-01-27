#include <v4r_config.h>
#ifndef HAVE_SIFTGPU
#include <v4r/features/opencv_sift_local_estimator.h>

namespace v4r
{

template<typename PointT>
bool
OpenCVSIFTLocalEstimation<PointT>::estimate (const pcl::PointCloud<PointT> & in, pcl::PointCloud<PointT> & processed, pcl::PointCloud<PointT> & keypoints, std::vector<std::vector<float> > & signatures)
{
  keypoint_indices_.clear();
  if(indices_.empty())
  {
    indices_.resize(in.points.size());
    for(size_t i=0; i < indices_.size(); i++)
        indices_[i] = i;
  }

  std::vector<bool> mask(in.points.size(), false);

  for(size_t i=0; i < indices_.size(); i++)
      mask[indices_[i]] = 1;

  cv::Mat colorImage = ConvertPCLCloud2Image (in);
  cv::Mat grayImage;
  cv::cvtColor (colorImage, grayImage, CV_BGR2GRAY);

  cv::Mat descriptors;
  std::vector<cv::KeyPoint> ks;

  sift_->operator ()(grayImage, cv::Mat(), ks, descriptors, false);

  //use indices_ to check if the keypoints and feature should be saved
  //compute SIFT keypoints and SIFT features
  //backproject sift keypoints to 3D and save in keypoints
  //save signatures
  signatures.resize (ks.size (), std::vector<float>(128));
  keypoints.points.resize(ks.size());
  size_t kept = 0;
  for(size_t i=0; i < ks.size(); i++)
  {
    int u,v;
    v = (int)(ks[i].pt.y+.5);
    u = (int)(ks[i].pt.x+.5);

    int idx = v*in.width + u;

    if(u >= 0 && v >= 0 && u < in.width && v < in.height && mask[idx] && pcl::isFinite(in.points[idx]) )
    {
        keypoints.points[kept] = in.points[idx];
        keypoint_indices_.push_back(idx);

        for (int k = 0; k < 128; k++)
          signatures[kept][k] = descriptors.at<float>(i,k);

        kept++;
    }
  }

  signatures.resize(kept);
  keypoints.points.resize(kept);
  pcl::copyPointCloud(in, indices_, processed);
  indices_.clear();
  std::cout << "Number of SIFT features:" << kept << std::endl;

  return true;
}

template class V4R_EXPORTS OpenCVSIFTLocalEstimation<pcl::PointXYZRGB>;
}

#endif
