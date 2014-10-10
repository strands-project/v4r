#include "v4r/EPUtils/EPUtils.hpp"

#include <iostream>

int main(int argc, char** argv)
{
  if(argc != 5)
  {
    std::cerr << "Usage: XYZRGB disparity width height" << std::endl;
    return(0);
  }
  
  std::string XYZRGB_name(argv[1]);
  std::string disparity_name(argv[2]);
  int width = atoi(argv[3]);
  int height = atoi(argv[4]);
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (XYZRGB_name,*cloud) == -1)
  {
    printf("[ERROR]: Couldn't read point cloud.\n");
    return -1;
  }
  
  cv::Mat disparity_image;
  EPUtils::pointCloud_2_disparity(disparity_image,cloud,width,height,pcl::PointIndices::Ptr(new pcl::PointIndices()),525,0.075,0.05);
  
  disparity_image.convertTo(disparity_image,CV_8U,1.0);
  
  cv::Mat disparity_image2, disparity_image_mask;
  disparity_image_mask = cv::Mat_<uchar>::zeros(disparity_image.rows,disparity_image.cols);
  
  for(int i = 0; i < disparity_image.rows; ++i)
  {
    for(int j = 0; j < disparity_image.cols; ++j)
    {
      if(disparity_image.at<uchar>(i,j) >= 255)
	disparity_image_mask.at<uchar>(i,j) = 1;
    }
  }
  
//   cv::imshow("disparity_image",disparity_image);
//   cv::imshow("disparity_image_mask",disparity_image_mask*255);
  
  cv::inpaint(disparity_image,disparity_image_mask,disparity_image2,5,cv::INPAINT_TELEA);
  
//   cv::imshow("disparity_image2",disparity_image2);
//   cv::waitKey(-1);

  cv::imwrite(disparity_name,disparity_image2);

  return 0;
}