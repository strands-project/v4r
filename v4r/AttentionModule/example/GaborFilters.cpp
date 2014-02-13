#include <opencv2/opencv.hpp>

#include "v4r/AttentionModule/AttentionModule.hpp"
#include "v4r/EPUtils/EPUtils.hpp"

#include <iostream>

int main(int argc, char** argv)
{
  
  // read image
  std::string image_name(argv[1]);
  cv::Mat image = cv::imread(image_name,-1);
  
  //cv::resize(image,image,cv::Size(0,0),0.5,0.5,cv::INTER_LINEAR);
  
  /*image.convertTo(image,CV_32F,1.0f/255);
  cv::Mat image_gray;
  cv::cvtColor(image,image_gray,CV_BGR2GRAY);
  
  float angle = 0;
  cv::Mat gaborKernel;
  EPUtils::makeGaborKernel2D(gaborKernel,90,2);
  cv::Scalar s = sum(gaborKernel);
  std::cerr << "sum = " << s(0) << std::endl;
  cv::Mat filter0,filter90;
  EPUtils::makeGaborFilter(filter0,filter90,0);
    
  cv::Mat temp;
  cv::filter2D(image_gray,temp,-1,gaborKernel);
  cv::Mat temp2;
  cv::filter2D(image_gray,temp2,-1,filter0);
  cv::Mat temp_abs;
  double min, max;
  minMaxLoc(temp,&min,&max);
  std::cerr << "min = " << min << std::endl;
  std::cerr << "max = " << max << std::endl;
  std::cerr << "value(10,10) = " << temp.at<float>(10,10) << std::endl;
  temp_abs = cv::abs(temp);
  //EPUtils::normalize(temp_abs);
  temp2 = cv::abs(temp2);
  EPUtils::normalize(temp2);
  cv::imshow("temp",temp_abs);
  //cv::imshow("temp2",temp2);
  
  cv::resize(gaborKernel,gaborKernel,cv::Size(0,0),3,3,cv::INTER_LINEAR);
  gaborKernel = gaborKernel * 255;
  cv::imshow("gaborKernel",gaborKernel);
  
  cv::waitKey();*/
  
  AttentionModule::OrientationSaliencyMap orientationSaliencyMap;
  orientationSaliencyMap.setImage(image);
  orientationSaliencyMap.setAngle(0);
  orientationSaliencyMap.setBandwidth(2);
  orientationSaliencyMap.setWidth(image.cols);
  orientationSaliencyMap.setHeight(image.rows);
  
  cv::Mat map;
  
  orientationSaliencyMap.calculateOrientationMap(map);
  
  cv::imshow("map",map);
  cv::waitKey();
  
  return(0);
}