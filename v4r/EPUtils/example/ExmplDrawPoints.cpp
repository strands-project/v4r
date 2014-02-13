#include "v4r/EPUtils/EPUtils.hpp"

#include <iostream>

int main(int argc, char** argv)
{
  if(argc != 4)
  {
    std::cerr << "Usage: attention_points_list image new_image" << std::endl;
    return(0);
  }
  
  std::string attention_points_file_name(argv[1]);
  std::string image_name(argv[2]);
  std::string new_image_name(argv[3]);
  
  // read attention points
  std::vector<cv::Point> attentionPoints;
  EPUtils::readAttentionPoints(attentionPoints,attention_points_file_name);
  // read labeling
  cv::Mat image = cv::imread(image_name,-1);
    
  EPUtils::drawAttentionPoints(image,attentionPoints);
  
  cv::imwrite(new_image_name,image);

  return 0;
}