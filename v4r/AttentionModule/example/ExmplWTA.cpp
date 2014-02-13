#include "v4r/AttentionModule/AttentionModule.hpp"
#include "v4r/EPUtils/EPUtils.hpp"

#include <time.h>

#include <iostream>
#include <fstream>

int main(int argc, char** argv)
{
  srand ( time(NULL) );

  if(argc != 4)
  {
    std::cerr << "Usage: image saliency_map output" << std::endl;
    return(0);
  }
  
  std::string image_name(argv[1]);
  std::string map_name(argv[2]);
  std::string output_file_name(argv[3]);
  
  // read image
  cv::Mat image = cv::imread(image_name,-1);
  // read saliency map
  cv::Mat saliencyMap = cv::imread(map_name,0);
  saliencyMap.convertTo(saliencyMap,CV_32F,1.0/255);
    
  AttentionModule::Params wtaParams;
  AttentionModule::defaultParams(wtaParams);
  //modify wta
  wtaParams.useRandom = false;
  wtaParams.useCentering = false;
  wtaParams.useMorphologyOpenning = false;

  std::vector<cv::Point> attentionPoints;
  attentionPoints.clear();
  cv::Mat temp;
  image.copyTo(temp);
  temp.convertTo(temp,CV_32F,1.0/255);
  AttentionModule::CalculateWTA(temp,saliencyMap,attentionPoints,0,wtaParams);

  EPUtils::writeAttentionPoints(attentionPoints,output_file_name);
    
  //save image with attantion points
  //EPUtils::drawAttentionPoints(image,attentionPoints);
    
  //cv::imshow("attention points",image);
  //cv::waitKey();

  return 0;
}