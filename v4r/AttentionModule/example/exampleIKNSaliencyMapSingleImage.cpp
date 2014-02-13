#include <opencv2/opencv.hpp>

#include "v4r/AttentionModule/IKNSaliencyMap.hpp"

int main(int argc, char** argv)
{
  
  std::string images_name(argv[1]);
  
  std::cerr << "Image: " << images_name << std::endl;
    
  cv::Mat image = cv::imread(images_name,-1);
  
  // start creating parameters
  AttentionModule::IKNMapParameters parameters;
  image.copyTo(parameters.image);
  parameters.width = image.cols;
  parameters.height = image.rows;
      
  AttentionModule::CalculateIKNMap(parameters);
  
  cv::imshow("map",parameters.map);
  cv::waitKey();
  
  cv::Mat ikn_map;
  parameters.map.convertTo(ikn_map,CV_8U,255);
    
  std::string map_path = boost::filesystem::basename(images_name) + ".pgm";
    
  cv::imwrite(map_path,ikn_map);
    
  return(0);
}