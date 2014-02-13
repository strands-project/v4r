#include <time.h>

#include "v4r/EPUtils/EPUtils.hpp"
#include "v4r/AttentionModule/AttentionModule.hpp"

int main(int argc, char** argv)
{
  if(argc != 4)
  {
    std::cerr << "Usage: image saliency_map output" << std::endl;
    return(0);
  }
  
  srand ( time(NULL) );
  
  std::string image_name(argv[1]);
  std::string saliency_map_name(argv[2]);
  std::string output_file_name(argv[3]);
    
  cv::Mat image = cv::imread(image_name,-1);
  cv::Mat map = cv::imread(saliency_map_name,-1);
  map.convertTo(map,CV_32F,1.0f/255);
    
  std::vector<EPUtils::ConnectedComponent> connectedComponents;
    
  float th = 0.1;
  EPUtils::extractConnectedComponents(map,connectedComponents,th);
  EPUtils::drawConnectedComponents(connectedComponents,image,cv::Scalar(255,0,0));
  //std::cerr << "Number of connected components: " << connectedComponents.size() << std::endl;
    
  std::vector<AttentionModule::SaliencyLine> saliencyLine;
  cv::Mat points_image = cv::Mat_<uchar>::zeros(image.rows,image.cols);
  std::vector<AttentionModule::PointSaliency> saliencyPoints;
    
  for(unsigned int i = 0; i < connectedComponents.size(); ++ i)
  {
    cv::Mat mask = cv::Mat_<uchar>::zeros(image.rows,image.cols);
    EPUtils::drawConnectedComponent(connectedComponents.at(i),mask,cv::Scalar(1));
    //cv::imshow("mask",255*mask);
    //cv::waitKey();
      
    AttentionModule::SaliencyLine saliencyLineCurent;
    AttentionModule::PointSaliency pointSaliencyCurrent;
    if(AttentionModule::extractSaliencyLine(mask,map,saliencyLineCurent))
    {
      //std::vector<cv::Point> saliencyLineCurent_points;
      //AttentionModule::createSimpleLine(saliencyLineCurent,saliencyLineCurent_points);
      //std::cerr << saliencyLineCurent_points.size() << std::endl;
      //EPUtils::drawAttentionPoints(image,saliencyLineCurent_points);
      //cv::imshow("skeleton",image);
      //cv::waitKey();
      
      saliencyLine.push_back(saliencyLineCurent);
      AttentionModule::selectSaliencyCenterPoint(saliencyLineCurent,pointSaliencyCurrent);
      
      saliencyPoints.push_back(pointSaliencyCurrent);
    }
  }
    
  std::sort(saliencyPoints.begin(),saliencyPoints.end(),AttentionModule::saliencyPointsSort);
    
  std::vector<cv::Point> attentionPoints;
  AttentionModule::createAttentionPoints(saliencyPoints,attentionPoints);
  
  EPUtils::writeAttentionPoints(attentionPoints,output_file_name);
  
  //EPUtils::drawAttentionPoints(image,attentionPoints);
  
  //cv::imshow("attention points", image);
  //cv::imshow("saliency map", map);
  //cv::waitKey();
}