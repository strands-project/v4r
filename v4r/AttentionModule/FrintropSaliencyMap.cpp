#include "FrintropSaliencyMap.hpp"

namespace AttentionModule
{

FrintropMapParameters::FrintropMapParameters()
{
  image = cv::Mat_<float>::zeros(0,0);
  R = cv::Mat_<float>::zeros(0,0);
  G = cv::Mat_<float>::zeros(0,0);
  B = cv::Mat_<float>::zeros(0,0);
  Y = cv::Mat_<float>::zeros(0,0);
  I = cv::Mat_<float>::zeros(0,0);
  normalization_type = EPUtils::NT_FRINTROP_NORM;
  width = 0;
  height = 0;
  numberOfOrientations = 4;
  // on intensity
  pyramidIOn.combination_type = AM_FRINTROP;
  pyramidIOn.start_level = 2;
  pyramidIOn.max_level = 4;
  pyramidIOn.normalization_type = EPUtils::NT_FRINTROP_NORM;
  pyramidIOn.R.resize(2);
  pyramidIOn.R.at(0)= 3;
  pyramidIOn.R.at(1) = 7;
  pyramidIOn.onSwitch = true;
  // off intensity
  pyramidIOff.combination_type = AM_FRINTROP;
  pyramidIOff.start_level = 2;
  pyramidIOff.max_level = 4;
  pyramidIOff.normalization_type = EPUtils::NT_FRINTROP_NORM;
  pyramidIOff.R.resize(2);
  pyramidIOff.R.at(0)= 3;
  pyramidIOff.R.at(1) = 7;
  pyramidIOff.onSwitch = false;
  // R pyramid
  pyramidR.combination_type = AM_FRINTROP;
  pyramidR.start_level = 2;
  pyramidR.max_level = 4;
  pyramidR.normalization_type = EPUtils::NT_FRINTROP_NORM;
  pyramidR.R.resize(2);
  pyramidR.R.at(0)= 3;
  pyramidR.R.at(1) = 7;
  pyramidR.onSwitch = true;
  // G pyramid
  pyramidG.combination_type = AM_FRINTROP;
  pyramidG.start_level = 2;
  pyramidG.max_level = 4;
  pyramidG.normalization_type = EPUtils::NT_FRINTROP_NORM;
  pyramidG.R.resize(2);
  pyramidG.R.at(0)= 3;
  pyramidG.R.at(1) = 7;
  pyramidG.onSwitch = true;
  // B pyramid
  pyramidB.combination_type = AM_FRINTROP;
  pyramidB.start_level = 2;
  pyramidB.max_level = 4;
  pyramidB.normalization_type = EPUtils::NT_FRINTROP_NORM;
  pyramidB.R.resize(2);
  pyramidB.R.at(0)= 3;
  pyramidB.R.at(1) = 7;
  pyramidB.onSwitch = true;
  // Y pyramid
  pyramidY.combination_type = AM_FRINTROP;
  pyramidY.start_level = 2;
  pyramidY.max_level = 4;
  pyramidY.normalization_type = EPUtils::NT_FRINTROP_NORM;
  pyramidY.R.resize(2);
  pyramidY.R.at(0)= 3;
  pyramidY.R.at(1) = 7;
  pyramidY.onSwitch = true;
  // orientations
  pyramidO.resize(numberOfOrientations);
  for(int i = 0; i < numberOfOrientations; ++ i)
  {
    pyramidO.at(i).combination_type = AM_SIMPLE;
    pyramidO.at(i).start_level = 2;
    pyramidO.at(i).max_level = 4;
    pyramidO.at(i).normalization_type = EPUtils::NT_FRINTROP_NORM;
  }
  cv::Mat map = cv::Mat_<float>::zeros(0,0);
}

int CalculateFrintropMap(FrintropMapParameters &parameters)
{
  if((( (parameters.width == 0) || (parameters.height == 0) ) && ( (parameters.map.rows == 0) || (parameters.map.cols == 0))) ||
     (  (parameters.image.rows == 0) || (parameters.image.cols == 0) ))
  {
    return(AM_IMAGE);
  }

  if((parameters.width == 0) || (parameters.height == 0))
  {
    parameters.height = parameters.map.rows;
    parameters.width  = parameters.map.cols;
  }
  
  if((parameters.image.cols != parameters.width) || (parameters.image.rows != parameters.height) || (parameters.image.channels() != 3))
  {
    return(AM_IMAGE);
  }

  // on intensity
  parameters.pyramidIOn.width = parameters.width;
  parameters.pyramidIOn.height = parameters.height;
  
  // off intensity
  parameters.pyramidIOff.width = parameters.width;
  parameters.pyramidIOff.height = parameters.height;
  
  // orientations
  parameters.pyramidO.clear();
  parameters.pyramidO.resize(parameters.numberOfOrientations);
  for(int i = 0; i < parameters.numberOfOrientations; ++i)
  {
    parameters.pyramidO.at(i).width = parameters.width;
    parameters.pyramidO.at(i).height = parameters.height;
    parameters.pyramidO.at(i).combination_type = AM_SIMPLE;
    parameters.pyramidO.at(i).start_level = 2;
    parameters.pyramidO.at(i).max_level = 4;
    parameters.pyramidO.at(i).normalization_type = EPUtils::NT_FRINTROP_NORM;
  }
  
  // R channel
  parameters.pyramidR.width = parameters.width;
  parameters.pyramidR.height = parameters.height;
  // G channel
  parameters.pyramidG.width = parameters.width;
  parameters.pyramidG.height = parameters.height;
  // B channel
  parameters.pyramidB.width = parameters.width;
  parameters.pyramidB.height = parameters.height;
  // Y channels
  parameters.pyramidY.width = parameters.width;
  parameters.pyramidY.height = parameters.height;
  
  CreateColorChannels(parameters);
  
  createFeatureMaps(parameters);
  
  float maxIntensityValue = std::max(parameters.pyramidIOn.max_map_value,parameters.pyramidIOff.max_map_value);
  cv::Mat intensity = parameters.pyramidIOn.map + parameters.pyramidIOff.map;
  EPUtils::normalize(intensity,EPUtils::NT_NONE,maxIntensityValue);
  EPUtils::normalize(intensity,parameters.normalization_type);
  
  cv::Mat orientation = cv::Mat_<float>::zeros(intensity.rows,intensity.cols);
  float maxOrientationValue = 0;
  for(int i = 0; i < parameters.numberOfOrientations; ++i)
  {
    maxOrientationValue = std::max(maxOrientationValue,parameters.pyramidO.at(i).max_map_value);
    orientation = orientation + parameters.pyramidO.at(i).map;
  }
  EPUtils::normalize(orientation,EPUtils::NT_NONE,maxOrientationValue);
  EPUtils::normalize(orientation,parameters.normalization_type);
  
  parameters.map = intensity + orientation;
  
  if(parameters.image.channels() > 1)
  {
    float maxColorValue = std::max(parameters.pyramidR.max_map_value,parameters.pyramidG.max_map_value);
    maxColorValue = std::max(maxColorValue,parameters.pyramidB.max_map_value);
    maxColorValue = std::max(maxColorValue,parameters.pyramidY.max_map_value);
    cv::Mat color = parameters.pyramidR.map + parameters.pyramidG.map + parameters.pyramidB.map + parameters.pyramidY.map;
    EPUtils::normalize(color,EPUtils::NT_NONE,maxColorValue);
    EPUtils::normalize(color,parameters.normalization_type);
    parameters.map = parameters.map + color;
  }
  
  //EPUtils::normalize(parameters.map);

  return(AM_OK);
}

void CreateColorChannels(FrintropMapParameters &parameters)
{
  if(parameters.image.channels() > 1)
    cv::cvtColor(parameters.image,parameters.I,CV_RGB2GRAY);
  else
    parameters.image.copyTo(parameters.I);
  
  parameters.I.convertTo(parameters.I,CV_32F,1.0f/255);
  
  if(parameters.image.channels() > 1)
  {
    ColorSaliencyMap colorSaliencyMap;
    colorSaliencyMap.setImage(parameters.image);
    colorSaliencyMap.setWidth(parameters.image.cols);
    colorSaliencyMap.setHeight(parameters.image.rows);
    colorSaliencyMap.setUseLAB(true);
    // red
    colorSaliencyMap.setColor(cv::Scalar(255,127));
    colorSaliencyMap.calculateColorMap(parameters.R);
    // green
    colorSaliencyMap.setColor(cv::Scalar(0,127));
    colorSaliencyMap.calculateColorMap(parameters.G);
    // blue
    colorSaliencyMap.setColor(cv::Scalar(127,0));
    colorSaliencyMap.calculateColorMap(parameters.B);
    // yellow
    colorSaliencyMap.setColor(cv::Scalar(127,255));
    colorSaliencyMap.calculateColorMap(parameters.Y);
  }
}

void createFeatureMaps(FrintropMapParameters &parameters)
{
  // intensity pyramid
  cv::buildPyramid(parameters.I,parameters.pyramidIOn.pyramidImages,parameters.pyramidIOn.max_level);
  combinePyramid(parameters.pyramidIOn);
  
  cv::buildPyramid(parameters.I,parameters.pyramidIOff.pyramidImages,parameters.pyramidIOff.max_level);
  combinePyramid(parameters.pyramidIOff);
    
  // orientation pyramids
  for (int o4 = 0; o4 < parameters.numberOfOrientations; ++o4)
  {
    float angle = o4*180.0/parameters.numberOfOrientations;
    cv::Mat gaborKernel0, gaborKernel90;
    EPUtils::makeGaborFilter(gaborKernel0,gaborKernel90,angle);
    
    parameters.pyramidO.at(o4).pyramidImages.resize(parameters.pyramidIOn.pyramidImages.size());
    parameters.pyramidO.at(o4).pyramidFeatures.resize(parameters.pyramidIOn.pyramidFeatures.size());
    
    for(unsigned int i = 0; i < parameters.pyramidIOn.pyramidImages.size(); ++i)
    {
      cv::Mat temp0, temp90;
      cv::filter2D(parameters.pyramidIOn.pyramidImages.at(i),temp0,-1,gaborKernel0);
      temp0 = cv::abs(temp0);
      cv::filter2D(parameters.pyramidIOn.pyramidImages.at(i),temp90,-1,gaborKernel90);
      temp90 = cv::abs(temp90);
      cv::add(temp0,temp90,parameters.pyramidO.at(o4).pyramidImages.at(i));
      parameters.pyramidO.at(o4).pyramidImages.at(i).copyTo(parameters.pyramidO.at(o4).pyramidFeatures.at(i));
    }
    
    combinePyramid(parameters.pyramidO.at(o4));
  }

  // color pyramids
  if(parameters.image.channels() > 1)
  {
    cv::buildPyramid(parameters.R,parameters.pyramidR.pyramidImages,parameters.pyramidR.max_level);
    cv::buildPyramid(parameters.G,parameters.pyramidG.pyramidImages,parameters.pyramidG.max_level);
    cv::buildPyramid(parameters.B,parameters.pyramidB.pyramidImages,parameters.pyramidB.max_level);
    cv::buildPyramid(parameters.Y,parameters.pyramidY.pyramidImages,parameters.pyramidY.max_level);
    
    combinePyramid(parameters.pyramidR);
    combinePyramid(parameters.pyramidG);
    combinePyramid(parameters.pyramidB);
    combinePyramid(parameters.pyramidY);
  }
}

} //AttentionModule