#include "IKNSaliencyMap.hpp"

namespace AttentionModule
{

IKNMapParameters::IKNMapParameters()
{
  image = cv::Mat_<float>::zeros(0,0);
  R = cv::Mat_<float>::zeros(0,0);
  G = cv::Mat_<float>::zeros(0,0);
  B = cv::Mat_<float>::zeros(0,0);
  Y = cv::Mat_<float>::zeros(0,0);
  I = cv::Mat_<float>::zeros(0,0);
  normalization_type = EPUtils::NT_NONMAX;
  width = 0;
  height = 0;
  weightOfColor = 1;
  weightOfIntensities = 1;
  weightOfOrientations = 1;
  numberOfOrientations = 4;
  pyramidI.combination_type = AM_ITTI;
  pyramidI.start_level = 0;
  pyramidI.max_level = 8;
  pyramidI.normalization_type = EPUtils::NT_NONMAX;
  pyramidRG.combination_type = AM_ITTI;
  pyramidRG.start_level = 0;
  pyramidRG.max_level = 8;
  pyramidRG.normalization_type = EPUtils::NT_NONMAX;
  pyramidBY.combination_type = AM_ITTI;
  pyramidBY.start_level = 0;
  pyramidBY.max_level = 8;
  pyramidBY.normalization_type = EPUtils::NT_NONMAX;
  pyramidO.resize(numberOfOrientations);
  for(int i = 0; i < numberOfOrientations; ++ i)
  {
    pyramidO.at(i).combination_type = AM_ITTI;
    pyramidO.at(i).start_level = 0;
    pyramidO.at(i).max_level = 8;
    pyramidO.at(i).normalization_type = EPUtils::NT_NONMAX;
  }
  
  cv::Mat map = cv::Mat_<float>::zeros(0,0);
}

int CalculateIKNMap(IKNMapParameters &parameters)
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

  if((parameters.weightOfColor == 0) || (parameters.weightOfIntensities  == 0) || (parameters.weightOfOrientations == 0))
  {
    return(AM_PARAMETERS);
  }

  // set intenstity pyramid
  parameters.pyramidI.width = parameters.width;
  parameters.pyramidI.height = parameters.height;
  checkLevels(parameters.pyramidI);
  // set RG pyramid
  parameters.pyramidRG.width = parameters.width;
  parameters.pyramidRG.height = parameters.height;
  checkLevels(parameters.pyramidRG);
  // set BY pyramid
  parameters.pyramidBY.width = parameters.width;
  parameters.pyramidBY.height = parameters.height;
  checkLevels(parameters.pyramidBY);
  // set orientation pyramid
  parameters.pyramidO.clear();
  parameters.pyramidO.resize(parameters.numberOfOrientations);
  for(int i = 0; i < parameters.numberOfOrientations; ++i)
  {
    parameters.pyramidO.at(i).width = parameters.width;
    parameters.pyramidO.at(i).height = parameters.height;
    parameters.pyramidO.at(i).combination_type = AM_ITTI;
    parameters.pyramidO.at(i).start_level = 0;
    parameters.pyramidO.at(i).max_level = 8;
    parameters.pyramidO.at(i).normalization_type = EPUtils::NT_NONMAX;
    checkLevels(parameters.pyramidO.at(i));
  }
  
  // create channels
  CreateColorChannels(parameters);
  
  // create feature maps
  createFeatureMaps(parameters);

  float totalWeight = parameters.weightOfColor + parameters.weightOfIntensities + parameters.weightOfOrientations;
  
  cv::Mat intensity = parameters.weightOfIntensities*parameters.pyramidI.map/totalWeight;
  
  cv::Mat orientation = cv::Mat_<float>::zeros(intensity.rows,intensity.cols);
  
  for(int i = 0; i < parameters.numberOfOrientations; ++i)
  {
    orientation = orientation + parameters.pyramidO.at(i).map;
  }
  EPUtils::normalize(orientation,parameters.normalization_type);
  orientation = parameters.weightOfOrientations*orientation/totalWeight;
  
  cv::Mat color;
  color = parameters.pyramidRG.map + parameters.pyramidBY.map;
  EPUtils::normalize(color,parameters.normalization_type);
  color = parameters.weightOfColor*color/totalWeight;
  
  parameters.map = intensity + color + orientation;

  return(AM_OK);
}

void CreateColorChannels(IKNMapParameters &parameters)
{
  parameters.I = cv::Mat_<float>::zeros(parameters.height,parameters.width);
  parameters.R = cv::Mat_<float>::zeros(parameters.height,parameters.width);
  parameters.G = cv::Mat_<float>::zeros(parameters.height,parameters.width);
  parameters.B = cv::Mat_<float>::zeros(parameters.height,parameters.width);
  parameters.Y = cv::Mat_<float>::zeros(parameters.height,parameters.width);

  float rr,gg,bb;
  float Imax = 0;
  
  for(int r = 0; r < parameters.height; ++r)
  {
    for (int c = 0; c < parameters.width; c++)
    {
      rr = parameters.image.at<uchar>(r,3*c+2);
      gg = parameters.image.at<uchar>(r,3*c+1);
      bb = parameters.image.at<uchar>(r,3*c+0);
      
      rr /= 255;
      gg /= 255;
      bb /= 255;
      
      if (Imax < (rr+gg+bb)/3)
      {
        Imax = (rr+gg+bb)/3;
      }
    }
  }

  for(int r = 0; r < parameters.height; ++r)
  {
    for (int c = 0; c < parameters.width; c++)
    {
      rr = parameters.image.at<uchar>(r,3*c+2);
      gg = parameters.image.at<uchar>(r,3*c+1);
      bb = parameters.image.at<uchar>(r,3*c+0);
      
      rr /= 255;
      gg /= 255;
      bb /= 255;
      
      float dI = (rr+gg+bb)/3;
      float dR = rr-(gg+bb)/2;
      float dG = gg-(rr+bb)/2;
      float dB = bb-(gg+rr)/2;
      float dY = (rr+gg)/2-(rr-gg>0? rr-gg:gg-rr)/2-bb;
      
      if (dI <= 0.1*Imax)
      {
        dI = 0;
        dR = 0;
        dG = 0;
        dB = 0;
        dY = 0;
      }
      else
      {
        dR /= dI;
        dG /= dI;
        dB /= dI;
        dY /= dI;
      }

      parameters.I.at<float>(r,c) = dI;
      parameters.R.at<float>(r,c) = dR;
      parameters.G.at<float>(r,c) = dG;
      parameters.B.at<float>(r,c) = dB;
      parameters.Y.at<float>(r,c) = dY;
    }
  }
}

void createFeatureMaps(IKNMapParameters &parameters)
{
  // intensity pyramid
  cv::buildPyramid(parameters.I,parameters.pyramidI.pyramidImages,parameters.pyramidI.max_level);
  combinePyramid(parameters.pyramidI);
    
  // orientation pyramids
  for (int o4 = 0; o4 < parameters.numberOfOrientations; ++o4)
  {
    float angle = o4*180.0/parameters.numberOfOrientations;
    cv::Mat gaborKernel0, gaborKernel90;
    EPUtils::makeGaborFilter(gaborKernel0,gaborKernel90,angle);
    
    parameters.pyramidO.at(o4).pyramidImages.resize(parameters.pyramidI.pyramidImages.size());
    
    for(unsigned int i = 0; i < parameters.pyramidI.pyramidImages.size(); ++i)
    {
      cv::Mat temp0, temp90;
      cv::filter2D(parameters.pyramidI.pyramidImages.at(i),temp0,-1,gaborKernel0);
      temp0 = cv::abs(temp0);
      cv::filter2D(parameters.pyramidI.pyramidImages.at(i),temp90,-1,gaborKernel90);
      temp90 = cv::abs(temp90);
      cv::add(temp0,temp90,parameters.pyramidO.at(o4).pyramidImages.at(i));
    }
    
    combinePyramid(parameters.pyramidO.at(o4));
  }

  // color pyramids
  std::vector<cv::Mat> pyramidR;
  cv::buildPyramid(parameters.R,pyramidR,parameters.pyramidI.max_level);
  
  std::vector<cv::Mat> pyramidG;
  cv::buildPyramid(parameters.G,pyramidG,parameters.pyramidI.max_level);

  std::vector<cv::Mat> pyramidB;
  cv::buildPyramid(parameters.B,pyramidB,parameters.pyramidI.max_level);

  std::vector<cv::Mat> pyramidY;
  cv::buildPyramid(parameters.Y,pyramidY,parameters.pyramidI.max_level);

  parameters.pyramidRG.pyramidImages.resize(pyramidR.size());
  parameters.pyramidRG.changeSign = true;
  parameters.pyramidBY.pyramidImages.resize(pyramidR.size());
  parameters.pyramidBY.changeSign = true;
  
  for(unsigned int i = 0; i < parameters.pyramidRG.pyramidImages.size(); ++i)
  {
    parameters.pyramidRG.pyramidImages.at(i) = pyramidR.at(i) - pyramidG.at(i);
    parameters.pyramidBY.pyramidImages.at(i) = pyramidB.at(i) - pyramidY.at(i);
  }
  
  combinePyramid(parameters.pyramidRG);
  combinePyramid(parameters.pyramidBY);
}

} //AttentionModule