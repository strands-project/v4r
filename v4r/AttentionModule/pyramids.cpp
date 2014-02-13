#include "pyramids.hpp"

namespace AttentionModule
{

PyramidParameters::PyramidParameters()
{
  combination_type = AM_SIMPLE;
  start_level = 0;
  max_level = 4;
  sm_level = 0;
  lowest_c = 2;
  highest_c = 4;
  smallest_cs = 3;
  largest_cs = 4;
  number_of_features = 0;
  normalization_type = EPUtils::NT_NONE;
  width = 0;
  height = 0;
  R.clear();
  changeSign = false;
  onSwitch = true;
  pyramidFeatures.clear();
  pyramidImages.clear();
  map = cv::Mat_<float>::zeros(0,0);
}

void PyramidParameters::print()
{
  std::cerr << "combination_type       = " << combination_type << std::endl;
  std::cerr << "start_level            = " << start_level << std::endl;
  std::cerr << "max_level              = " << max_level << std::endl;
  std::cerr << "sm_level               = " << sm_level << std::endl;
  std::cerr << "lowest_c               = " << lowest_c << std::endl;
  std::cerr << "highest_c              = " << highest_c << std::endl;
  std::cerr << "smallest_cs            = " << smallest_cs << std::endl;
  std::cerr << "largest_cs             = " << largest_cs << std::endl;
  std::cerr << "number_of_features     = " << number_of_features << std::endl;
  std::cerr << "normalization_type     = " << normalization_type << std::endl;
  std::cerr << "width                  = " << width << std::endl;
  std::cerr << "height                 = " << height << std::endl;
  std::cerr << "changeSign             = " << changeSign << std::endl;
  std::cerr << "pyramidFeatures.size() = " << pyramidFeatures.size() << std::endl;
  std::cerr << "pyramidImages.size()   = " << pyramidImages.size() << std::endl;
}

void combinePyramid(PyramidParameters &parameters)
{
  switch(parameters.combination_type)
  {
    case AM_SIMPLE:
      combinePyramidSimple(parameters);
      return;
    case AM_ITTI:
      combinePyramidCenterSurround(parameters);
      return;
    case AM_FRINTROP:
      combinePyramidFrintrop(parameters);
      return;
    default:
      combinePyramidSimple(parameters);
      return;
  }
}

void combinePyramidSimple(PyramidParameters &parameters)
{
  int sm_level = parameters.sm_level;
  int sm_width = parameters.pyramidImages.at(sm_level).cols;
  int sm_height = parameters.pyramidImages.at(sm_level).rows;
  parameters.map = cv::Mat_<float>::zeros(sm_height,sm_width);
  
  for(int i = parameters.start_level; i <= parameters.max_level; ++i)
  {
    cv::Mat temp;
    cv::resize(parameters.pyramidFeatures.at(i),temp,parameters.map.size());
    parameters.map = parameters.map + temp;
  }
  
  double maxValue, minValue;
  cv::minMaxLoc(parameters.map,&minValue,&maxValue);
  parameters.max_map_value = maxValue;
  EPUtils::normalize(parameters.map,parameters.normalization_type);
}

void checkLevels(PyramidParameters &parameters)
{
  parameters.max_level = parameters.highest_c + parameters.largest_cs;
  
  int nw = parameters.width;
  int nh = parameters.height;
  int current_level = 0;
  while (((nw/2) >= 1) && ((nh/2) >= 1))
  {
    ++current_level;
    nw = nw/2;
    nh = nh/2;
  }
  
  parameters.max_level = (current_level>parameters.max_level ? parameters.max_level:current_level);
  
  if(parameters.max_level < parameters.highest_c + parameters.largest_cs)
    parameters.largest_cs -= (parameters.highest_c + parameters.largest_cs-parameters.max_level);
  
  if (parameters.largest_cs < parameters.smallest_cs)
    parameters.largest_cs = parameters.smallest_cs = 0;
  
  if(parameters.sm_level > parameters.max_level)
  {
    parameters.sm_level = parameters.max_level;
  }
  
  parameters.number_of_features = (parameters.highest_c-parameters.lowest_c+1)*(parameters.largest_cs-parameters.smallest_cs+1);
}

void combinePyramidCenterSurround(PyramidParameters &parameters)
{ 
  parameters.pyramidFeatures.clear();
  parameters.pyramidFeatures.resize(parameters.number_of_features);
  
  for (int i = parameters.lowest_c; i <= parameters.highest_c; ++i)
  {
    for (int j = parameters.smallest_cs; j <= parameters.largest_cs; ++j)
    {
      int current = (parameters.largest_cs-parameters.smallest_cs+1)*(i-parameters.lowest_c)+(j-parameters.smallest_cs);
      if (current < parameters.number_of_features)
      {
        cv::Mat temp;
        resize(parameters.pyramidImages.at(i+j),temp,cv::Size(parameters.pyramidImages.at(i).cols,parameters.pyramidImages.at(i).rows));
	if(parameters.changeSign)
	  temp = -1 * temp;
	
        cv::absdiff(parameters.pyramidImages.at(i),temp,parameters.pyramidFeatures.at(current));
	EPUtils::normalize(parameters.pyramidFeatures.at(current),parameters.normalization_type);
      }
    }
  }
  
  parameters.map = cv::Mat_<float>::zeros(parameters.pyramidImages.at(parameters.sm_level).rows,parameters.pyramidImages.at(parameters.sm_level).cols);
  
  for (int i=0; i < parameters.number_of_features; ++i)
  {
    cv::Mat temp;
    cv::resize(parameters.pyramidFeatures.at(i),temp,cv::Size(parameters.pyramidImages.at(parameters.sm_level).cols,parameters.pyramidImages.at(parameters.sm_level).rows));
    cv::add(parameters.map,temp,parameters.map);
  }
  
  if(!parameters.changeSign)
    EPUtils::normalize(parameters.map,parameters.normalization_type);
}

void combinePyramidFrintrop(PyramidParameters &parameters)
{
  parameters.number_of_features = (parameters.max_level-parameters.start_level+1)*parameters.R.size();
  
  parameters.pyramidFeatures.resize(parameters.number_of_features);
  
  for(int s = parameters.start_level; s <= parameters.max_level; ++s)
  {
    for(unsigned int r = 0; r < parameters.R.size(); ++r)
    {
      cv::Mat kernel = cv::Mat_<float>::ones(parameters.R.at(r),parameters.R.at(r));
      kernel = kernel / (parameters.R.at(r)*parameters.R.at(r));
      
      int current = (parameters.R.size()-0)*(s-parameters.start_level)+(r-0);
      if (current < parameters.number_of_features)
      {
        cv::Mat temp;
        filter2D(parameters.pyramidImages.at(s),temp,parameters.pyramidImages.at(s).depth(),kernel);

        if(parameters.onSwitch)
          temp = parameters.pyramidImages.at(s) - temp;
        else
          temp = temp - parameters.pyramidImages.at(s);
  
        cv::max(temp,0.0,parameters.pyramidFeatures.at(current));
      }
    }
  }
  
  parameters.map = cv::Mat_<float>::zeros(parameters.pyramidImages.at(parameters.sm_level).rows,parameters.pyramidImages.at(parameters.sm_level).cols);
  
  for (int i=0; i < parameters.number_of_features; ++i)
  {
    cv::Mat temp;
    cv::resize(parameters.pyramidFeatures.at(i),temp,cv::Size(parameters.pyramidImages.at(parameters.sm_level).cols,parameters.pyramidImages.at(parameters.sm_level).rows));
    cv::add(parameters.map,temp,parameters.map);
  }
  
  double maxValue, minValue;
  cv::minMaxLoc(parameters.map,&minValue,&maxValue);
  parameters.max_map_value = maxValue;
  EPUtils::normalize(parameters.map,parameters.normalization_type);
}

}