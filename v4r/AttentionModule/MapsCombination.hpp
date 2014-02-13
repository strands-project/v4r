#ifndef MAPS_COMBINATION_HPP
#define MAPS_COMBINATION_HPP

#include "headers.hpp"

namespace AttentionModule
{

enum CombinationTypes
{
  AM_SUM      = 0,
  AM_MUL,
  AM_MIN,
  AM_MAX
};
  
// assume that maps are normalized to (0,1) range
int CombineMaps(std::vector<cv::Mat> &maps, cv::Mat &combinedMap, int combination_type = AM_SUM, 
                int normalization_type = EPUtils::NT_NONE);
  
} //namespace AttentionModule

#endif //MAPS_COMBINATION_HPP
