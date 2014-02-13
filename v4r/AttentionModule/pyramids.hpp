#ifndef PYRAMIDS_HPP
#define PYRAMIDS_HPP

#include "headers.hpp"

namespace AttentionModule
{

enum PyramidTypes
{
  AM_SIMPLE      = 0,
  AM_ITTI,
  AM_FRINTROP,
};
  
struct PyramidParameters
{
  int                  combination_type;
  int                  start_level;
  int                  max_level;
  int                  sm_level;
  int                  lowest_c;
  int                  highest_c;
  int                  smallest_cs;
  int                  largest_cs;
  int                  number_of_features;
  int                  normalization_type;
  int                  width;
  int                  height;
  std::vector<int>     R;
  bool                 onSwitch;
  bool                 changeSign;
  std::vector<cv::Mat> pyramidFeatures;
  std::vector<cv::Mat> pyramidImages;
  float                max_map_value;
  cv::Mat              map;
  PyramidParameters();
  void print();
};
  
void combinePyramid(PyramidParameters &parameters);
void combinePyramidSimple(PyramidParameters &parameters);
void combinePyramidCenterSurround(PyramidParameters &parameters);
void combinePyramidFrintrop(PyramidParameters &parameters);
void checkLevels(PyramidParameters &parameters);

}
#endif