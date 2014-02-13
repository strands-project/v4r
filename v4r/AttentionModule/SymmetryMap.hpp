#ifndef SYMMETRY_MAP_HPP
#define SYMMETRY_MAP_HPP

#include "headers.hpp"

namespace AttentionModule
{

struct SymmetryMapParameters
{
  cv::Mat image;
  int     normalization_type;
  int     filter_size;
  int     width;
  int     height;
  int     startlevel;       //Set the highest scale
  int     maxlevel;         //Set the smalest scale (All scales between will be calculated)
  int     R1;               //Set the smaler Box that woun't be calculated
  int     R2;               //Set the bigger Box
  int     S;                //Set the standard deviaion for the distance between the pixel
  int     saliencyMapLevel; // level of the saliency map
  cv::Mat map;
  SymmetryMapParameters();
};

int CalculateSymmetryMap(SymmetryMapParameters &parameters);

} //namespace AttentionModule

#endif //SYMMETRY_MAP_HPP