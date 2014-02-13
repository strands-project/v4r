#ifndef NORMALIZATION_HPP
#define NORMALIZATION_HPP

#include "headers.hpp"

namespace EPUtils
{

enum NormalizationTypes
{
  NT_NONE        = 0,
  NT_NONMAX,
  NT_FRINTROP_NORM,
  NT_EMPTY,
  NT_MAX_DIVIDE,
  NT_NONE_REAL,
};
  
void computeLocalMax(cv::Mat &image, int &numLocalMax, float &averageLocalMax, float threshold = 0);
/**
 * normalizes image
 * */
void normalize(cv::Mat &map, int normalization_type = NT_NONE, float newMaxValue = 1, float newMinValue = 0);
void normalizeNonMax(cv::Mat &map);
void normalizeFrintrop(cv::Mat &map);
void normalizeMin2Zero(cv::Mat &map);
/**
 * normalizes image by simply dividind image by its maximum value
 * */
void normalizeMax2One(cv::Mat &map);

} //namespace EPUtils

#endif //NORMALIZATION_HPP