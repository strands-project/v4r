#ifndef LOCATION_MAP_HPP
#define LOCATION_MAP_HPP

#include "headers.hpp"

namespace AttentionModule
{

enum LocationTypes
{
  AM_CENTER      = 0,
  AM_LEFT_CENTER,
  AM_LEFT,
  AM_RIGHT_CENTER,
  AM_RIGHT,
  AM_TOP_CENTER,
  AM_TOP,
  AM_BOTTOM_CENTER,
  AM_BOTTOM,
  AM_CUSTOM,
};
  
class LocationSaliencyMap
{
public:
  
  LocationSaliencyMap();
  LocationSaliencyMap(int location_, int height_, int width_, int filter_size_, cv::Mat &mask_);
  LocationSaliencyMap(int location_, int height_, int width_);
  
  void setMask(cv::Mat &mask_);
  void setFilterSize(int filter_size_);
  void setWidth(int width_);
  void setHeight(int height_);
  void setLocation(int location_);
  void setCenter(cv::Point _center_point);
  bool updateMask(cv::Mat &new_mask_);
/**
 * calculates location map
 * */
  int calculateLocationMap(cv::Mat &map);

private:

/**
 * parameters for location saliency map
 * */

cv::Mat           mask;
cv::Mat           new_mask;
int               filter_size;
int               width;
int               height;
int               location;
cv::Mat           map;
bool              getUpdate;
bool              maskInUse;
cv::Point         center_point;

};

} // namespace AttentionModule

#endif //LOCATION_MAP_HPP