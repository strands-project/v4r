#ifndef ORIENTATION_MAP_HPP
#define ORIENTATION_MAP_HPP

#include "headers.hpp"
#include "pyramids.hpp"

namespace AttentionModule
{

/**
 * parameters for orientation saliency map
 * */

class OrientationSaliencyMap
{
public:
  OrientationSaliencyMap();
  void setImage(cv::Mat &image_);
  void setMask(cv::Mat &mask_);
  void setAngle(float angle_);
  void setBandwidth(float bandwidth_);
  void setFilterSize(int filter_size_);
  void setWidth(int width_);
  void setHeight(int height_);
  bool updateMask(cv::Mat &new_mask_);
  float getMaxSum();
/**
* calculates single color map
* */
  int calculateOrientationMap(cv::Mat &map_);
  int calculateOrientationMapPyramid(cv::Mat &map_);
private:
  cv::Mat           image;
  cv::Mat           mask;
  float             angle;
  float             max_sum;
  float             bandwidth;
  int               filter_size;
  int               width;
  int               height;
  cv::Mat           map;
  PyramidParameters pyramidParameters;
  bool              getUpdate;
  bool              maskInUse;

};

} // namespace AttentionModule

#endif //ORIENTATION_MAP_HPP