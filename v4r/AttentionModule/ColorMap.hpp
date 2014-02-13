#ifndef COLOR_MAP_HPP
#define COLOR_MAP_HPP

#include "headers.hpp"
#include "pyramids.hpp"

namespace AttentionModule
{

/**
 * parameters for surface curvature saliency map
 * */

class ColorSaliencyMap
{
public:
  ColorSaliencyMap();
  void setImage(cv::Mat &image_);
  void setMask(cv::Mat &mask_);
  void setNormalizationType(int normalization_type_);
  void setFilterSize(int filter_size_);
  void setWidth(int width_);
  void setHeight(int height_);
  void setUseLAB(bool useLAB_);
  void setColor(cv::Scalar color_);
  bool updateMask(cv::Mat &new_mask_);
/**
* calculates single color map
* */
  int calculateColorMap(cv::Mat &map_);
  int calculateColorMapPyramid(cv::Mat &map_);
private:
  cv::Mat           image;
  cv::Mat           mask;
  int               normalization_type;
  int               filter_size;
  int               width;
  int               height;
  bool              useLAB;
  cv::Scalar        color;
  cv::Mat           map;
  PyramidParameters pyramidParameters;
  bool              getUpdate;
  bool              maskInUse;

};

} // namespace AttentionModule

#endif //COLOR_MAP_HPP