#include "ColorMap.hpp"

namespace AttentionModule
{

ColorSaliencyMap::ColorSaliencyMap()
{
  image = cv::Mat_<float>::zeros(0,0);
  mask  = cv::Mat_<float>::zeros(0,0);
  normalization_type = EPUtils::NT_NONE;
  filter_size = 5;
  map = cv::Mat_<float>::zeros(0,0);
  width = 0;
  height = 0;
  useLAB = false;
  color = cv::Scalar(0,0,0);
  getUpdate = false;
  maskInUse = false;
}

void ColorSaliencyMap::setImage(cv::Mat &image_)
{
  image_.copyTo(image);
}

void ColorSaliencyMap::setMask(cv::Mat &mask_)
{
  mask_.copyTo(mask);
}

void ColorSaliencyMap::setNormalizationType(int normalization_type_)
{
  normalization_type = normalization_type_;
}

void ColorSaliencyMap::setFilterSize(int filter_size_)
{
  filter_size = filter_size_;
}

void ColorSaliencyMap::setWidth(int width_)
{
  width = width_;
}

void ColorSaliencyMap::setHeight(int height_)
{
  height = height_;
}

void ColorSaliencyMap::setUseLAB(bool useLAB_)
{
  useLAB = useLAB_;
}

void ColorSaliencyMap::setColor(cv::Scalar color_)
{
  color = color_;
}
bool ColorSaliencyMap::updateMask(cv::Mat &new_mask_)
{
  assert ( (new_mask_.rows == height) && (new_mask_.cols == width) );
  
  //std::cerr << "MASK ARRIVED" << std::endl;
  if(!getUpdate)
  {
    new_mask_.copyTo(mask);
    getUpdate = true;
    //std::cerr << "WE GOT THE MASK" << std::endl;
    return(true);
  }
  else
  {
    getUpdate = false;
    int counter = 0;
    while(maskInUse)
    {
      counter++;
    }
    new_mask_.copyTo(mask);
    getUpdate = true;
    //std::cerr << "WE GOT THE MASK" << std::endl;
    return(true);
  }
  return(true);
}

int ColorSaliencyMap::calculateColorMap(cv::Mat &map_)
{
  if((( (width == 0) || (height == 0) ) && ( (map.rows == 0) || (map.cols == 0))) ||
     (  (image.rows == 0) || (image.cols == 0) ))
  {
    return(AM_IMAGE);
  }

  if((width == 0) || (height == 0))
  {
    height = image.rows;
    width  = image.cols;
  }
  
  if((image.cols != width) || (image.rows != height) || (image.channels() != 3))
  {
    return(AM_IMAGE);
  }
  
  //if((mask.cols != width) || (mask.rows != height))
  //{
    //mask = cv::Mat_<uchar>::ones(height,width);
  //}
  
  cv::blur(image,image,cv::Size(filter_size,filter_size));
  
  cv::Mat used_mask = cv::Mat_<uchar>::ones(height,width);
  
  if(getUpdate)
  {
    maskInUse = true;
    mask.copyTo(used_mask);
    getUpdate = false;
    maskInUse = false;
  }

  if(useLAB)
  {
    cvtColor(image,image,CV_BGR2Lab);
  }

  float r_color = 0;
  float g_color = 0;
  float b_color = 0;
  float a_color = 0;

  float max_dist;
  if(useLAB)
  {
    a_color = color(0);
    b_color = color(1);
    a_color = a_color/255;
    b_color = b_color/255;
    
    // red color (1,0.5)
    float max2red = sqrt((a_color-1)*(a_color-1) + (b_color-0.5)*(b_color-0.5));
    // green color (0,0.5)
    float max2green = sqrt((a_color-0)*(a_color-0) + (b_color-0.5)*(b_color-0.5));
    // blue color (0.5,0)
    float max2blue = sqrt((a_color-0.5)*(a_color-0.5) + (b_color-0)*(b_color-0));
    // yellow color (0.5,1)
    float max2yellow = sqrt((a_color-0.5)*(a_color-0.5) + (b_color-1)*(b_color-1));
    
    max_dist = std::max(max2red,max2green);
    max_dist = std::max(max_dist,max2yellow);
    max_dist = std::max(max_dist,max2blue);
  }
  else
  {
    r_color = color(0);
    g_color = color(1);
    b_color = color(2);
    r_color = r_color/255;
    g_color = g_color/255;
    b_color = b_color/255;
    
    // red color (1,0,0)
    float max2red = sqrt((r_color-1)*(r_color-1) + (g_color-0)*(g_color-0) + (b_color-0)*(b_color-0));
    // green color (0,1,0)
    float max2green = sqrt((r_color-0)*(r_color-0) + (g_color-1)*(g_color-1) + (b_color-0)*(b_color-0));
    // blue color (0,0,1)
    float max2blue = sqrt((r_color-0)*(r_color-0) + (g_color-0)*(g_color-0) + (b_color-1)*(b_color-1));
    // red-green color (1,1,0)
    float max2red_green = sqrt((r_color-1)*(r_color-1) + (g_color-1)*(g_color-1) + (b_color-0)*(b_color-0));
    // red-blue color (1,0,1)
    float max2red_blue = sqrt((r_color-1)*(r_color-1) + (g_color-0)*(g_color-0) + (b_color-1)*(b_color-1));
    // green-blue color (0,1,1)
    float max2green_blue = sqrt((r_color-0)*(r_color-0) + (g_color-1)*(g_color-1) + (b_color-1)*(b_color-1));
    // black color (0,0,0)
    float max2black = sqrt((r_color-0)*(r_color-0) + (g_color-0)*(g_color-0) + (b_color-0)*(b_color-0));
    // white color (1,1,1)
    float max2white = sqrt((r_color-1)*(r_color-1) + (g_color-1)*(g_color-1) + (b_color-1)*(b_color-1));
    
    max_dist = std::max(max2red,max2green);
    max_dist = std::max(max_dist,max2blue);
    max_dist = std::max(max_dist,max2red_green);
    max_dist = std::max(max_dist,max2red_blue);
    max_dist = std::max(max_dist,max2green_blue);
    max_dist = std::max(max_dist,max2black);
    max_dist = std::max(max_dist,max2white);
  }

  map = cv::Mat_<float>::zeros(height,width);
  
  if(useLAB)
  {
    for(int r = 0; r < height; ++r)
    {
      if(getUpdate)
      {
	maskInUse = true;
        mask.copyTo(used_mask);
        getUpdate = false;
        maskInUse = false;
      }
      for(int c = 0; c < width; ++c)
      {
	if(used_mask.at<uchar>(r,c))
	{
	  float aa = image.at<uchar>(r,3*c+1);
          float bb = image.at<uchar>(r,3*c+2);
	  aa = aa/255;
	  bb = bb/255;
	
          float dist = (aa - a_color)*(aa - a_color) + (bb - b_color)*(bb - b_color);
	  dist = sqrt(dist);
          map.at<float>(r,c) = 1.0 - dist/max_dist;
	}
      }
    }
  }
  else
  {
    for(int r = 0; r < height; ++r)
    {
      if(getUpdate)
      {
	maskInUse = true;
        mask.copyTo(used_mask);
        getUpdate = false;
        maskInUse = false;
      }
      for(int c = 0; c < width; ++c)
      {
	if(used_mask.at<uchar>(r,c))
	{
	  float rr = image.at<uchar>(r,3*c+2);
          float gg = image.at<uchar>(r,3*c+1);
          float bb = image.at<uchar>(r,3*c+0);
	  rr = rr/255;
	  gg = gg/255;
	  bb = bb/255;
	
          float dist = (rr - r_color)*(rr - r_color) + (gg - g_color)*(gg - g_color) + (bb - b_color)*(bb - b_color);
	  dist = sqrt(dist);
          map.at<float>(r,c) = 1.0 - dist/max_dist;
	}
      }
    }
  }
  
  getUpdate = false;

  //cv::blur(map,map,cv::Size(filter_size,filter_size));

  //EPUtils::normalize(map,normalization_type);

  map.copyTo(map_);
  return(AM_OK);
}

int ColorSaliencyMap::calculateColorMapPyramid(cv::Mat &map_)
{
  if((( (width == 0) || (height == 0) ) && ( (map.rows == 0) || (map.cols == 0))) ||
     (  (image.rows == 0) || (image.cols == 0) ))
  {
    return(AM_IMAGE);
  }

  if((width == 0) || (height == 0))
  {
    height = map.rows;
    width  = map.cols;
  }
  
  if((image.cols != width) || (image.rows != height) || (image.channels() != 3))
  {
    return(AM_IMAGE);
  }
  
  // calculate puramid with saliency maps
  int max_level = pyramidParameters.max_level + 1;
  pyramidParameters.pyramidImages.clear();
  cv::buildPyramid(image,pyramidParameters.pyramidImages,max_level);
  pyramidParameters.pyramidFeatures.clear();
  pyramidParameters.pyramidFeatures.resize(pyramidParameters.pyramidImages.size());
  
  for(int i = pyramidParameters.start_level; i <= pyramidParameters.max_level; ++i)
  {
    // start creating parameters
    ColorSaliencyMap parameters_current;
    parameters_current.setWidth(pyramidParameters.pyramidImages.at(i).cols);
    parameters_current.setHeight(pyramidParameters.pyramidImages.at(i).rows);
    parameters_current.setImage(pyramidParameters.pyramidImages.at(i));
    parameters_current.setColor(color);
    parameters_current.setUseLAB(useLAB); 
    
    parameters_current.calculateColorMap(pyramidParameters.pyramidFeatures.at(i)); 
  }
  // combine saliency maps
  combinePyramid(pyramidParameters);
  pyramidParameters.map.copyTo(map);
  
  map.copyTo(map_);
  return(0);
}

} //namespace AttentionModule