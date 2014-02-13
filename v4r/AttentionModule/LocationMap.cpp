#include "LocationMap.hpp"

namespace AttentionModule
{

LocationSaliencyMap::LocationSaliencyMap()
{
  mask  = cv::Mat_<float>::zeros(0,0);
  filter_size = 5;
  map = cv::Mat_<float>::zeros(0,0);
  width = 0;
  height = 0;
  location = AM_CENTER;
  getUpdate = false;
  maskInUse = false;
}

LocationSaliencyMap::LocationSaliencyMap(int location_, int height_, int width_, int filter_size_, cv::Mat &mask_)
{
  location = location_;
  height = height_;
  width = width_;
  filter_size = filter_size_;
  getUpdate = false;
  maskInUse = false;
  mask_.copyTo(mask);
}

LocationSaliencyMap::LocationSaliencyMap(int location_, int height_, int width_)
{
  location = location_;
  height = height_;
  width = width_;
  filter_size = 5;
  getUpdate = false;
  maskInUse = false;
  mask  = cv::Mat_<float>::ones(height,width);
}

bool LocationSaliencyMap::updateMask(cv::Mat &new_mask_)
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

int LocationSaliencyMap::calculateLocationMap(cv::Mat &map_)
{
  if(( (width == 0) || (height == 0) ) && ( (map.rows == 0) || (map.cols == 0)))
  {
    return(AM_IMAGE);
  }

  if((width == 0) || (height == 0))
  {
    height = map.rows;
    width  = map.cols;
  }
  
//   if((mask.cols != width) || (mask.rows != height))
//   {
//    mask = cv::Mat_<uchar>::ones(height,width);
//   }
  
  cv::Mat used_mask = cv::Mat_<uchar>::ones(height,width);
  
  if(getUpdate)
  {
    maskInUse = true;
    mask.copyTo(used_mask);
    getUpdate = false;
    maskInUse = false;
  }

  cv::Point center;
  float a = 1;
  float b = 1;

  switch(location)
  {
    case AM_CENTER:
      center = cv::Point(width/2,height/2);
      break;
    case AM_LEFT_CENTER:
      center = cv::Point(width/8,height/2);
      break;
    case AM_LEFT:
      center = cv::Point(width/8,height/2);
      b = 0;
      break;
    case AM_RIGHT_CENTER:
      center = cv::Point(7*width/8,height/2);
      break;
    case AM_RIGHT:
      center = cv::Point(7*width/8,height/2);
      b = 0;
      break;
    case AM_TOP_CENTER:
      center = cv::Point(width/2,20/*height/4*/);
      break;
    case AM_TOP:
      center = cv::Point(width/2,20/*height/4*/);
      a = 0;
      break;
    case AM_BOTTOM_CENTER:
      center = cv::Point(width/2,7*height/8);
      break;
    case AM_BOTTOM:
      center = cv::Point(width/2,7*height/8);
      a = 0;
      break;
    case AM_CUSTOM:
//       std::cerr << "here" << std::endl;
      center = center_point;
      break;
    default:
      center = cv::Point(width/2,height/2);
      break;
  }
  
  map = cv::Mat_<float>::zeros(height,width);
  
  for(int r = 0; r < height; ++r)
  {
    for(int c = 0; c < width; ++c)
    {
//       if(getUpdate)
//       {
// 	maskInUse = true;
//         mask.copyTo(used_mask);
//         getUpdate = false;
//         maskInUse = false;
//       }
//       if(used_mask.at<uchar>(r,c))
//       {
        float dx = c-center.x;
        dx = a*(dx/width);
        float dy = r-center.y;
        dy = b*(dy/height);
        float value = dx*dx + dy*dy;
        map.at<float>(r,c) = exp(-200*value);
//       }
    }
  }
  
  //cv::blur(map,map,cv::Size(filter_size,filter_size));

//   EPUtils::normalize(map);

//   cv::imshow("map",map);
//   cv::waitKey(-1);

  getUpdate = false;

  map.copyTo(map_);
  return(AM_OK);
}

void LocationSaliencyMap::setMask(cv::Mat &mask_)
{
  mask_.copyTo(mask);
}

void LocationSaliencyMap::setFilterSize(int filter_size_)
{
  filter_size = filter_size_;
}

void LocationSaliencyMap::setWidth(int width_)
{
  width = width_;
}

void LocationSaliencyMap::setHeight(int height_)
{
  height = height_;
}

void LocationSaliencyMap::setLocation(int location_)
{
  location = location_;
}

void LocationSaliencyMap::setCenter(cv::Point _center_point)
{
  center_point = _center_point;
}

} //namespace AttentionModule