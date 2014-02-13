#include "OrientationMap.hpp"

namespace AttentionModule
{

OrientationSaliencyMap::OrientationSaliencyMap()
{
  image = cv::Mat_<float>::zeros(0,0);
  mask  = cv::Mat_<float>::zeros(0,0);
  angle = 0;
  max_sum = 1;
  bandwidth = 2;
  filter_size = 5;
  map = cv::Mat_<float>::zeros(0,0);
  width = 0;
  height = 0;
  getUpdate = false;
  maskInUse = false;
}

void OrientationSaliencyMap::setImage(cv::Mat &image_)
{
  image_.copyTo(image);
}

void OrientationSaliencyMap::setMask(cv::Mat &mask_)
{
  mask_.copyTo(mask);
}

void OrientationSaliencyMap::setAngle(float angle_)
{
  angle = angle_;
}

void OrientationSaliencyMap::setBandwidth(float bandwidth_)
{
  bandwidth = bandwidth_;
}

void OrientationSaliencyMap::setFilterSize(int filter_size_)
{
  filter_size = filter_size_;
}

void OrientationSaliencyMap::setWidth(int width_)
{
  width = width_;
}

void OrientationSaliencyMap::setHeight(int height_)
{
  height = height_;
}

bool OrientationSaliencyMap::updateMask(cv::Mat &new_mask_)
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

float OrientationSaliencyMap::getMaxSum()
{
  return(max_sum);
}

int OrientationSaliencyMap::calculateOrientationMap(cv::Mat &map_)
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
  
  cv::blur(image,image,cv::Size(filter_size,filter_size));
  // convert to grey
  image.convertTo(image,CV_32F,1.0f/255);
  cv::Mat image_gray;
  cv::cvtColor(image,image_gray,CV_BGR2GRAY); // TODO: fix this
  
  cv::Mat used_mask = cv::Mat_<uchar>::ones(height,width);
  
  if(getUpdate)
  {
    maskInUse = true;
    mask.copyTo(used_mask);
    getUpdate = false;
    maskInUse = false;
  }

  map = cv::Mat_<float>::zeros(height,width);
  
  //create Gabor kernel
  cv::Mat gaborKernel;
  EPUtils::makeGaborKernel2D(gaborKernel,max_sum,angle,bandwidth);
  assert (gaborKernel.rows == gaborKernel.cols);
  assert (gaborKernel.rows % 2 == 1);
  int gaborFilerSize = gaborKernel.rows / 2;
  
  //for(int r = gaborFilerSize; r < height-gaborFilerSize; ++r)
  for(int r = 0; r < height; ++r)
  {
    if(getUpdate)
    {
      maskInUse = true;
      mask.copyTo(used_mask);
      getUpdate = false;
      maskInUse = false;
    }
    //for(int c = gaborFilerSize; c < width-gaborFilerSize; ++c)
    for(int c = 0; c < width; ++c)
    {
      if(used_mask.at<uchar>(r,c))
      {
	float value = 0;
	for(int j = r-gaborFilerSize; j <= r+gaborFilerSize; ++j) // rows
	{
	  int yy = j;
	  if(j < 0)
	  {
	    yy = -j;
	  }
	  if(j >= height)
	  {
	    yy = height-(j-height+1)-1;
	  }
	  /*if(j >= height)
	    continue;*/
	  
	  for(int i = c-gaborFilerSize; i <= c+gaborFilerSize; ++i) // cols
	  {
	    int xx = i;
	    if(i < 0)
	    {
	      xx = -i;
	    }
	    if(i >= width)
	    {
	      xx = width-(i-width+1)-1;
	    }
	    
	    value += image_gray.at<float>(yy,xx)*gaborKernel.at<float>(j-(r-gaborFilerSize),(i-(c-gaborFilerSize)));
	  }
	}
	
	//value = value > 0 ? value : -value;
	//value = sqrt(value/max_sum);
	//value = value/max_sum;
	
	map.at<float>(r,c) = value;
      }
    }
  }
  
  //cv::minMaxLoc(map,min,max);
  
  getUpdate = false;
  
  map = cv::abs(map);
  map = map / max_sum;
  cv::sqrt(map,map);
  
  double min, max;
  cv::minMaxLoc(map,&min,&max);
  std::cerr << "min = " << min << std::endl;
  std::cerr << "max = " << max << std::endl;
  //std::cerr << "max_sum = " << max_sum << std::endl;
  
  //cv::blur(map,map,cv::Size(filter_size,filter_size));
  //EPUtils::normalize(map,normalization_type);

  map.copyTo(map_);
  return(AM_OK);
}

int OrientationSaliencyMap::calculateOrientationMapPyramid(cv::Mat &map_)
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
    OrientationSaliencyMap parameters_current;
    parameters_current.setWidth(pyramidParameters.pyramidImages.at(i).cols);
    parameters_current.setHeight(pyramidParameters.pyramidImages.at(i).rows);
    parameters_current.setImage(pyramidParameters.pyramidImages.at(i));
    parameters_current.setAngle(angle);
    parameters_current.setBandwidth(bandwidth); 
    
    parameters_current.calculateOrientationMap(pyramidParameters.pyramidFeatures.at(i)); 
  }
  // combine saliency maps
  combinePyramid(pyramidParameters);
  pyramidParameters.map.copyTo(map);
  
  map.copyTo(map_);
  return(0);
}

} //namespace AttentionModule