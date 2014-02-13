#include "SurfaceHeightMap.hpp"
#include <sys/time.h>

namespace AttentionModule
{

SurfaceHeightSaliencyMap::SurfaceHeightSaliencyMap()
{
  //cloud = NULL;
  //indices = NULL;
  //coefficients = NULL;
  
  distance_from_top = 0;
  normalization_type = EPUtils::NT_NONE_REAL;
  filter_size = 5;
  max_distance = 0.005;
  map = cv::Mat_<float>::zeros(0,0);
  mask = cv::Mat_<uchar>::zeros(0,0);
  width = 0;
  height = 0;
  heightType = AM_TALL;
  cameraParametrs.clear();
  getUpdate = false;
  maskInUse = false;
}

void SurfaceHeightSaliencyMap::setCloud(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud_)
{
  cloud = cloud_;
}

void SurfaceHeightSaliencyMap::setIndices(pcl::PointIndices::Ptr indices_)
{
  indices = indices_;
}

void SurfaceHeightSaliencyMap::setModelCoefficients(pcl::ModelCoefficients::Ptr coefficients_)
{
  coefficients = coefficients_;
}

void SurfaceHeightSaliencyMap::setDistanceFromTop(float distance_from_top_)
{
  distance_from_top = distance_from_top_;
}

void SurfaceHeightSaliencyMap::setNormalizationType(int normalization_type_)
{
  normalization_type = normalization_type_;
}

void SurfaceHeightSaliencyMap::setFilterSize(int filter_size_)
{
  filter_size = filter_size_;
}

void SurfaceHeightSaliencyMap::setMaxDistance(int max_distance_)
{
  max_distance = max_distance_;
}

void SurfaceHeightSaliencyMap::setWidth(int width_)
{
  width = width_;
}

void SurfaceHeightSaliencyMap::setHeight(int height_)
{
  height = height_;
}

void SurfaceHeightSaliencyMap::setHeightType(int heightType_)
{
  heightType = heightType_;
}

void SurfaceHeightSaliencyMap::setMask(cv::Mat &mask_)
{
  mask.copyTo(mask);
}

void SurfaceHeightSaliencyMap::setCameraParameters(std::vector<float> &cameraParametrs_)
{
  cameraParametrs = cameraParametrs_;
}

bool SurfaceHeightSaliencyMap::updateMask(cv::Mat &new_mask_)
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

int SurfaceHeightSaliencyMap::calculateSurfaceHeightMap(cv::Mat &map_)
{
  //unsigned long long lasttime;
  //unsigned long long currenttime;
  //unsigned long long timediff;
  //struct timeval tv;
  //gettimeofday(&tv, NULL);
  //lasttime = (unsigned) (tv.tv_sec * 1000) + (unsigned) (tv.tv_usec / 1000.0); //convert to milliseconds
  
  if(cloud == NULL)
  {
    return(AM_POINTCLOUD);
  }
  
  if(indices == NULL)
  {
    return(AM_POINTCLOUD);
  }
  
  if(coefficients == NULL)
  {
    return(AM_POINTCLOUD);
  }

  if(( (width == 0) || (height == 0) ) && ( (map.rows == 0) || (map.cols == 0) ))
  {
    return(AM_IMAGE);
  }

  if((width == 0) || (height == 0))
  {
    height = map.rows;
    width  = map.cols;
  }

  if(coefficients->values.size() != 4)
  {
    return(AM_PLANECOEFFICIENTS);
  }
  
  //if((mask.cols != width) || (mask.rows != height))
  //{
  //  mask = cv::Mat_<uchar>::ones(height,width);
  //}
  
  cv::Mat used_mask = cv::Mat_<uchar>::ones(height,width);
  
  // update the mask
  if(getUpdate)
  {
    maskInUse = true;
    mask.copyTo(used_mask);
    getUpdate = false;
    maskInUse = false;
  }

  // calculate projections on the plane
  pcl::PointCloud<pcl::PointXYZ>::Ptr points_projected (new pcl::PointCloud<pcl::PointXYZ>());
  std::vector<float> distances;
  // calculate coordinates of the projections and normalized distances to the plane
  EPUtils::ProjectPointsOnThePlane(coefficients,cloud,points_projected,distances,indices);

  // create kd-tree to search for neighbouring points
  pcl::search::KdTree<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(points_projected);

  std::vector<int> pointIdxRadiusSearch;
  std::vector<float> pointRadiusSquaredDistance;
  
  // max distance from the plane in the current line of points
  std::vector<float> max_distance;
  max_distance.resize(distances.size(),0);
  // are point already used
  std::vector<bool> used_points;
  used_points.resize(distances.size(),false);

  float radius = 0.09;
  float shift_from_plane = 0.2;
  
  //gettimeofday(&tv, NULL);
  //currenttime = (unsigned) (tv.tv_sec * 1000) + (unsigned) (tv.tv_usec / 1000.0); //convert to milliseconds
  //timediff = currenttime - lasttime;
  //std::cerr << "Non incremental time = " << timediff << std::endl;
  //lasttime = currenttime;
  
  for(unsigned int i = 0; i < used_points.size(); ++i)
  {
    if(used_points.at(i))
    {
      continue;
    }
    
    if(getUpdate)
    {
      maskInUse = true;
      mask.copyTo(used_mask);
      getUpdate = false;
      maskInUse = false;
    }
    int idx = indices->indices.at(i);
    int c = idx % width;
    int r = idx / width;
    if(!(used_mask.at<uchar>(r,c)))
    {
      continue;
    }
    
    pointIdxRadiusSearch.clear();
    pointRadiusSquaredDistance.clear();
      
    pcl::PointXYZ searchPoint = points_projected->points.at(i);
      
    // if we found points close to the given point
    if(kdtree.radiusSearch(searchPoint,radius,pointIdxRadiusSearch,pointRadiusSquaredDistance))
    {
      float max_distance_val = 0;
      float average_distance = 0;
      int average_counter = 0;
      // find point with biggest distance to plane
      for(unsigned int j = 0; j < pointIdxRadiusSearch.size(); ++j)
      {
        int index = pointIdxRadiusSearch.at(j);
        average_distance += distances.at(index);
        average_counter += 1;
        if(!used_points.at(index))
        {
	  if((distances.at(index) > max_distance_val) && (distances.at(index) > shift_from_plane))
	  {
	    max_distance_val = distances.at(index);
          }
        }
      }
      if(average_counter>0)
      {
        average_distance /= average_counter;
      }
	
      // set biggest distance
      for(unsigned int j = 0; j < pointIdxRadiusSearch.size(); ++j)
      {
	int index = pointIdxRadiusSearch.at(j);
	if(!used_points.at(index))
	{
	  max_distance.at(index) = max_distance_val;
	  //max_distance.at(index) = average_distance;
          used_points.at(index) = true;
	}
      }
    }
  }

  map = cv::Mat_<float>::zeros(height,width);
  
  float heightCoefficient = getHeightCoefficient(heightType);
  for(unsigned int i = 0; i < max_distance.size(); ++i)
  {
    if(getUpdate)
    {
      maskInUse = true;
      mask.copyTo(used_mask);
      getUpdate = false;
      maskInUse = false;
    }
    int idx = indices->indices.at(i);
    int c = idx % width;
    int r = idx / width;
    if(!(used_mask.at<uchar>(r,c)))
    {
      continue;
    }
    
    float value = (max_distance.at(i)-shift_from_plane)/(1.0-shift_from_plane);
    //float value = max_distance.at(i);
    if(value > 0)
    {
      float t1 = 1.0-heightCoefficient;
      float t2 = heightCoefficient;
      value = t1*value + t2*(1-value);
      map.at<float>(r,c) = value;
    }
    else
    {
      map.at<float>(r,c) = 0;
    }
  }

  //cv::blur(map,map,cv::Size(filter_size,filter_size));

  //EPUtils::normalize(map,normalization_type);
  
  getUpdate = false;
  
  map.copyTo(map_);
  
  //gettimeofday(&tv, NULL);
  //currenttime = (unsigned) (tv.tv_sec * 1000) + (unsigned) (tv.tv_usec / 1000.0); //convert to milliseconds
  //timediff = currenttime - lasttime;
  //std::cerr << "Incremental time = " << timediff << std::endl;
  
  return(AM_OK);
}

int SurfaceHeightSaliencyMap::calculateSurfaceDistanceMap(cv::Mat &map_)
{

  if(cloud == NULL)
  {
    return(AM_POINTCLOUD);
  }
  
  if(indices == NULL)
  {
    return(AM_POINTCLOUD);
  }
  
  if(coefficients == NULL)
  {
    return(AM_POINTCLOUD);
  }

  if(( (width == 0) || (height == 0) ) && ( (map.rows == 0) || (map.cols == 0) ))
  {
    return(AM_IMAGE);
  }

  if((width == 0) || (height == 0))
  {
    height = map.rows;
    width  = map.cols;
  }

  if(coefficients->values.size() != 4)
  {
    return(AM_PLANECOEFFICIENTS);
  }

  // Retrieve Ground Plane Coefficients
  float a = coefficients->values.at(0);
  float b = coefficients->values.at(1);
  float c = coefficients->values.at(2);
  float d = coefficients->values.at(3);

  float max_dist = 0;
  cv::Mat distance_map = cv::Mat_<float>::zeros(height,width);

  for(unsigned int pi = 0; pi < indices->indices.size(); ++pi)
  {
    int index = indices->indices.at(pi);
    pcl::PointXYZ current_point = cloud->points.at(index);

    float dist = pcl::pointToPlaneDistance(current_point,a,b,c,d);

    if(dist > max_dist)
      max_dist = dist;

    int xx = index % width;
    int yy = index / width;
    distance_map.at<float>(yy,xx) = dist;
  }

  if(max_dist < max_distance)
  {
    return(AM_POINTCLOUD);
  }

  float d0 = distance_from_top * max_dist;
  float a_param = -1.0/((max_dist-d0)*(max_dist-d0));
  float b_param = 2.0/(max_dist-d0);

  map = cv::Mat_<float>::zeros(height,width);

  for(int r = 0; r < height; ++r)
  {
    for(int c = 0; c < width; ++c)
    {
      float dist = distance_map.at<float>(r,c);
      if(dist > 0)
      {
        map.at<float>(r,c) = (a_param * dist * dist + b_param * dist);

        if(map.at<float>(r,c) < 0)
        {
          map.at<float>(r,c) = 0;
        }
      }
    }
  }

  cv::blur(map,map,cv::Size(filter_size,filter_size));

  EPUtils::normalize(map,normalization_type);
  map.copyTo(map_);

  return(AM_OK);
}

int SurfaceHeightSaliencyMap::calculateSurfaceHeightMapPyramid(cv::Mat &map_)
{
  if(cloud == NULL)
  {
    return(AM_POINTCLOUD);
  }
  
  if(indices == NULL)
  {
    return(AM_POINTCLOUD);
  }
  
  // set values of the input image
  if(( (width == 0) || (height == 0) ) && ( (map.rows == 0) || (map.cols == 0) ))
  {
    return(AM_IMAGE);
  }

  if((width == 0) || (height == 0))
  {
    height = map.rows;
    width  = map.cols;
  }
  
  if(cameraParametrs.size() != 4)
  {
    return(AM_CAMERAPARAMETRS);
  }
  
  // create depth
  cv::Mat depth;
  EPUtils::PointCloud2Depth(depth,cloud,width,height,indices);
  
  // calculate puramid with saliency maps
  int max_level = pyramidParameters.max_level + 1;
  pyramidParameters.pyramidImages.clear();
  cv::buildPyramid(depth,pyramidParameters.pyramidImages,max_level);
  pyramidParameters.pyramidFeatures.clear();
  pyramidParameters.pyramidFeatures.resize(pyramidParameters.pyramidImages.size());
  
  for(int i = pyramidParameters.start_level; i <= pyramidParameters.max_level; ++i)
  {
    int scalingFactor = pow(2.0f,i);
    std::vector<float> cameraParametrsCur;
    cameraParametrsCur.resize(4);
    cameraParametrsCur.at(0) = cameraParametrs.at(0)/scalingFactor;
    cameraParametrsCur.at(1) = cameraParametrs.at(1)/scalingFactor;
    cameraParametrsCur.at(2) = cameraParametrs.at(2)/scalingFactor;
    cameraParametrsCur.at(3) = cameraParametrs.at(3)/scalingFactor;
    
    // start creating parameters
    SurfaceHeightSaliencyMap parameters_current;
    parameters_current.setWidth(pyramidParameters.pyramidImages.at(i).cols);
    parameters_current.setHeight(pyramidParameters.pyramidImages.at(i).rows);
    
    // create scaled point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudCur(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointIndices::Ptr indicesCur(new pcl::PointIndices());
    EPUtils::Depth2PointCloud(cloudCur,indicesCur,pyramidParameters.pyramidImages.at(i),cameraParametrsCur);
    
    parameters_current.setCloud(cloudCur);
    parameters_current.setIndices(indicesCur);
    parameters_current.setModelCoefficients(coefficients);
    
    parameters_current.calculateSurfaceHeightMap(pyramidParameters.pyramidFeatures.at(i));
  }

  // combine saliency maps
  combinePyramid(pyramidParameters);
  pyramidParameters.map.copyTo(map);
  
  map.copyTo(map_);
  return(AM_OK);
}

float SurfaceHeightSaliencyMap::getHeightCoefficient(int heightType)
{
  switch(heightType)
  {
    case AM_SHORT:
      return(1.0);
    case AM_TALL:
      return(0.0);
  }
  return(0.0);
}

} //namespace AttentionModule