#include <opencv2/opencv.hpp>

#include <iostream>

#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/common/common.h>
#include <pcl/common/distances.h>
#include <pcl/common/intersections.h>

#include "v4r/EPUtils/EPUtils.hpp"
#include "Symmetry3DMap2.hpp"

namespace AttentionModule {
  
pcl::PointXYZ operator+(const pcl::PointXYZ p1, const pcl::PointXYZ p2) 
{
  pcl::PointXYZ p;
  p.x = p1.x + p2.x;
  p.y = p1.y + p2.y;
  p.z = p1.z + p2.z;
  return(p);
}

pcl::Normal operator+(const pcl::Normal n1, const pcl::Normal n2) 
{
  pcl::Normal n;
  n.normal[0] = n1.normal[0] + n2.normal[0];
  n.normal[1] = n1.normal[1] + n2.normal[1];
  n.normal[2] = n1.normal[2] + n2.normal[2];
  return(n);
}

pcl::Normal PointsPair2Vector(const pcl::PointXYZ p1, const pcl::PointXYZ p2) 
{
  pcl::Normal vect;
  vect.normal[0] = p1.x - p2.x;
  vect.normal[1] = p1.y - p2.y;
  vect.normal[2] = p1.z - p2.z;
  
  return(vect);
}

float Distance2PlaneSigned(pcl::Normal vect, pcl::Normal norm)
{
  float dist = vect.normal[0]*norm.normal[0] + vect.normal[1]*norm.normal[1] + vect.normal[2]*norm.normal[2];
  return(dist);
}

Symmetry3DMap::Symmetry3DMap()
{
  cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
  normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>());
  indices = pcl::PointIndices::Ptr(new pcl::PointIndices());
  normalizationType = EPUtils::NT_NONE;
  R = 20;
  filterSize = 5;
  map = cv::Mat_<float>::zeros(0,0);
  width = 0;
  height = 0;
  cameraParametrs.clear();
}

void Symmetry3DMap::setCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_)
{
  cloud = cloud_;
}
  
void Symmetry3DMap::setNormals(pcl::PointCloud<pcl::Normal>::Ptr normals_)
{
  normals = normals_;
}
  
void Symmetry3DMap::setIndices(pcl::PointIndices::Ptr indices_)
{
  indices = indices_;
}
  
void Symmetry3DMap::setR(int R_)
{
  R = R_;
}

  //void setNormalizationType(int normalizationType_);
  //void setFilterSize(int filterSize_);
  
void Symmetry3DMap::setWidth(int width_)
{
  width = width_;
}
  
void Symmetry3DMap::setHeight(int height_)
{
  height = height_;
}

void Symmetry3DMap::setPyramidMode(bool pyramidMode_)
{
  pyramidMode = pyramidMode_;
}

  //void setCameraParameters(std::vector<float> cameraParametrs_);
  //void setPyramidParameters(AttentionModule::PyramidParameters pyramidParameters_);
  
  //pcl::PointCloud<pcl::PointXYZ>::Ptr getCloud();
  //pcl::PointCloud<pcl::Normal>::Ptr getNormals();
  //pcl::PointIndices::Ptr getIndices();


void Symmetry3DMap::createLookUpMap(cv::Mat &lookupMap)
{
  // create normals lookup map
  lookupMap = cv::Mat_<int>::zeros(height,width);
  lookupMap = lookupMap - 1;
  for(unsigned int pi = 0; pi < indices->indices.size(); ++pi)
  {
    int index = indices->indices.at(pi);
    
    int r = index / width;
    int c = index % width;
    
    lookupMap.at<int>(r,c) = pi;
  }
}

void Symmetry3DMap::checkParameters()
{
  // dummy function
}

void Symmetry3DMap::compute()
{
  if(pyramidMode)
    computePyramid();
  else
    computeSingle();
}

void Symmetry3DMap::computeSingle()
{
  // create normals lookup map
  cv::Mat lookupTableNormals = cv::Mat_<int>::zeros(height,width);
  
  createLookUpMap(lookupTableNormals);
  
  std::vector<cv::Point> shifts;
  
  // create pairs of points
  for(int rr = 0; rr < R/2; ++rr)
  {
    for(int cc = 0; cc < R/2; ++cc)
    {
      int distr = R/2 - rr;
      int distc = R/2 - cc;
      
      shifts.push_back(cv::Point(-distc,-distr));
      shifts.push_back(cv::Point( distc, distr));
      shifts.push_back(cv::Point( distc,-distr));
      shifts.push_back(cv::Point(-distc, distr));
    }
  }
  
  map = cv::Mat_<float>::zeros(height,width);
  
  for(unsigned int idx = 0; idx < indices->indices.size(); ++idx)
  {
    int rr = indices->indices.at(idx) / width;
    int cc = indices->indices.at(idx) % width;
    
    if( (rr < R/2) || (rr >= height - R/2) || (cc < R/2) || (cc >= width - R/2) )
    {
      continue;
    }
  
    pcl::PointCloud<pcl::Normal>::Ptr small_normals(new pcl::PointCloud<pcl::Normal>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr small_cloud(new pcl::PointCloud<pcl::PointXYZ>());

    for(unsigned int pi = 0; pi < shifts.size(); ++pi)
    {
      int r1 = rr + shifts.at(pi).y;
      int c1 = cc + shifts.at(pi).x;
     
      int index = lookupTableNormals.at<int>(r1,c1); //number of the indece
    
      if(index >= 0)
      {
        small_cloud->points.push_back(cloud->points.at(indices->indices.at(index)));
        small_normals->points.push_back(normals->points.at(index));
      }
    }

    small_cloud->width = small_cloud->points.size();
    small_cloud->height = 1;

    if((small_cloud->points.size() <= 0) || (small_normals->points.size() <= 0))
      continue;

    std::vector<pcl::Normal> axis;
    EPUtils::principleAxis(small_normals,axis);

    std::vector<float> W;
    W.resize(axis.size());

    for(unsigned int axis_num = 0; axis_num < axis.size(); ++axis_num)
    {
      W.at(axis_num) = 0;
  
      // create plane
      pcl::Normal plane_normal;
      plane_normal = axis.at(axis_num);
      plane_normal = EPUtils::normalize(plane_normal);

      pcl::PointXYZ point0 = cloud->points.at(indices->indices.at(idx));
     
      float a = plane_normal.normal[0];
      float b = plane_normal.normal[1];
      float c = plane_normal.normal[2];
      float d = -(a*point0.x + b*point0.y + c*point0.z);
 
      pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
      coefficients->values.resize(4);
      coefficients->values.at(0) = a;
      coefficients->values.at(1) = b;
      coefficients->values.at(2) = c;
      coefficients->values.at(3) = d;
    
      pcl::PointCloud<pcl::PointXYZ>::Ptr points_projected (new pcl::PointCloud<pcl::PointXYZ>);
      std::vector<float> distances;
      pcl::PointIndices::Ptr small_indices(new pcl::PointIndices());
      EPUtils::ProjectPointsOnThePlane(coefficients,small_cloud,points_projected,distances,small_indices,false);

      MiddlePoint leftPoint, rightPoint;
  
      for(unsigned int pi = 0; pi < small_cloud->size(); ++pi)
      {
        pcl::PointXYZ point_pi = small_cloud->points.at(pi);
       
        pcl::Normal pip0 = PointsPair2Vector(point_pi,point0);
       
        float dist_to_the_plane = Distance2PlaneSigned(pip0,plane_normal);
       
        if(dist_to_the_plane > 0)
        {
          leftPoint.num += 1;
          leftPoint.distance += distances.at(pi);
    
          leftPoint.normal.normal[0] += small_normals->points.at(pi).normal[0];
          leftPoint.normal.normal[1] += small_normals->points.at(pi).normal[1];
          leftPoint.normal.normal[2] += small_normals->points.at(pi).normal[2];
     
          leftPoint.point.x += point_pi.x;
          leftPoint.point.y += point_pi.y;
          leftPoint.point.z += point_pi.z;
        }
        else if(dist_to_the_plane < 0)
        {
          rightPoint.num += 1;
          rightPoint.distance += distances.at(pi);
   
          rightPoint.normal.normal[0] += small_normals->points.at(pi).normal[0];
          rightPoint.normal.normal[1] += small_normals->points.at(pi).normal[1];
          rightPoint.normal.normal[2] += small_normals->points.at(pi).normal[2];
   
          rightPoint.point.x += point_pi.x;
          rightPoint.point.y += point_pi.y;
          rightPoint.point.z += point_pi.z;
        }
      }

      if((leftPoint.num > 0) && (rightPoint.num > 0))
      {
        leftPoint.distance /= leftPoint.num;
        rightPoint.distance /= rightPoint.num;
 
        float Wi = rightPoint.distance - leftPoint.distance;
        Wi = (Wi > 0 ? Wi : -Wi);
    
        leftPoint.normal.normal[0] /= leftPoint.num;
        leftPoint.normal.normal[1] /= leftPoint.num;
        leftPoint.normal.normal[2] /= leftPoint.num;
     
        rightPoint.normal.normal[0] /= rightPoint.num;
        rightPoint.normal.normal[1] /= rightPoint.num;
        rightPoint.normal.normal[2] /= rightPoint.num;
    
        leftPoint.point.x /= leftPoint.num;
        leftPoint.point.y /= leftPoint.num;
        leftPoint.point.z /= leftPoint.num;
     
        rightPoint.point.x /= rightPoint.num;
        rightPoint.point.y /= rightPoint.num;
        rightPoint.point.z /= rightPoint.num;
       
        pcl::Normal lineNormal;
        lineNormal.normal[0] = leftPoint.point.x - rightPoint.point.x;
        lineNormal.normal[1] = leftPoint.point.y - rightPoint.point.y;
        lineNormal.normal[2] = leftPoint.point.z - rightPoint.point.z;
        lineNormal = EPUtils::normalize(lineNormal);
     
        //pcl::Normal N;
        //EPUtils::calculatePlaneNormal(lineNormal,rightPoint.normal,N);
        //float Ci;
        //EPUtils::calculateCosine(leftPoint.normal,N,Ci);
        //Ci = Ci > 0 ? Ci : -Ci;
        //Ci = sqrt(1-Ci*Ci);
	pcl::Normal N, N2;
        N = EPUtils::calculatePlaneNormal(leftPoint.normal,rightPoint.normal);
	N2 = EPUtils::crossProduct(N,lineNormal);
	float Ci = sqrt(N2.normal[0]*N2.normal[0] + N2.normal[1]*N2.normal[1] + N2.normal[2]*N2.normal[2]);
        //EPUtils::calculateCosine(leftPoint.normal,N,Ci);
        //Ci = Ci > 0 ? Ci : -Ci;
        //Ci = sqrt(1-Ci*Ci);
     
        float d=plane_normal.normal[0]*(leftPoint.point.x-point0.x)+plane_normal.normal[1]*(leftPoint.point.y-point0.y)+plane_normal.normal[2]*(leftPoint.point.z-point0.z);
        float cos_left=0, cos_right=0;
        cos_left = EPUtils::calculateCosine(leftPoint.normal,plane_normal);
        cos_right = EPUtils::calculateCosine(rightPoint.normal,plane_normal);
        bool point_is_ok=true;
        if (d<0)
        {
          // cos_left > 90deg && cos_right < 90deg
          point_is_ok = (cos_left<0) && (cos_right>0); 
        }
        else 
        {
          // cos_left < 90deg && cos_right > 90deg
          point_is_ok = (cos_left>0) && (cos_right<0);
        }
     
        float cos1, cos2;
        cos1 = EPUtils::calculateCosine(lineNormal,leftPoint.normal);
        cos2 = EPUtils::calculateCosine(lineNormal,rightPoint.normal);
        //cos2 = -cos2;
    
        float alpha1 = acos(cos1);
        float alpha2 = acos(cos2);
    
        float Si;
        Si = (1-cos(alpha1+alpha2))*(1-cos(alpha1-alpha2));
     
        float Di = rightPoint.point.z - leftPoint.point.z;
        Di = Di > 0 ? Di : -Di;
     
        if((leftPoint.point.z > 0) && (rightPoint.point.z > 0) && (rightPoint.distance > 0) && (leftPoint.distance > 0) && point_is_ok)
        {
          W.at(axis_num) = exp(-1000*Wi)*exp(-1000*Di)*Si*Ci;
        }
      }
    }

    // calculate number of valid principle planes
    float W_max = 0;
    for(unsigned int axis_num = 0; axis_num < axis.size(); ++axis_num)
    {
      if(W.at(axis_num) > W_max)
      {
        W_max = W.at(axis_num);
      }
    }
  
    map.at<float>(rr,cc) = W_max;

  }
  
  int filter_size = filterSize;
  cv::blur(map,map,cv::Size(filter_size,filter_size));
  
  //EPUtils::normalize(map,normalization_type);
  double minVal, maxVal;
  cv::minMaxLoc(map,&minVal,&maxVal);
  map = (1.0/maxVal)*map;
}

void Symmetry3DMap::computePyramid()
{
  // create depth
  cv::Mat depth;
  EPUtils::PointCloud2Depth(depth,cloud,width,height,indices);
  
  normalizationType = EPUtils::NT_NONMAX;
  pyramidParameters.start_level = 0;
  pyramidParameters.max_level = 2;
  
  cameraParametrs.clear();
  cameraParametrs.resize(4);
  cameraParametrs.at(0) = 525.0f;
  cameraParametrs.at(1) = 525.0f;
  cameraParametrs.at(2) = 319.5f;
  cameraParametrs.at(3) = 239.5f;
  
  pyramidParameters.combination_type = AttentionModule::AM_SIMPLE;
  pyramidParameters.normalization_type = EPUtils::NT_NONMAX;
  
  // calculate puramid with saliency maps
  int max_level = pyramidParameters.max_level + 1;
  pyramidParameters.pyramidImages.clear();
  cv::buildPyramid(depth,pyramidParameters.pyramidImages,max_level);
  pyramidParameters.pyramidFeatures.clear();
  pyramidParameters.pyramidFeatures.resize(pyramidParameters.pyramidImages.size());
  
  //parameters.pyramidParameters.print();
  
  for(int i = pyramidParameters.start_level; i <= pyramidParameters.max_level; ++i)
  {
    int scalingFactor = pow(2.0f,i);
    std::vector<float> cameraParametrs_current;
    cameraParametrs_current.resize(4);
    cameraParametrs_current.at(0) = cameraParametrs.at(0)/scalingFactor;
    cameraParametrs_current.at(1) = cameraParametrs.at(1)/scalingFactor;
    cameraParametrs_current.at(2) = cameraParametrs.at(2)/scalingFactor;
    cameraParametrs_current.at(3) = cameraParametrs.at(3)/scalingFactor;
    
    // start creating parameters
    Symmetry3DMap map_current;
    map_current.setWidth(pyramidParameters.pyramidImages.at(i).cols);
    map_current.setHeight(pyramidParameters.pyramidImages.at(i).rows);
    
    // create scaled point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_current(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointIndices::Ptr indices_current(new pcl::PointIndices());
    EPUtils::Depth2PointCloud(cloud_current,indices_current,pyramidParameters.pyramidImages.at(i),cameraParametrs_current);
    map_current.setCloud(cloud_current);
    map_current.setIndices(indices_current);
    
    //calculate point cloud normals
    pcl::PointCloud<pcl::Normal>::Ptr normals_current(new pcl::PointCloud<pcl::Normal> ());
    if(!pclAddOns::ComputePointNormals<pcl::PointXYZ>(cloud_current,indices_current,normals_current))
      return;//(AttentionModule::AM_NORMALCLOUD);
    map_current.setNormals(normals_current);
    
    //CalculateSymmetry3DMap(parameters_current);
    map_current.computeSingle();
    cv::Mat map_current_image;
    map_current.getMap(map_current_image);
    map_current_image.copyTo(pyramidParameters.pyramidFeatures.at(i));
    map_current_image.copyTo(pyramidParameters.pyramidImages.at(i));
  }

  // combine saliency maps
  combinePyramid(pyramidParameters);
  pyramidParameters.map.copyTo(map);
}

/*boost::shared_ptr<pcl::visualization::PCLVisualizer> normalsVis (
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{
  // --------------------------------------------------------
  // -----Open 3D viewer and add point cloud and normals-----
  // --------------------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
  viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal> (cloud, normals, 1, 0.05, "normals");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}*/

} //namespace AttentionModule