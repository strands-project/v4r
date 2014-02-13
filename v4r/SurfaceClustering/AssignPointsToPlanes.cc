/**
 *  Copyright (C) 2012  
 *    Andreas Richtsfeld, Johann Prankl, Thomas Mörwald
 *    Automation and Control Institute
 *    Vienna University of Technology
 *    Gusshausstraße 25-29
 *    1170 Vienn, Austria
 *    ari(at)acin.tuwien.ac.at
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see http://www.gnu.org/licenses/
 */


#include "AssignPointsToPlanes.hh"

namespace surface 
{

using namespace std;


template<typename T1,typename T2>
inline T1 Dot3(const T1 v1[3], const T2 v2[3])
{
  return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

/********************** AssignPointsToPlanes ************************
 * Constructor/Destructor
 */
AssignPointsToPlanes::AssignPointsToPlanes(Parameter p)
 : width(0), height(0)
{
  setParameter(p);
}

AssignPointsToPlanes::~AssignPointsToPlanes()
{}



/************************** PRIVATE ************************/
/**
 * IndexToPointSurface
 */
void AssignPointsToPlanes::IndexToPointSurface(pcl::PointCloud<pcl::PointXYZRGB> &cloud, SurfaceModel &surf, int idx)
{
  float a = surf.coeffs[0];
  float b = surf.coeffs[1];
  float c = surf.coeffs[2];
  float d = surf.coeffs[3];
  
  for (unsigned i=0; i<surf.indices.size(); i++)
  {
    if (fabs(Plane::ImpPointDist(a,b,c,d,&cloud.points[surf.indices[i]].x)) < param.inlDist)
      ptsSurface[surf.indices[i]].push_back( idx );
  }
}


/**
 * AssignPoints
 */
void AssignPointsToPlanes::FilterPointsNormalDist(pcl::PointCloud<pcl::Normal> &normals, std::vector<SurfaceModel::Ptr > &planes)
{
  // index to array 
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > nPlanes(planes.size());
  ptsSurface.resize(width*height);
  for(unsigned i=0; i<ptsSurface.size(); i++)
    ptsSurface[i].clear();
  
  for (unsigned i=0; i<planes.size(); i++)
  {
    SurfaceModel &plane = *planes[i];
    IndexToPointSurface(*cloud, plane, i);     // index to ptsPlaneProbs
    plane.indices.clear();
    plane.error.clear();
    plane.probs.clear();
    nPlanes[i] = Eigen::Vector3f(plane.coeffs[0],plane.coeffs[1],plane.coeffs[2]);
    nPlanes[i].normalize();
  }

  // assign points
  float cosAng, maxCosAng;
  int idx=INT_MAX;
  for(unsigned i=0; i<ptsSurface.size(); i++)
  {
    if (ptsSurface[i].size()>0)
    {
      maxCosAng=-FLT_MAX;
      std::vector<int> &surf = ptsSurface[i];

      for (unsigned j=0; j<surf.size(); j++)
      {
        pcl::Normal &n = normals.points[i];
        if (n.normal[0] == n.normal[0])
        {
          cosAng=Dot3(&n.normal[0], &nPlanes[surf[j]][0]);
          if (cosAng>maxCosAng)
          {
            maxCosAng = cosAng;
            idx = surf[j];
          }
        }
      }
      if (maxCosAng > 0)
        planes[idx]->indices.push_back(i);
    }
  }

  // filter small planes
  std::vector<SurfaceModel::Ptr > tmp;
  for (unsigned i=0; i<planes.size(); i++)
  {
    if (planes[i]->indices.size() >= (unsigned)param.minPoints)
      tmp.push_back(planes[i]);
  }
 
  planes = tmp; 
}

/**
 * ClusterNormals
 */
void AssignPointsToPlanes::ClusterNeighbours(pcl::PointCloud<pcl::PointXYZRGB> &cloud,std::vector<float> &coeffs, vector<int> &pts)
{
  unsigned idx;
  short x,y;
  queue.clear();

  if (pts.size()==0)
    return;
  
  mcnt++;
  //mask[pts[0]] = mcnt;
  queue=pts;//.push_back(pts[0]);

  while (queue.size()>0)
  {
    idx = queue.back();
    queue.pop_back();
    x = X(idx);
    y = Y(idx);

    for (int v=y-1; v<=y+1; v++)
    {
      for (int u=x-1; u<=x+1; u++)
      {
        if (v>0 && u>0 && v<height && u<width)
        {
          idx = GetIdx(u,v);
          if (mask[idx] != mcnt)
          {
            if(umask[idx]==0)
            {
              pcl::PointXYZRGB &pt = cloud.points[idx];
              if (pt.x==pt.x && fabs(Plane::ImpPointDist(coeffs[0],coeffs[1],coeffs[2],coeffs[3], &pt.x)) < param.inlDist)
              {
                pts.push_back(idx);
                queue.push_back(idx);
              }
            }
            mask[idx] = mcnt;
          }
        }
      }
    }
  }
}


/**
 * AccumulateNeighbours
 */
void AssignPointsToPlanes::AccumulateNeighbours(pcl::PointCloud<pcl::PointXYZRGB> &cloud, vector<SurfaceModel::Ptr> &planes)
{
  mask.clear();
  mask.resize(width*height,0);
  mcnt=0;
  umask.clear();
  umask.resize(width*height,0);

  //set umask
  /*cv::Mat_<uchar> tmp1,tmp2;
  tmp1 = cv::Mat_<uchar>::zeros(height,width);
  tmp2 = cv::Mat_<uchar>::zeros(height,width);*/

  for (unsigned i=0; i<planes.size(); i++)
  {
    SurfaceModel &p = *planes[i];
    for (unsigned j=0; j<p.indices.size(); j++)
    {
      umask[p.indices[j]] = 1;
      //tmp1(Y(p.indices[j]), X(p.indices[j]))=255;
    }
  }

  // add points
  for (unsigned i=0; i<planes.size(); i++)
  {
    ClusterNeighbours(cloud, planes[i]->coeffs, planes[i]->indices);
  }

  
/*for (unsigned i=0; i<planes.size(); i++)
{
  SurfaceModel &p = *planes[i];
  for (unsigned j=0; j<p.indices.size(); j++)
  {
tmp2(Y(p.indices[j]), X(p.indices[j]))=255;
  }
}*/
//cv::imwrite("tmp1.jpg",tmp1);
//cv::imwrite("tmp2.jpg",tmp2);

}

/**
 * ComputeLSPlanes
 */
void AssignPointsToPlanes::ComputeLSPlanes(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, std::vector<SurfaceModel::Ptr> &planes)
{
  pcl::SampleConsensusModelPlane<pcl::PointXYZRGB> lsPlane(cloud);
  Eigen::VectorXf coeffs(4);
  Eigen::Vector3d n0(0., 0., 1.);

  for (unsigned i=0; i<planes.size(); i++)
  {
    SurfaceModel &plane = *planes[i];

    lsPlane.optimizeModelCoefficients(plane.indices, coeffs, coeffs);

    if (Dot3(&coeffs[0], &n0[0]) > 0)
      coeffs*=-1.;

    plane.coeffs.resize(4);
    plane.coeffs[0] = coeffs[0];
    plane.coeffs[1] = coeffs[1];
    plane.coeffs[2] = coeffs[2];
    plane.coeffs[3] = coeffs[3];
  } 
}   



/************************** PUBLIC *************************/

/**
 * compute
 * Attention! 
 * This method assumes plane parameter with normals which are oriented towards the camera! 
 */
void AssignPointsToPlanes::compute(vector<SurfaceModel::Ptr> &planes)
{
  if (cloud.get()==0 || width==0 || height==0) 
    throw std::runtime_error ("[AssignPointsToPlanes::compute] Input point cloud not set!");

  FilterPointsNormalDist(*normals, planes);
  ComputeLSPlanes(cloud, planes);
  AccumulateNeighbours(*cloud, planes);
  FilterPointsNormalDist(*normals, planes);
}

/**
 * setInputNormals
 */
void AssignPointsToPlanes::setInputNormals(const pcl::PointCloud<pcl::Normal>::Ptr &_normals)
{
  normals = _normals;  
}

/**
 * setInputCloud
 */
void AssignPointsToPlanes::setInputCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_cloud)
{
  cloud = _cloud;
  width = cloud->width;
  height = cloud->height;
}

/**
 * setParameter
 */
void AssignPointsToPlanes::setParameter(Parameter p)
{
  param = p;
}



} //-- THE END --

