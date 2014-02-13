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

#include "SubsamplePointCloud2.hh"

namespace surface 
{

using namespace std;

float SubsamplePointCloud2::NaN  = std::numeric_limits<float>::quiet_NaN(); 

/********************** SubsamplePointCloud2 ************************
 * Constructor/Destructor
 */
SubsamplePointCloud2::SubsamplePointCloud2(Parameter p)
{
  setParameter(p);
}

SubsamplePointCloud2::~SubsamplePointCloud2()
{}


/************************** PRIVATE ************************/

void SubsamplePointCloud2::SubsampleMean(pcl::PointCloud<pcl::PointXYZRGB> &in, pcl::PointCloud<pcl::PointXYZRGB> &out)
{
  #pragma omp parallel for
  for (int v=0; v<(int)out.height; v++)
  {
    Eigen::Vector4f mean;
    int uIn, vIn, cnt;

    for (int u=0; u<(int)out.width; u++)
    {
      cnt=0;
      mean.setZero();
      uIn = u*2;
      vIn = v*2;
      pcl::PointXYZRGB &pt0 = in(uIn, vIn);

      if (!IsNaN(pt0)) {
        for (int y = vIn; y<=vIn+1; y++) {
          for (int x = uIn; x<=uIn+1; x++) {
            if (x>=0 && x<width && y>=0 && y<height) {
              pcl::PointXYZRGB &pt = in(x,y);
              if (!IsNaN(pt) && SqrDistance(pt0,pt) < sqrDist) {
                mean += pt.getVector4fMap();
                cnt++;
              }
            }
          }
        }
      }

      pcl::PointXYZRGB &ptOut = out(u,v);
      if (cnt>0) {
        mean /= (float)cnt;
        ptOut.x = mean[0];
        ptOut.y = mean[1];
        ptOut.z = mean[2];
        ptOut.rgb = pt0.rgb;
      }
      else {
        ptOut = pt0;
      }
    }
  }
}



/************************** PUBLIC *************************/

/**
 * setInputCloud
 */
void SubsamplePointCloud2::setInputCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_cloud)
{
  if (!_cloud->isOrganized())
    throw std::runtime_error ("[SubsamplePointCloud2::setInputCloud] Point cloud needs to be organized!");

  cloud = _cloud;
  width = cloud->width;
  height = cloud->height;
}

/**
 * Subsample point cloud
 */
void SubsamplePointCloud2::compute()
{
  if (cloud.get() == 0)
    throw std::runtime_error ("[SubsamplePointCloud2::compute] No point cloud available!");

  // init
  if (resCloud.get()==0 || (&*resCloud) == (&*cloud) )
    resCloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
  resCloud->header   = cloud->header;
  resCloud->width    = cloud->width/2;
  resCloud->height   = cloud->height/2;
  resCloud->is_dense = cloud->is_dense;
  resCloud->resize(resCloud->width*resCloud->height);

  SubsampleMean(*cloud,*resCloud);
}

/**
 * get resized cloud
 */
void SubsamplePointCloud2::getCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_cloud)
{
  _cloud = resCloud;
}

/**
 * setParameter
 */
void SubsamplePointCloud2::setParameter(Parameter p)
{
  param = p;
  sqrDist = pow(param.dist, 2);
}


} //-- THE END --

