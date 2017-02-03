/******************************************************************************
 * Copyright (c) 2016 Johann Prankl
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/

#include "OcclusionClustering.hh"
#include "v4r/common/impl/Vector.hpp"

namespace v4r
{

using namespace std;
  
/********************** OcclusionClustering ************************
 * Constructor/Destructor
 */
OcclusionClustering::OcclusionClustering(const Parameter &_p)
 : param(_p)
{
  nbs.resize(4);
  nbs[0] = cv::Point(-1,0);
  nbs[1] = cv::Point(1,0);
  nbs[2] = cv::Point(0,-1);
  nbs[3] = cv::Point(0,1);
}

OcclusionClustering::~OcclusionClustering()
{
}

/************************** PRIVATE ************************/


/**
 * @brief OcclusionClustering::clusterNaNs
 * @param _cloud
 * @param _mask
 * @param _contour
 */
void OcclusionClustering::clusterNaNs(const cv::Point &_start, const pcl::PointCloud<pcl::PointXYZRGB> &_cloud, cv::Mat_<unsigned char> &_mask, std::vector<Eigen::Vector3f> &_contour, std::vector<cv::Point> &_points)
{
  cv::Point pt0, pt;
  int queue_idx = 0;
  int width = _cloud.width;
  int height = _cloud.height;

  _contour.clear();
  _mask(_start) = 1;
  _points.assign(1,_start);
  queue.assign(1,_start);

  // start clustering
  while (((int)queue.size()) > queue_idx)
  {
    // extract current index
    pt0 = queue.at(queue_idx);
    queue_idx++;


    for (unsigned i=0; i<nbs.size(); i++)
    {
      pt = pt0+nbs[i];

      if ( (pt.x < 0) || (pt.y < 0) || pt.x >= width || pt.y >= height )
        continue;

      if (_mask(pt)!=0) // used point
        continue;

      if(isnan(_cloud(pt.x,pt.y)))
      {
        _mask(pt)=1;
        queue.push_back(pt);
        _points.push_back(pt);
      }
      else _contour.push_back(_cloud(pt.x,pt.y).getVector3fMap());
    }
  }
}

/**
 * @brief OcclusionClustering::getDepthVariance
 * @param _contour
 * @return
 */
double OcclusionClustering::getDepthVariance(const std::vector<Eigen::Vector3f> &_contour)
{
  double var=0, mean = 0;
  for (unsigned i=0; i<_contour.size(); i++)
    mean += _contour[i][2];
  mean /= (double)_contour.size();
  for (unsigned i=0; i<_contour.size(); i++)
    var += sqr((_contour[i][2]-mean));
  var /= (double)_contour.size();
  return var;
}


/************************** PUBLIC *************************/


/**
 * Compute
 */
void OcclusionClustering::compute(const pcl::PointCloud<pcl::PointXYZRGB> &_cloud, cv::Mat_<unsigned char> &_mask)
{
  _mask = cv::Mat_<unsigned char>::zeros(_cloud.height, _cloud.width);
  double thr_var_depth = param.thr_std_dev*param.thr_std_dev;

  for (int v=0; v<(int)_cloud.height; v++)
  {
    for (int u=0; u<(int)_cloud.width; u++)
    {
      if (isnan(_cloud(u,v)) && _mask(v,u)==0)
      {
        clusterNaNs(cv::Point(u,v), _cloud, _mask, contour, points);
        if (contour.size()>=3 && getDepthVariance(contour)>thr_var_depth)
        {
          for (unsigned i=0; i<points.size(); i++)
            _mask(points[i]) = 255;
        }
      }
    }
  }
}


} //-- THE END --

