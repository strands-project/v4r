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

#ifndef KP_OCCLUSION_CLUSTERING_HH
#define KP_OCCLUSION_CLUSTERING_HH

#include <iostream>
#include <vector>
#include <queue>
#include <set>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#ifndef Q_MOC_RUN
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#endif


namespace v4r
{



/**
 * OcclusionClustering
 */
class OcclusionClustering
{
public:

  /**
   * @brief The Parameter class
   */
  class Parameter
  {
  public:
    double thr_std_dev;
    Parameter()
      : thr_std_dev(0.02) {}
  };


private:
  std::vector<cv::Point> queue;
  std::vector<Eigen::Vector3f> contour;
  std::vector<cv::Point> points;
  std::vector<cv::Point> nbs;

  void clusterNaNs(const cv::Point &_start, const pcl::PointCloud<pcl::PointXYZRGB> &_cloud, cv::Mat_<unsigned char> &_mask, std::vector<Eigen::Vector3f> &_contour, std::vector<cv::Point> &_points);
  double getDepthVariance(const std::vector<Eigen::Vector3f> &_contour);

  inline bool isnan(const pcl::PointXYZRGB &pt);

public:
  Parameter param;
  OcclusionClustering(const Parameter &_p=Parameter());
  ~OcclusionClustering();

  /** get occlusion mask (occlusion from stereo) **/
  void compute(const pcl::PointCloud<pcl::PointXYZRGB> &_cloud, cv::Mat_<unsigned char> &_mask);

  typedef boost::shared_ptr< ::v4r::OcclusionClustering> Ptr;
  typedef boost::shared_ptr< ::v4r::OcclusionClustering const> ConstPtr;
};


/**
 * @brief OcclusionClustering::isnan
 * @param pt
 * @return
 */
inline bool OcclusionClustering::isnan(const pcl::PointXYZRGB &pt)
{
  if (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z))
    return true;
  return false;
}

}

#endif

