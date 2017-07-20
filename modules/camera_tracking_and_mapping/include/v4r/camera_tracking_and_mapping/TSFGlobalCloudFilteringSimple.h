/**
 * $Id$
 * 
 * Software License Agreement (GNU General Public License)
 *
 *  Copyright (C) 2015:
 *
 *    Johann Prankl, prankl@acin.tuwien.ac.at
 *    Aitor Aldoma, aldoma@acin.tuwien.ac.at
 *
 *      Automation and Control Institute
 *      Vienna University of Technology
 *      Gusshausstraße 25-29
 *      1170 Vienn, Austria
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
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @author Johann Prankl
 *
 */

#ifndef KP_TSF_GLOBAL_CLOUD_FILTER_SIMPLE_HH
#define KP_TSF_GLOBAL_CLOUD_FILTER_SIMPLE_HH

#include <iostream>
#include <fstream>
#include <float.h>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <boost/shared_ptr.hpp>
#include <v4r/common/impl/DataMatrix2D.hpp>
#include <v4r/camera_tracking_and_mapping/PoissonTriangulation.hh>
#include <v4r/camera_tracking_and_mapping/TSFData.h>
#include <v4r/core/macros.h>


namespace v4r
{



/**
 * TSFGlobalCloudFilteringSimple
 */
class V4R_EXPORTS TSFGlobalCloudFilteringSimple
{
public:

  /**
   * Parameter
   */
  class Parameter
  {
  public:
    float voxel_size;            // voxel size for the final filtering
    double thr_weight;           // e.g. 10    // surfel threshold for the final model
    double thr_delta_angle;      // e.g. 80°
    double poisson_depth;
    int samples_per_node;
    bool crop_mesh;
    int erosion_size;
    bool filter_largest_cluster;
    Parameter()
      : voxel_size(0.001), thr_weight(10), thr_delta_angle(75), poisson_depth(7), samples_per_node(2), crop_mesh(true), erosion_size(7), filter_largest_cluster(false) {}
  };


private:
  Parameter param;

  Eigen::Matrix4f base_transform;
  Eigen::Vector3f bb_min, bb_max;

  cv::Mat_<double> intrinsic;
  cv::Mat_<double> dist_coeffs;

  cv::Mat_<unsigned char> tmp_mask;

  void getMask(const v4r::DataMatrix2D<v4r::Surfel> &sf_cloud, cv::Mat_<unsigned char> &mask);
  void filterCluster(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud_filt);


public:

  TSFGlobalCloudFilteringSimple(const Parameter &p=Parameter());
  ~TSFGlobalCloudFilteringSimple();

  void getGlobalCloudFiltered(const std::vector< v4r::TSFFrame::Ptr > &frames, pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud);
  void getMesh(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &cloud, pcl::PolygonMesh &mesh);

  void setBaseTransform(const Eigen::Matrix4f &transform);
  void setROI(const Eigen::Vector3f &bb_lowerleft, const Eigen::Vector3f &bb_upperright);

  void setParameter(const Parameter &p);
  void setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs);

  inline void getInterpolatedRGB(const v4r::DataMatrix2D<v4r::Surfel> &cloud, const cv::Point2f &pt, unsigned char &r, unsigned char &g, unsigned char &b);

  typedef boost::shared_ptr< ::v4r::TSFGlobalCloudFilteringSimple> Ptr;
  typedef boost::shared_ptr< ::v4r::TSFGlobalCloudFilteringSimple const> ConstPtr;
};



/*************************** INLINE METHODES **************************/
/**
 * @brief TSFGlobalCloudFilteringSimple::getInterpolatedRGB
 * @param cloud
 * @param pt
 * @param r
 * @param g
 * @param b
 */
inline void TSFGlobalCloudFilteringSimple::getInterpolatedRGB(const v4r::DataMatrix2D<v4r::Surfel> &cloud, const cv::Point2f &pt, unsigned char &r, unsigned char &g, unsigned char &b)
{
  if (pt.x<cloud.cols-1 && pt.y<cloud.rows-1)
  {
    int xt = (int) pt.x;
    int yt = (int) pt.y;
    float ax = pt.x - xt;
    float ay = pt.y - yt;
    r = (unsigned char)( (1.-ax) * (1.-ay) * cloud(yt,xt).r +
                          ax     * (1.-ay) * cloud(yt,xt+1).r +
                         (1.-ax) *  ay     * cloud(yt+1,xt).r +
                          ax     *  ay     * cloud(yt+1,xt+1).r );
    g = (unsigned char)( (1.-ax) * (1.-ay) * cloud(yt,xt).g +
                          ax     * (1.-ay) * cloud(yt,xt+1).g +
                         (1.-ax) *  ay     * cloud(yt+1,xt).g +
                          ax     *  ay     * cloud(yt+1,xt+1).g );
    b = (unsigned char)( (1.-ax) * (1.-ay) * cloud(yt,xt).b +
                          ax     * (1.-ay) * cloud(yt,xt+1).b +
                         (1.-ax) *  ay     * cloud(yt+1,xt).b +
                          ax     *  ay     * cloud(yt+1,xt+1).b );
  }
  else if( pt.x<cloud.cols && pt.y<cloud.rows )
  {
    const v4r::Surfel &sf = cloud((int)pt.y,(int)pt.x);
    r = (unsigned char)sf.r;
    g = (unsigned char)sf.g;
    b = (unsigned char)sf.b;
  }
}

} //--END--

#endif

