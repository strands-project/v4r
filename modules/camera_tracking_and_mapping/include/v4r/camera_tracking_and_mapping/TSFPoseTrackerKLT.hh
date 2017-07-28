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

#ifndef KP_TSF_POSE_TRACKER_KLT_HH
#define KP_TSF_POSE_TRACKER_KLT_HH

#include <iostream>
#include <fstream>
#include <float.h>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <boost/thread/mutex.hpp>
#include <boost/thread.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <boost/shared_ptr.hpp>
#include <v4r/recognition/RansacSolvePnPdepth.h>
#include <v4r/common/impl/DataMatrix2D.hpp>
#include <v4r/camera_tracking_and_mapping/TSFData.h>
#include <v4r/core/macros.h>


namespace v4r
{



/**
 * TSFPoseTrackerKLT
 */
class V4R_EXPORTS TSFPoseTrackerKLT
{
public:

  /**
   * Parameter
   */
  class Parameter
  {
  public:
    cv::TermCriteria termcrit;
    cv::Size win_size;
    cv::Size subpix_win_size;
    int max_count;
    double pcent_reinit;
    double conf_tracked_points_norm;
    v4r::RansacSolvePnPdepth::Parameter rt;
    Parameter()
      : termcrit(cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.03)), win_size(cv::Size(31,31)),
        subpix_win_size(cv::Size(10,10)), max_count(500), pcent_reinit(0.5), conf_tracked_points_norm(250),
        rt(v4r::RansacSolvePnPdepth::Parameter(1.5, 0.01, 5000, INT_MIN, 4, 0.015)){}
  };


 

private:
  Parameter param;

  cv::Mat_<double> intrinsic;

  std::vector<uchar> status;
  std::vector<float> err;
  std::vector<int> inliers;

  bool run, have_thread;

  boost::thread th_obectmanagement;
  boost::thread th_init;

  TSFData *data;

  v4r::RansacSolvePnPdepth::Ptr pnp;
  std::vector<int> plk_converged;
  std::vector<float> depth;

  double kal_dt;
  cv::Mat_<double> measurement;
  cv::KalmanFilter kalmanFilter;

  void operate();


  void getImage(const v4r::DataMatrix2D<Surfel> &cloud, cv::Mat &im);
  bool needReinit(const std::vector<cv::Point2f> &points);
  bool trackCamera(double &conf_ransac_iter, double &conf_tracked_points);
  void getPoints3D(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const std::vector<cv::Point2f> &points, std::vector<Eigen::Vector3f> &points3d);
  void filterValidPoints3D(std::vector<cv::Point2f> &points, std::vector<Eigen::Vector3f> &points3d);
  void filterValidPoints3D(std::vector<cv::Point2f> &pts1, std::vector<Eigen::Vector3f> &pts3d1, std::vector<cv::Point2f> &pts2, std::vector<Eigen::Vector3f> &pts3d2);
  void filterInliers(std::vector<cv::Point2f> &pts1, std::vector<Eigen::Vector3f> &pts3d1, std::vector<cv::Point2f> &pts2, std::vector<Eigen::Vector3f> &pts3d2, std::vector<int> &inliers);
  void filterConverged(std::vector<cv::Point2f> &pts1, std::vector<Eigen::Vector3f> &pts3d1, std::vector<cv::Point2f> &pts2, std::vector<Eigen::Vector3f> &pts3d2, std::vector<int> &converged);

  void initKalmanFilter( cv::KalmanFilter &kf, double dt);
  void updateKalmanFilter( cv::KalmanFilter &kf, const Eigen::Matrix4f &pose, Eigen::Matrix4f &kal_pose, bool have_pose );

  cv::Mat euler2rot(const cv::Mat & euler);
  cv::Mat rot2euler(const cv::Mat & rotationMatrix);

  inline float getInterpolated(const v4r::DataMatrix2D<v4r::Surfel> &cloud, const cv::Point2f &pt);
  inline float getInterpolated(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const cv::Point2f &pt);
  inline float sqr(const float &d) {return d*d;}


public:
  cv::Mat dbg;

  TSFPoseTrackerKLT(const Parameter &p=Parameter());
  ~TSFPoseTrackerKLT();

  void start();
  void stop();

  inline bool isStarted() {return have_thread;}

  void reset();

  void setData(TSFData *_data) { data = _data; }
  void track(double &conf_ransac_iter, double &conf_tracked_points);

  void setCameraParameter(const cv::Mat &_intrinsic);
  void setParameter(const Parameter &p);

  typedef boost::shared_ptr< ::v4r::TSFPoseTrackerKLT> Ptr;
  typedef boost::shared_ptr< ::v4r::TSFPoseTrackerKLT const> ConstPtr;
};



/*************************** INLINE METHODES **************************/


/**
 * @brief TSFPoseTrackerKLT::getInterpolated
 * @param cloud
 * @param pt
 * @return
 */
inline float TSFPoseTrackerKLT::getInterpolated(const v4r::DataMatrix2D<v4r::Surfel> &cloud, const cv::Point2f &pt)
{
  int xt = (int) pt.x;
  int yt = (int) pt.y;
  float ax = pt.x - xt;
  float ay = pt.y - yt;
  float d=0;
  float sn=0, n;

  if (!std::isnan(cloud(yt,xt).pt[0]))
  {
    n = (1.-ax) * (1.-ay);
    d += n * cloud(yt,xt).pt[2];
    sn += n;
  }
  if (!std::isnan(cloud(yt,xt+1).pt[0]))
  {
    n = ax * (1.-ay);
    d += n * cloud(yt,xt+1).pt[2];
    sn += n;
  }
  if (!std::isnan(cloud(yt+1,xt).pt[0]))
  {
    n = (1.-ax) * ay;
    d += n * cloud(yt+1,xt).pt[2];
    sn += n;
  }
  if (!std::isnan(cloud(yt+1,xt+1).pt[0]))
  {
    n = ax * ay;
    d += n * cloud(yt+1,xt+1).pt[2];
    sn += n;
  }

  return (d>0.?d/sn:std::numeric_limits<float>::quiet_NaN());
}

/**
 * @brief TSFPoseTrackerKLT::getInterpolated
 * @param cloud
 * @param pt
 * @return
 */
inline float TSFPoseTrackerKLT::getInterpolated(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const cv::Point2f &pt)
{
  int xt = (int) pt.x;
  int yt = (int) pt.y;
  float ax = pt.x - xt;
  float ay = pt.y - yt;
  float d=0;
  float sn=0, n;

  if (!std::isnan(cloud(xt,yt).x))
  {
    n = (1.-ax) * (1.-ay);
    d += n * cloud(xt,yt).z;
    sn += n;
  }
  if (!std::isnan(cloud(xt+1,yt).x))
  {
    n = ax * (1.-ay);
    d += n * cloud(xt+1, yt).z;
    sn += n;
  }
  if (!std::isnan(cloud(xt, yt+1).x))
  {
    n = (1.-ax) * ay;
    d += n * cloud(xt, yt+1).z;
    sn += n;
  }
  if (!std::isnan(cloud(xt+1, yt+1).x))
  {
    n = ax * ay;
    d += n * cloud(xt+1,yt+1).z;
    sn += n;
  }

  return (d>0.?d/sn:std::numeric_limits<float>::quiet_NaN());
}


} //--END--

#endif

