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

#ifndef KP_TSF_DATA_HH
#define KP_TSF_DATA_HH

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <boost/thread/mutex.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <v4r/common/impl/DataMatrix2D.hpp> 
#include <v4r/camera_tracking_and_mapping/Surfel.hh>
#include <v4r/camera_tracking_and_mapping/TSFFrame.hh>
#include <queue>
#include <v4r/core/macros.h>

namespace v4r
{



/**
 * TSFData
 */
class V4R_EXPORTS TSFData 
{
public:
  boost::mutex mtx_shm;

  //// -> track (tf), -> init-tracking, -> integrate cloud

  bool need_init;
  int init_points;
  int lk_flags;

  cv::Mat image;
  cv::Mat prev_gray, gray;
  pcl::PointCloud<pcl::PointXYZRGB> cloud;  ///// new cloud
  uint64_t timestamp;

  std::vector<cv::Point2f> points[2];
  std::vector<Eigen::Vector3f> points3d[2];
  Eigen::Matrix4f pose;      /// global pose of the current frame (depth, gray, points[1], ....)
  bool have_pose;

  v4r::DataMatrix2D<Surfel>::Ptr filt_cloud;
  Eigen::Matrix4f filt_pose;
  uint64_t filt_timestamp;
  uint64_t kf_timestamp;
  Eigen::Matrix4f kf_pose;   /// pose of the keyframe (points[0], points3d[0], normals, prev_gray

  std::queue<TSFFrame::Ptr> map_frames;
  int cnt_pose_lost_map;
  int nb_frames_integrated;
  Eigen::Matrix4f last_pose_map;

  TSFData();
  ~TSFData();

  void reset();

  inline void lock() { mtx_shm.lock(); }
  inline void unlock() { mtx_shm.unlock(); }

  static void convert(const v4r::DataMatrix2D<v4r::Surfel> &sf_cloud, pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud, const double &thr_weight=-1000000, const double &thr_delta_angle=180. );
  static void convert(const v4r::DataMatrix2D<v4r::Surfel> &sf_cloud, cv::Mat &image);
};



/*************************** INLINE METHODES **************************/

} //--END--

#endif

