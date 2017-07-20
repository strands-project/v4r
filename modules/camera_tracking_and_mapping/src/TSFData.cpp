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


#include <v4r/camera_tracking_and_mapping/TSFData.h>
#include <v4r/common/convertImage.h>




namespace v4r
{


using namespace std;




/************************************************************************************
 * Constructor/Destructor
 */
TSFData::TSFData()
 : need_init(false), init_points(0), lk_flags(0), timestamp(std::numeric_limits<uint64_t>::max()), pose(Eigen::Matrix4f::Identity()), have_pose(false), filt_pose(Eigen::Matrix4f::Identity()), filt_timestamp(std::numeric_limits<uint64_t>::max()), kf_timestamp(std::numeric_limits<uint64_t>::max()), kf_pose(Eigen::Matrix4f::Identity()), cnt_pose_lost_map(0), nb_frames_integrated(0)
{
  filt_cloud.reset(new DataMatrix2D<Surfel>() );
  last_pose_map(0,0) = std::numeric_limits<float>::quiet_NaN();
}

TSFData::~TSFData()
{
}




/***************************************************************************************/

/**
 * @brief TSFData::reset
 */
void TSFData::reset()
{
  lock();
  need_init = false;
  init_points = 0;
  lk_flags = 0;
  gray = cv::Mat();
  prev_gray = cv::Mat();
  points[0].clear(); points[1].clear();
  points3d[0].clear(); points3d[1].clear();
  cloud.clear();
  pose = Eigen::Matrix4f::Identity();
  kf_pose = Eigen::Matrix4f::Identity();
  filt_cloud.reset(new DataMatrix2D<Surfel>() );
  filt_pose = Eigen::Matrix4f::Identity();
  timestamp = std::numeric_limits<uint64_t>::max();
  filt_timestamp = std::numeric_limits<uint64_t>::max();
  kf_timestamp=std::numeric_limits<uint64_t>::max();
  have_pose = false;
  map_frames = std::queue<TSFFrame::Ptr>();
  cnt_pose_lost_map = 0;
  last_pose_map(0,0) = std::numeric_limits<float>::quiet_NaN();
  unlock();
}


/**
 * @brief convert
 * @param sf_cloud
 * @param cloud
 * @param thr_weight
 * @param thr_delta_angle
 */
void TSFData::convert(const v4r::DataMatrix2D<v4r::Surfel> &sf_cloud, pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud, const double &thr_weight, const double &thr_delta_angle )
{
  cloud.resize(sf_cloud.data.size());
  cloud.width = sf_cloud.cols;
  cloud.height = sf_cloud.rows;
  cloud.is_dense = false;
  double cos_rad_thr_delta_angle = cos(thr_delta_angle*M_PI/180.);
  for (unsigned i=0; i<sf_cloud.data.size(); i++)
  {
    const v4r::Surfel &s = sf_cloud.data[i];
    pcl::PointXYZRGBNormal &o = cloud.points[i];
    if (s.weight>=thr_weight && s.n.dot(-s.pt.normalized()) > cos_rad_thr_delta_angle )
    {
      o.getVector3fMap() = s.pt;
      o.getNormalVector3fMap() = s.n;
    }
    else
    {
      o.getVector3fMap() = Eigen::Vector3f(std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN());
      o.getNormalVector3fMap() = Eigen::Vector3f(std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN());
    }
    o.r = s.r;
    o.g = s.g;
    o.b = s.b;
  }
}


/**
 * @brief TSFData::convert
 * @param sf_cloud
 * @param image
 */
void TSFData::convert(const v4r::DataMatrix2D<v4r::Surfel> &sf_cloud, cv::Mat &image)
{
  image = cv::Mat_<cv::Vec3b>(sf_cloud.rows, sf_cloud.cols);

  for (int v = 0; v < sf_cloud.rows; v++)
  {
    for (int u = 0; u < sf_cloud.cols; u++)
    {
      cv::Vec3b &cv_pt = image.at<cv::Vec3b> (v, u);
      const v4r::Surfel &s = sf_cloud(v,u);

      cv_pt[2] = (unsigned char)s.r;
      cv_pt[1] = (unsigned char)s.g;
      cv_pt[0] = (unsigned char)s.b;
    }
  }
}



}












