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

#ifndef KP_TSF_FRAME_HH
#define KP_TSF_FRAME_HH

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <boost/shared_ptr.hpp>
#include <v4r/keypoints/impl/triple.hpp>
#include <v4r/common/impl/DataMatrix2D.hpp>
#include <v4r/camera_tracking_and_mapping/Surfel.hh>
#include <v4r/core/macros.h>



namespace v4r
{



/**
 * TSFFrame
 */
class V4R_EXPORTS TSFFrame 
{
public:
  int idx;
  Eigen::Matrix4f pose;
  Eigen::Matrix4f delta_cloud_rgb_pose;
  v4r::DataMatrix2D<Surfel> sf_cloud;

  std::vector<cv::Point2f> points;
  std::vector<Eigen::Vector3f> points3d;
  std::vector<Eigen::Vector3f> normals;

  cv::Mat descs;
  std::vector<cv::KeyPoint> keys;
  std::vector<Eigen::Vector3f> keys3d;

  int fw_link, bw_link;
  std::vector<int> loop_links;

  bool have_track;

  std::vector< std::vector< v4r::triple<int, cv::Point2f, Eigen::Vector3f > > > projections;

  TSFFrame();
  TSFFrame(const int &_idx, const Eigen::Matrix4f &_pose, const v4r::DataMatrix2D<Surfel> &_sf_cloud, bool _have_track);
  ~TSFFrame();

  typedef boost::shared_ptr< ::v4r::TSFFrame> Ptr;
  typedef boost::shared_ptr< ::v4r::TSFFrame const> ConstPtr;
};



/*************************** INLINE METHODES **************************/

} //--END--

#endif

