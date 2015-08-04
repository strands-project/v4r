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
 * @author Johann Prankl, Aitor Aldoma
 *
 */

#ifndef KP_REFINE_PROJECTED_POINT_LOCATION_LK_BASE_HH
#define KP_REFINE_PROJECTED_POINT_LOCATION_LK_BASE_HH

#include <vector>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <v4r/common/impl/SmartPtr.hpp>

namespace v4r
{

/**
 * RefineProjectedPointLocationLKbase
 */
class RefineProjectedPointLocationLKbase
{
public:

public:
  RefineProjectedPointLocationLKbase(){};
  virtual ~RefineProjectedPointLocationLKbase(){};

  virtual void setSourceImage(const cv::Mat_<unsigned char> &_im_src, const Eigen::Matrix4f &_pose_src) {};
  virtual void setTargetImage(const cv::Mat_<unsigned char> &_im_tgt, const Eigen::Matrix4f &_pose_tgt) {};
  virtual void refineImagePoints(const std::vector<Eigen::Vector3f> &pts, 
        const std::vector<Eigen::Vector3f> &normals, 
        std::vector<cv::Point2f> &im_pts_tgt, std::vector<int> &converged) {};

  virtual void setSourceCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs) {};
  virtual void setTargetCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs) {};

  typedef SmartPtr< ::v4r::RefineProjectedPointLocationLKbase> Ptr;
  typedef SmartPtr< ::v4r::RefineProjectedPointLocationLKbase const> ConstPtr;
};




/*********************** INLINE METHODES **************************/


}

#endif

