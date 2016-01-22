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

#ifndef KP_REFINE_PROJECTED_POINT_LOCATION_LK_HH
#define KP_REFINE_PROJECTED_POINT_LOCATION_LK_HH

#include <vector>
#include <iostream>
#include <stdexcept>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <v4r/common/impl/SmartPtr.hpp>
#include <v4r/reconstruction/RefineProjectedPointLocationLKbase.h>

namespace v4r
{

/**
 * RefineProjectedPointLocationLK
 */
class RefineProjectedPointLocationLK : public RefineProjectedPointLocationLKbase
{
public:
  class Parameter
  {
  public:
    float step_factor;        // 2.???
    float min_determinant;
    float min_displacement;   //0.1
    int max_iterations;       //10
    int max_residual;
    float ncc_residual;      // 0.3
    bool use_ncc;
    cv::Size patch_size;
    Parameter(float _step_factor=10., float _min_determinant=0.01, float _min_displacement=0.1, 
      int _max_iterations=10, int _max_residual=15., float _ncc_residual=0.3,
      bool _use_ncc=true, const cv::Size &_patch_size=cv::Size(15,15))
    : step_factor(_step_factor), min_determinant(_min_determinant), min_displacement(_min_displacement),
      max_iterations(_max_iterations), max_residual(_max_residual), ncc_residual(_ncc_residual),
      use_ncc(_use_ncc), patch_size(_patch_size) {}
  };

private:
  Parameter param;

  cv::Mat_<double> src_intrinsic, tgt_intrinsic;
  cv::Mat_<double> src_dist_coeffs, tgt_dist_coeffs;
  Eigen::Matrix3f src_C, tgt_C;


  cv::Mat_<unsigned char> im_src, im_tgt;
  cv::Mat_<float> im_tgt_dx, im_tgt_dy;
  Eigen::Matrix4f pose_src, pose_tgt;
  Eigen::Matrix4f inv_pose_tgt, delta_pose;
  Eigen::Matrix3f R_tgt, delta_R;
  Eigen::Vector3f t_tgt, delta_t;

  std::vector<float> residuals;

  void getIntensityDifference(const cv::Mat_<unsigned char> &im1, cv::Mat_<unsigned char> &im2, 
        const cv::Point2f &pt1, const cv::Point2f &pt2, int width, int height, cv::Mat_<float> &diff);
  void getPatchInterpolated(const cv::Mat_<unsigned char> &image, const cv::Point2f &pt, 
        cv::Mat_<unsigned char> &patch, int width, int height);
  void getGradientSum(const cv::Mat_<float> &dx1, const cv::Mat_<float> &dy1, const cv::Mat_<float> &dx2, 
        const cv::Mat_<float> &dy2, const cv::Point2f &pt1, const cv::Point2f &pt2, int width, int height, 
        cv::Mat_<float> &dx, cv::Mat_<float> &dy);
  void getGradientMatrix22(const cv::Mat_<float> &dx, const cv::Mat_<float> &dy, float &gxx, float &gxy, float &gyy);
  void getErrorVector2(const cv::Mat_<float> &diff, const cv::Mat_<float> &dx, const cv::Mat_<float> &dy, 
        cv::Point2f &err);
  bool solve(const cv::Point2f &err, float gxx, float gxy, float gyy, cv::Point2f &delta);

  inline float getInterpolated(const cv::Mat_<unsigned char> &im, const float &x, const float &y);
  inline float getInterpolated(const cv::Mat_<float> &im, const float &x, const float &y);


public:
  RefineProjectedPointLocationLK(const Parameter &p=Parameter());
  virtual ~RefineProjectedPointLocationLK();

  virtual void setSourceImage(const cv::Mat_<unsigned char> &_im_src, const Eigen::Matrix4f &_pose_src);
  virtual void setTargetImage(const cv::Mat_<unsigned char> &_im_tgt, const Eigen::Matrix4f &_pose_tgt);
  virtual void refineImagePoints(const std::vector<Eigen::Vector3f> &pts, 
        const std::vector<Eigen::Vector3f> &normals, 
        std::vector<cv::Point2f> &im_pts_tgt, std::vector<int> &converged);

  virtual const std::vector<float> &getResiduals()const {return residuals;}

  virtual void setSourceCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs);
  virtual void setTargetCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs);

  typedef SmartPtr< ::v4r::RefineProjectedPointLocationLK> Ptr;
  typedef SmartPtr< ::v4r::RefineProjectedPointLocationLK const> ConstPtr;
};




/*********************** INLINE METHODES **************************/

/** 
 * getInterpolated
 * get a bilinear interpolated pixel
 */
inline float RefineProjectedPointLocationLK::getInterpolated(const cv::Mat_<unsigned char> &im, const float &x, const float &y)
{
  int xt = (int) x;
  int yt = (int) y;
  float ax = x - xt;
  float ay = y - yt;

  float right, top, top_right;
  right = top = top_right = 0.f;

  if ( (xt+1) < im.cols )
      right = im(yt, xt+1);

  if ( (yt+1) < im.rows )
      top = im(yt+1, xt);

  if( ( (yt+1) < im.rows ) &&  ( (xt+1) < im.cols ) )
      top_right = im(yt+1, xt+1);

  return ( (1.-ax) * (1.-ay) * im(yt,xt) + 
            ax     * (1.-ay) * right +
           (1.-ax) *  ay     * top +
            ax     *  ay     * top_right );
}

/** 
 * getInterpolated
 * get a bilinear interpolated pixel
 */
inline float RefineProjectedPointLocationLK::getInterpolated(const cv::Mat_<float> &im, const float &x, const float &y)
{
  int xt = (int) x;
  int yt = (int) y;
  float ax = x - xt;
  float ay = y - yt;

  float right, top, top_right;
  right = top = top_right = 0.f;

  if ( (xt+1) < im.cols )
      right = im(yt, xt+1);

  if ( (yt+1) < im.rows )
      top = im(yt+1, xt);

  if( ( (yt+1) < im.rows ) &&  ( (xt+1) < im.cols ) )
      top_right = im(yt+1, xt+1);

  return ( (1.-ax) * (1.-ay) * im(yt,xt) +
            ax     * (1.-ay) * right +
           (1.-ax) *  ay     * top +
            ax     *  ay     * top_right );
}


}

#endif

