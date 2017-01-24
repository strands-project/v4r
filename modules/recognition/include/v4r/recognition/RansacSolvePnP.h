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


#ifndef KP_RANSAC_SOLVE_PNP_HH
#define KP_RANSAC_SOLVE_PNP_HH

#include <stdio.h>
#include <string>
#include <stdexcept>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>
#include <v4r/core/macros.h>



namespace v4r
{


/**
 * RansacSolvePnP
 */
class V4R_EXPORTS RansacSolvePnP
{
public:
  class Parameter
  {
  public:
    double inl_dist;
    double eta_ransac;               // eta for pose ransac
    unsigned max_rand_trials;         // max. number of trials for pose ransac
    int pnp_method;            // cv::ITERATIVE, cv::P3P
    int nb_ransac_points;
    Parameter(double _inl_dist=3, double _eta_ransac=0.01, unsigned _max_rand_trials=5000,
      int _pnp_method=INT_MIN, int _nb_ransac_points=4)
    : inl_dist(_inl_dist), eta_ransac(_eta_ransac), max_rand_trials(_max_rand_trials),
      pnp_method(_pnp_method), nb_ransac_points(_nb_ransac_points) {}
  };


private:
  Parameter param;

  float sqr_inl_dist;

  cv::Mat_<double> dist_coeffs;
  cv::Mat_<double> intrinsic;
  
  cv::Mat_<unsigned char> im_gray;
  std::vector< cv::Point2f > im_points;
  std::vector< int > inliers;


  void getRandIdx(int size, int num, std::vector<int> &idx);
  unsigned countInliers(const std::vector<cv::Point3f> &points, const std::vector<cv::Point2f> &im_points, const Eigen::Matrix4f &pose);
  void getInliers(const std::vector<cv::Point3f> &points, const std::vector<cv::Point2f> &im_points, const Eigen::Matrix4f &pose, std::vector<int> &inliers);


  inline void cvToEigen(const cv::Mat_<double> &R, const cv::Mat_<double> &t, Eigen::Matrix4f &pose);
  inline bool contains(const std::vector<int> &idx, int num);



public:
  cv::Mat dbg;

  RansacSolvePnP(const Parameter &p=Parameter());
  ~RansacSolvePnP();

  int ransacSolvePnP(const std::vector<cv::Point3f> &points, const std::vector<cv::Point2f> &im_points, Eigen::Matrix4f &pose, std::vector<int> &inliers);
  void setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs);
  void setParameter(const Parameter &_p);

  typedef boost::shared_ptr< ::v4r::RansacSolvePnP> Ptr;
  typedef boost::shared_ptr< ::v4r::RansacSolvePnP const> ConstPtr;
};


/***************************** inline methods *******************************/
/**
 * cvToEigen
 */
inline void RansacSolvePnP::cvToEigen(const cv::Mat_<double> &R, const cv::Mat_<double> &t, Eigen::Matrix4f &pose)
{
  pose.setIdentity();

  pose(0,0) = R(0,0); pose(0,1) = R(0,1); pose(0,2) = R(0,2);
  pose(1,0) = R(1,0); pose(1,1) = R(1,1); pose(1,2) = R(1,2);
  pose(2,0) = R(2,0); pose(2,1) = R(2,1); pose(2,2) = R(2,2);

  pose(0,3) = t(0,0);
  pose(1,3) = t(1,0);
  pose(2,3) = t(2,0);
}

inline bool RansacSolvePnP::contains(const std::vector<int> &idx, int num)
{
  for (unsigned i=0; i<idx.size(); i++)
    if (idx[i]==num)
      return true;
  return false;
}




} //--END--

#endif

