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



#include <v4r/recognition/RansacSolvePnP.h>
#include <v4r/reconstruction/impl/projectPointToImage.hpp>
#include <iostream>

#if CV_MAJOR_VERSION < 3
#define HAVE_OCV_2
#endif

namespace v4r
{

using namespace std;



/************************************************************************************
 * Constructor/Destructor
 */
RansacSolvePnP::RansacSolvePnP(const Parameter &p)
 : param(p)
{
  setParameter(p);
}

RansacSolvePnP::~RansacSolvePnP()
{
}

/**
 * getRandIdx
 */
void RansacSolvePnP::getRandIdx(int size, int num, std::vector<int> &idx)
{
  int temp;
  idx.clear();
  for (int i=0; i<num; i++)
  {
    do{
      temp = rand()%size;
    }while(contains(idx,temp));
    idx.push_back(temp);
  }
}

/**
 * countInliers
 */
unsigned RansacSolvePnP::countInliers(const std::vector<cv::Point3f> &points, const std::vector<cv::Point2f> &_im_points, const Eigen::Matrix4f &pose)
{
  unsigned cnt=0;

  Eigen::Vector2f im_pt;
  Eigen::Vector3f pt3;
  bool have_dist = !dist_coeffs.empty();
  
  Eigen::Matrix3f R = pose.topLeftCorner<3, 3>();
  Eigen::Vector3f t = pose.block<3,1>(0, 3);

  for (unsigned i=0; i<points.size(); i++)
  {
    pt3 = R*Eigen::Map<const Eigen::Vector3f>(&points[i].x) + t;

    if (have_dist)
      projectPointToImage(&pt3[0], intrinsic.ptr<double>(), dist_coeffs.ptr<double>(), &im_pt[0]);
    else projectPointToImage(&pt3[0], intrinsic.ptr<double>(), &im_pt[0]);

    if ((im_pt - Eigen::Map<const Eigen::Vector2f>(&_im_points[i].x)).squaredNorm() < sqr_inl_dist)
    {
      cnt++;
    }
  }

  return cnt;
}

/**
 * getInliers
 */
void RansacSolvePnP::getInliers(const std::vector<cv::Point3f> &points, const std::vector<cv::Point2f> &_im_points, const Eigen::Matrix4f &pose, std::vector<int> &_inliers)
{
  Eigen::Vector2f im_pt;
  Eigen::Vector3f pt3;
  bool have_dist = !dist_coeffs.empty();
  
  Eigen::Matrix3f R = pose.topLeftCorner<3, 3>();
  Eigen::Vector3f t = pose.block<3,1>(0, 3);

  _inliers.clear();

  for (unsigned i=0; i<points.size(); i++)
  {
    pt3 = R*Eigen::Map<const Eigen::Vector3f>(&points[i].x) + t;

    if (have_dist)
      projectPointToImage(&pt3[0], intrinsic.ptr<double>(), dist_coeffs.ptr<double>(), &im_pt[0]);
    else projectPointToImage(&pt3[0], intrinsic.ptr<double>(), &im_pt[0]);

    if ((im_pt - Eigen::Map<const Eigen::Vector2f>(&_im_points[i].x)).squaredNorm() < sqr_inl_dist)
    {
      _inliers.push_back(i);
    }
  }
}





/******************************* PUBLIC ***************************************/



/**
 * ransacSolvePnP
 */
int RansacSolvePnP::ransacSolvePnP(const std::vector<cv::Point3f> &points, const std::vector<cv::Point2f> &_im_points, Eigen::Matrix4f &pose, std::vector<int> &_inliers)
{
  int k=0;
  float sig=param.nb_ransac_points, sv_sig=0.;
  float eps = sig/(float)points.size();
  std::vector<int> indices;
  std::vector<cv::Point3f> model_pts(param.nb_ransac_points);
  std::vector<cv::Point2f> query_pts(param.nb_ransac_points);
  cv::Mat_<double> R(3,3), rvec, tvec, sv_rvec, sv_tvec;
  _inliers.clear();

  while (pow(1. - pow(eps,param.nb_ransac_points), k) >= param.eta_ransac && k < (int)param.max_rand_trials)
  {
    getRandIdx(points.size(), param.nb_ransac_points, indices);

    for (unsigned i=0; i<indices.size(); i++)
    {
      model_pts[i] = points[indices[i]];
      query_pts[i] = _im_points[indices[i]];
    }

    cv::solvePnP(cv::Mat(model_pts), cv::Mat(query_pts), intrinsic, dist_coeffs, rvec, tvec, false, param.pnp_method);

    cv::Rodrigues(rvec, R);
    cvToEigen(R, tvec, pose);

    sig = countInliers(points, _im_points, pose);

    if (sig > sv_sig)
    {
      sv_sig = sig;
      rvec.copyTo(sv_rvec);
      tvec.copyTo(sv_tvec);
      eps = sv_sig / (float)points.size();
    }

    k++;
  }

  if (sv_sig<4) return INT_MAX;

  cv::Rodrigues(sv_rvec, R);
  cvToEigen(R, sv_tvec, pose);
  getInliers(points, _im_points, pose, _inliers);

  model_pts.resize(_inliers.size());
  query_pts.resize(_inliers.size());

  for (unsigned i=0; i<_inliers.size(); i++)
  {
    model_pts[i] = points[_inliers[i]];
    query_pts[i] = _im_points[_inliers[i]];
  }

  #ifdef HAVE_OCV_2
  cv::solvePnP(cv::Mat(model_pts), cv::Mat(query_pts), intrinsic, dist_coeffs, sv_rvec, sv_tvec, true, cv::ITERATIVE );
  #else
  cv::solvePnP(cv::Mat(model_pts), cv::Mat(query_pts), intrinsic, dist_coeffs, sv_rvec, sv_tvec, true, cv::SOLVEPNP_ITERATIVE );
  #endif

  cv::Rodrigues(sv_rvec, R);
  cvToEigen(R, sv_tvec, pose);

  getInliers(points, _im_points, pose, _inliers);

  //if (!dbg.empty()) cout<<"Num ransac trials: "<<k<<endl;
  return k;
}





/**
 * setSourceCameraParameter
 */
void RansacSolvePnP::setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs)
{
  dist_coeffs = cv::Mat_<double>();
  if (_intrinsic.type() != CV_64F)
    _intrinsic.convertTo(intrinsic, CV_64F);
  else intrinsic = _intrinsic;
  if (!_dist_coeffs.empty())
  {
    dist_coeffs = cv::Mat_<double>::zeros(1,8);
    for (int i=0; i<_dist_coeffs.cols*_dist_coeffs.rows; i++)
      dist_coeffs(0,i) = _dist_coeffs.at<double>(0,i);
  }
}

/**
 * @brief RansacSolvePnP::setParameter
 * @param _p
 */
void RansacSolvePnP::setParameter(const Parameter &_p)
{
  param = _p;
  sqr_inl_dist = param.inl_dist*param.inl_dist;

#ifdef HAVE_OCV_2
if (param.pnp_method==INT_MIN) param.pnp_method = cv::P3P;
#else
if (param.pnp_method==INT_MIN) param.pnp_method = cv::SOLVEPNP_P3P;
#endif
}




}












