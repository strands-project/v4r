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



#include <v4r/recognition/RansacSolvePnPdepth.h>
#include <v4r/reconstruction/impl/projectPointToImage.hpp>
#include <v4r/reconstruction/impl/ReprojectionError.hpp>
#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#if CV_MAJOR_VERSION < 3
#define HAVE_OCV_2
#endif

namespace v4r
{

using namespace std;



/************************************************************************************
 * Constructor/Destructor
 */
RansacSolvePnPdepth::RansacSolvePnPdepth(const Parameter &p)
 : param(p)
{
  setParameter(p);
}

RansacSolvePnPdepth::~RansacSolvePnPdepth()
{
}

/**
 * getRandIdx
 */
void RansacSolvePnPdepth::getRandIdx(int size, int num, std::vector<int> &idx)
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
unsigned RansacSolvePnPdepth::countInliers(const std::vector<cv::Point3f> &points, const std::vector<cv::Point2f> &_im_points, const std::vector<float> &_inv_depth, const Eigen::Matrix4f &pose)
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

    if ((im_pt - Eigen::Map<const Eigen::Vector2f>(&_im_points[i].x)).squaredNorm() < sqr_inl_dist_px &&
        (isnan(_inv_depth[i]) || _inv_depth[i]-(1./pt3[2]) < param.inl_dist_z) )
    {
      cnt++;
    }
  }

  return cnt;
}

/**
 * getInliers
 */
void RansacSolvePnPdepth::getInliers(const std::vector<cv::Point3f> &points, const std::vector<cv::Point2f> &_im_points, const std::vector<float> &_inv_depth, const Eigen::Matrix4f &pose, std::vector<int> &_inliers)
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

    if ( (im_pt - Eigen::Map<const Eigen::Vector2f>(&_im_points[i].x)).squaredNorm() < sqr_inl_dist_px &&
         (isnan(_inv_depth[i]) || _inv_depth[i]-(1./pt3[2]) < param.inl_dist_z) )
    {
      _inliers.push_back(i);
    }
  }
}

/**
 * @brief RansacSolvePnPdepth::convertToLM
 * @param points
 * @param pose
 */
void RansacSolvePnPdepth::convertToLM(const std::vector<cv::Point3f> &points, Eigen::Matrix4f &pose)
{
  Eigen::Matrix3d R = pose.topLeftCorner<3, 3>().cast<double>();
  ceres::RotationMatrixToAngleAxis(&R(0,0), &pose_Rt(0));
  pose_Rt.tail<3>() = pose.block<3,1>(0, 3).cast<double>();
  points3d.resize(points.size());
  for (unsigned i=0; i<points.size(); i++)
    points3d[i] = Eigen::Vector3f::Map(&points[i].x).cast<double>();
}

/**
 * @brief RansacSolvePnPdepth::convertFromLM
 * @param pose
 */
void RansacSolvePnPdepth::convertFromLM(Eigen::Matrix4f &pose)
{
  Eigen::Matrix3d R;
  ceres::AngleAxisToRotationMatrix(&pose_Rt(0), &R(0,0));
  pose.topLeftCorner<3, 3>() = R.cast<float>();
  pose.block<3,1>(0, 3) = pose_Rt.tail<3>().cast<float>();

}

/**
 * @brief RansacSolvePnPdepth::optimizePoseLM
 * @param _points3d
 * @param _im_points
 * @param _inv_depth
 * @param _pose_Rt
 * @param _inliers
 */
void RansacSolvePnPdepth::optimizePoseLM(std::vector<Eigen::Vector3d> &_points3d, const std::vector<cv::Point2f> &_im_points, const std::vector<float> &_inv_depth, Eigen::Matrix<double, 6, 1> &_pose_Rt, const std::vector<int> &_inliers)
{
  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);

  if (lm_intrinsics.size()==4) // no distortions
  {
    for (unsigned i=0; i<_points3d.size(); i++)
    {
      if (isnan(_inv_depth[i]))
      {
        problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction< NoDistortionReprojectionError, 3, 4, 6, 3 >(
                new NoDistortionReprojectionError(_im_points[i].x,_im_points[i].y)),
              (param.use_robust_loss?new ceres::CauchyLoss(param.loss_scale):NULL), &lm_intrinsics[0], &_pose_Rt[0], &_points3d[i][0]);
      }
      else
      {
        problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction< NoDistortionReprojectionAndDepthError, 3, 4, 6, 3 >(
                new NoDistortionReprojectionAndDepthError(_im_points[i].x,_im_points[i].y, _inv_depth[i], param.depth_error_scale)),
              (param.use_robust_loss?new ceres::CauchyLoss(param.loss_scale):NULL), &lm_intrinsics[0], &_pose_Rt[0], &_points3d[i][0]);
      }
      problem.SetParameterBlockConstant(&_points3d[i][0]);
    }
  }
  else if (lm_intrinsics.size()==9) // radial distortions
  {
    for (unsigned i=0; i<_points3d.size(); i++)
    {
      if (isnan(_inv_depth[i]))
      {
        problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction< RadialDistortionReprojectionError, 3, 9, 6, 3 >(
                new RadialDistortionReprojectionError(_im_points[i].x,_im_points[i].y)),
              (param.use_robust_loss?new ceres::CauchyLoss(param.loss_scale):NULL), &lm_intrinsics[0], &_pose_Rt[0], &_points3d[i][0]);
      }
      else
      {
        problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction< RadialDistortionReprojectionAndDepthError, 3, 9, 6, 3 >(
                new RadialDistortionReprojectionAndDepthError(_im_points[i].x,_im_points[i].y, _inv_depth[i], param.depth_error_scale)),
              (param.use_robust_loss?new ceres::CauchyLoss(param.loss_scale):NULL), &lm_intrinsics[0], &_pose_Rt[0], &_points3d[i][0]);
      }
      problem.SetParameterBlockConstant(&_points3d[i][0]);
    }
  }
  else cout<<"hmm, not supported intrinsics!"<<endl;

  problem.SetParameterBlockConstant(&lm_intrinsics[0]);

    // Configure the solver.
  ceres::Solver::Options options;
  options.use_nonmonotonic_steps = true;
  options.preconditioner_type = ceres::SCHUR_JACOBI;
  options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.use_inner_iterations = false;
  options.max_num_iterations = 100;

//  options.minimizer_progress_to_stdout = false;
  options.minimizer_progress_to_stdout = true;

  // Solve!
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << "Final report:\n" << summary.FullReport();
}




/******************************* PUBLIC ***************************************/



/**
 * ransacSolvePnP
 */
int RansacSolvePnPdepth::ransac(const std::vector<cv::Point3f> &points, const std::vector<cv::Point2f> &_im_points, Eigen::Matrix4f &pose, std::vector<int> &_inliers, const std::vector<float> &_depth)
{
  int k=0;
  float sig=param.nb_ransac_points, sv_sig=0.;
  float eps = sig/(float)points.size();
  std::vector<int> indices;
  std::vector<cv::Point3f> model_pts(param.nb_ransac_points);
  std::vector<cv::Point2f> query_pts(param.nb_ransac_points);
  cv::Mat_<double> R(3,3), rvec, tvec, sv_rvec, sv_tvec;
  inv_depth.resize(_im_points.size(),std::numeric_limits<float>::quiet_NaN());
  for (unsigned i=0; i<_depth.size(); i++)
    if (!isnan(_depth[i]) && _depth[i]>std::numeric_limits<float>::epsilon()) inv_depth[i] = 1./_depth[i];
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

    sig = countInliers(points, _im_points, inv_depth, pose);

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
  getInliers(points, _im_points, inv_depth, pose, _inliers);

  convertToLM(points, pose);

  optimizePoseLM(points3d, _im_points, inv_depth, pose_Rt, _inliers);

  convertFromLM(pose);

  getInliers(points, _im_points, inv_depth, pose, _inliers);

  //if (!dbg.empty()) cout<<"Num ransac trials: "<<k<<endl;
  return k;
}





/**
 * setSourceCameraParameter
 */
void RansacSolvePnPdepth::setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs)
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
  if (!_dist_coeffs.empty()) {
    lm_intrinsics.resize(9);
    lm_intrinsics[4] = dist_coeffs(0,0);
    lm_intrinsics[5] = dist_coeffs(0,1);
    lm_intrinsics[6] = dist_coeffs(0,5);
    lm_intrinsics[7] = dist_coeffs(0,2);
    lm_intrinsics[8] = dist_coeffs(0,3);
  } else lm_intrinsics.resize(4);
  lm_intrinsics[0] = intrinsic(0,0);
  lm_intrinsics[1] = intrinsic(1,1);
  lm_intrinsics[2] = intrinsic(0,2);
  lm_intrinsics[3] = intrinsic(1,2);
}

/**
 * @brief RansacSolvePnPdepth::setParameter
 * @param _p
 */
void RansacSolvePnPdepth::setParameter(const Parameter &_p)
{
  param = _p;
  sqr_inl_dist_px = param.inl_dist_px*param.inl_dist_px;

#ifdef HAVE_OCV_2
if (param.pnp_method==INT_MIN) param.pnp_method = cv::P3P;
#else
if (param.pnp_method==INT_MIN) param.pnp_method = cv::SOLVEPNP_P3P;
#endif
}




}












