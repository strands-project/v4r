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

#include <v4r/reconstruction/ProjLKPoseTrackerLM.h>
#include <v4r/reconstruction/impl/projectPointToImage.hpp>
#include <v4r/reconstruction/impl/ReprojectionError.hpp>

namespace v4r
{

using namespace std;


/************************************************************************************
 * Constructor/Destructor
 */
ProjLKPoseTrackerLM::ProjLKPoseTrackerLM(const Parameter &p)
 : param(p)
{ 
  plk.reset(new RefineProjectedPointLocationLK(p.plk_param) );

  sqr_inl_dist = param.inl_dist*param.inl_dist;
}

ProjLKPoseTrackerLM::~ProjLKPoseTrackerLM()
{
}

/**
 * getRandIdx
 */
void ProjLKPoseTrackerLM::getRandIdx(int size, int num, std::vector<int> &idx)
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
unsigned ProjLKPoseTrackerLM::countInliers(const std::vector<Eigen::Vector3d> &points, const std::vector<cv::Point2f> &im_points, const Eigen::Matrix4d &pose)
{
  unsigned cnt=0;

  Eigen::Vector2f im_pt;
  Eigen::Vector3d pt3;
  bool have_dist = !tgt_dist_coeffs.empty();
  
  Eigen::Matrix3d R = pose.topLeftCorner<3, 3>();
  Eigen::Vector3d t = pose.block<3,1>(0, 3);

  for (unsigned i=0; i<points.size(); i++)
  {
    pt3 = R*points[i] + t;

    if (have_dist)
      projectPointToImage(&pt3[0], tgt_intrinsic.ptr<double>(), tgt_dist_coeffs.ptr<double>(), &im_pt[0]);
    else projectPointToImage(&pt3[0], tgt_intrinsic.ptr<double>(), &im_pt[0]);

    if ((im_pt - Eigen::Map<const Eigen::Vector2f>(&im_points[i].x)).squaredNorm() < sqr_inl_dist)
    {
      cnt++;
    }
  }

  return cnt;
}

/**
 * getInliers
 */
void ProjLKPoseTrackerLM::getInliers(const std::vector<Eigen::Vector3d> &points, const std::vector<cv::Point2f> &im_points, const Eigen::Matrix4d &pose, std::vector<int> &inliers)
{
  Eigen::Vector2f im_pt;
  Eigen::Vector3d pt3;
  bool have_dist = !tgt_dist_coeffs.empty();
  
  Eigen::Matrix3d R = pose.topLeftCorner<3, 3>();
  Eigen::Vector3d t = pose.block<3,1>(0, 3);

  inliers.clear();

  for (unsigned i=0; i<points.size(); i++)
  {
    pt3 = R*points[i] + t;

    if (have_dist)
      projectPointToImage(&pt3[0], tgt_intrinsic.ptr<double>(), tgt_dist_coeffs.ptr<double>(), &im_pt[0]);
    else projectPointToImage(&pt3[0], tgt_intrinsic.ptr<double>(), &im_pt[0]);

    if ((im_pt - Eigen::Map<const Eigen::Vector2f>(&im_points[i].x)).squaredNorm() < sqr_inl_dist)
    {
      inliers.push_back(i);
    }
  }
}


/**
 * optimizePoseLM
 */
void ProjLKPoseTrackerLM::optimizePoseLM(const std::vector<Eigen::Vector3d> &points, const std::vector<int> &pt_indices, const std::vector<cv::Point2f> &im_points, const std::vector<int> &im_indices, Eigen::Matrix4d &pose)
{
  Eigen::Matrix3d R = pose.topLeftCorner<3, 3>();
  Eigen::Vector3d t = pose.block<3,1>(0, 3);
  Eigen::Matrix<double, 6, 1> pose_Rt;
  ceres::RotationMatrixToAngleAxis(&R(0,0), &pose_Rt(0));           
  pose_Rt.tail<3>() = t;  

  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);

  if (lm_intrinsics.size()==4) // no distortions
  {
    for (unsigned i=0; i<pt_indices.size(); i++) {
      const cv::Point2f &im_pt = im_points[im_indices[i]];
      double *pt3 = (double*)&points[pt_indices[i]][0];
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction< NoDistortionReprojectionError, 2, 4, 6, 3 >(
          new NoDistortionReprojectionError(im_pt.x,im_pt.y)), NULL,&lm_intrinsics[0],&pose_Rt[0],pt3);
      problem.SetParameterBlockConstant(pt3);
    }
  } 
  else if (lm_intrinsics.size()==9) // radial distortions
  {
    for (unsigned i=0; i<pt_indices.size(); i++) {
      const cv::Point2f &im_pt = im_points[im_indices[i]];
      double *pt3 = (double*)&points[pt_indices[i]][0];
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction< RadialDistortionReprojectionError, 2, 9, 6, 3 >(
          new RadialDistortionReprojectionError(im_pt.x,im_pt.y)), NULL,&lm_intrinsics[0],&pose_Rt[0],pt3);
      problem.SetParameterBlockConstant(pt3);
    }
  }
  
  problem.SetParameterBlockConstant(&lm_intrinsics[0]);

    // Configure the solver.
  ceres::Solver::Options options;
  options.use_nonmonotonic_steps = true;
  options.preconditioner_type = ceres::SCHUR_JACOBI;
  options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.use_inner_iterations = false;
  options.max_num_iterations = 100;

  //if (dbg.empty())
    options.minimizer_progress_to_stdout = false;
  //else options.minimizer_progress_to_stdout = true;

  // Solve!
  ceres::Solver::Summary summary;

  ceres::Solve(options, &problem, &summary);

  // copy back pose 
  ceres::AngleAxisToRotationMatrix(&pose_Rt(0), &R(0,0));
  t = pose_Rt.tail<3>();
  pose.topLeftCorner<3, 3>() = R;
  pose.block<3,1>(0, 3) = t;

  //if (!dbg.empty()) std::cout << "Final report:\n" << summary.FullReport();
}

/**
 * optimizePoseRobustLossLM
 */
void ProjLKPoseTrackerLM::optimizePoseRobustLossLM(const std::vector<Eigen::Vector3d> &points, const std::vector<int> &pt_indices, const std::vector<cv::Point2f> &im_points, const std::vector<int> &im_indices, Eigen::Matrix4d &pose)
{
  Eigen::Matrix3d R = pose.topLeftCorner<3, 3>();
  Eigen::Vector3d t = pose.block<3,1>(0, 3);
  Eigen::Matrix<double, 6, 1> pose_Rt;
  ceres::RotationMatrixToAngleAxis(&R(0,0), &pose_Rt(0));
  pose_Rt.tail<3>() = t;

  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);

  if (lm_intrinsics.size()==4) // no distortions
  {
    for (unsigned i=0; i<pt_indices.size(); i++) {
      const cv::Point2f &im_pt = im_points[im_indices[i]];
      double *pt3 = (double*)&points[pt_indices[i]][0];
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction< NoDistortionReprojectionError, 2, 4, 6, 3 >(
          new NoDistortionReprojectionError(im_pt.x,im_pt.y)), new ceres::CauchyLoss(param.loss_scale),&lm_intrinsics[0],&pose_Rt[0],pt3);
      problem.SetParameterBlockConstant(pt3);
    }
  }
  else if (lm_intrinsics.size()==9) // radial distortions
  {
    for (unsigned i=0; i<pt_indices.size(); i++) {
      const cv::Point2f &im_pt = im_points[im_indices[i]];
      double *pt3 = (double*)&points[pt_indices[i]][0];
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction< RadialDistortionReprojectionError, 2, 9, 6, 3 >(
          new RadialDistortionReprojectionError(im_pt.x,im_pt.y)), new ceres::CauchyLoss(param.loss_scale),&lm_intrinsics[0],&pose_Rt[0],pt3);
      problem.SetParameterBlockConstant(pt3);
    }
  }
  
  problem.SetParameterBlockConstant(&lm_intrinsics[0]);

    // Configure the solver.
  ceres::Solver::Options options;
  options.use_nonmonotonic_steps = true;
  options.preconditioner_type = ceres::SCHUR_JACOBI;
  options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.use_inner_iterations = false;
  options.max_num_iterations = 100;

  //if (dbg.empty())
    options.minimizer_progress_to_stdout = false;
  //else options.minimizer_progress_to_stdout = true;

  // Solve!
  ceres::Solver::Summary summary;

  ceres::Solve(options, &problem, &summary);

  // copy back pose 
  ceres::AngleAxisToRotationMatrix(&pose_Rt(0), &R(0,0));
  t = pose_Rt.tail<3>();
  pose.topLeftCorner<3, 3>() = R;
  pose.block<3,1>(0, 3) = t;

  //if (!dbg.empty()) std::cout << "Final report:\n" << summary.FullReport();
}


/**
 * ransacPoseLM
 */
void ProjLKPoseTrackerLM::ransacPoseLM(const std::vector<Eigen::Vector3d> &points, const std::vector<cv::Point2f> &im_points, Eigen::Matrix4d &pose, std::vector<int> &inliers)
{
  int k=0;
  float sig=4, sv_sig=0.;
  float eps = sig/(float)points.size();
  std::vector<int> indices;
  Eigen::Matrix4d tmp_pose=pose;

  while (pow(1. - pow(eps,4), k) >= param.eta_ransac && k < (int)param.max_rand_trials)
  {
    getRandIdx(points.size(), 4, indices);

    optimizePoseLM(points, indices, im_points, indices, tmp_pose); 

    sig = countInliers(points, im_points, tmp_pose);

    if (sig > sv_sig)
    {
      sv_sig = sig;
      pose = tmp_pose;
      eps = sv_sig / (float)points.size();
    } else tmp_pose = pose;

    k++;
  }

  if (!dbg.empty()) cout<<"Num ransac trials: "<<k<<endl;

  if (sv_sig<4) return;

  getInliers(points, im_points, pose, inliers);
  if (param.use_robust_loss)
    optimizePoseRobustLossLM(points, inliers, im_points, inliers, pose);
  else optimizePoseLM(points, inliers, im_points, inliers, pose);
  getInliers(points, im_points, pose, inliers);
}




/******************************* PUBLIC ***************************************/

/**
 * detect
 */
double ProjLKPoseTrackerLM::detect(const cv::Mat &image, Eigen::Matrix4f &pose)
{
  inliers.clear();

  if (model.get()==0)
    throw std::runtime_error("[ProjLKPoseTrackerLM::detect] No model available!");
  if (src_intrinsic.empty()||tgt_intrinsic.empty())
    throw std::runtime_error("[ProjLKPoseTrackerLM::detect] Intrinsic camera parameter not set!");


  if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
  else im_gray = image;

  ObjectView &m = *model;

  std::vector<Eigen::Vector3d> model_pts;
  std::vector<cv::Point2f> query_pts;
  std::vector<Eigen::Vector3f> points, normals;
  Eigen::Matrix4d pose_64f;
  std::vector<int> lk_inliers, lm_inliers;

  m.getPoints(points);
  m.getNormals(normals);

  // tracking
  plk->setTargetImage(im_gray,pose);

  plk->refineImagePoints(points, normals, im_points, converged);

  for (unsigned z=0; z<im_points.size(); z++)
  {
    if (converged[z]==1)
    {
      lk_inliers.push_back(z);
      const Eigen::Vector3f &pt = points[z];
      query_pts.push_back(im_points[z]);
      model_pts.push_back(Eigen::Vector3d(pt[0],pt[1],pt[2]));
      if (!dbg.empty()) cv::line(dbg,model->keys[z].pt, im_points[z],CV_RGB(255,0,0));
    }
  }

  if (int(query_pts.size())<4) return 0.;

  for (unsigned v=0; v<4; v++)
    for (unsigned u=0; u<4; u++)
      pose_64f(v,u) = pose(v,u);

  if (param.use_ransac)
  {
    ransacPoseLM(model_pts, query_pts, pose_64f, lm_inliers);
  } 
  else 
  {
    vector<int> indices(model_pts.size());
    for (unsigned i=0; i<indices.size(); i++) indices[i]=i;

    if (param.use_robust_loss)
      optimizePoseRobustLossLM(model_pts, indices, query_pts, indices, pose_64f);
    else optimizePoseLM(model_pts, indices, query_pts, indices, pose_64f);
  
    getInliers(model_pts, query_pts, pose_64f, lm_inliers);
  }

  if (int(lm_inliers.size())<4) return 0.;

  for (unsigned i=0; i<lm_inliers.size(); i++) {
    inliers.push_back(lk_inliers[lm_inliers[i]]);
    if (!dbg.empty()) cv::circle(dbg,im_points[inliers.back()],2,CV_RGB(0,255,0));
  }

  for (unsigned v=0; v<4; v++)
    for (unsigned u=0; u<4; u++)
      pose(v,u) = pose_64f(v,u);

  pose_R = pose.topLeftCorner<3, 3>();
  pose_t = pose.block<3,1>(0, 3);

  return double(inliers.size())/double(model->points.size());
}

/**
 * getProjections
 * @param im_pts <model_point_index, projection>
 */
void ProjLKPoseTrackerLM::getProjections(std::vector< std::pair<int,cv::Point2f> > &im_pts)
{
  im_pts.clear();
  if (im_points.size()!=model->points.size() || inliers.size()>im_points.size())
    return;

  for (unsigned i=0; i<inliers.size(); i++)
    im_pts.push_back(make_pair(inliers[i],im_points[inliers[i]]));
}

/**
 * setModel
 */
void ProjLKPoseTrackerLM::setModel(const ObjectView::Ptr &_model, const Eigen::Matrix4f &_pose) 
{  
  model=_model; 

  im_points.resize(model->keys.size());
  plk->setSourceImage(model->image, _pose);
}


/**
 * setSourceCameraParameter
 */
void ProjLKPoseTrackerLM::setSourceCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs)
{
  src_dist_coeffs = cv::Mat_<double>();
  if (_intrinsic.type() != CV_64F)
    _intrinsic.convertTo(src_intrinsic, CV_64F);
  else src_intrinsic = _intrinsic;
  if (!_dist_coeffs.empty())
  {
    src_dist_coeffs = cv::Mat_<double>::zeros(1,8);
    for (int i=0; i<_dist_coeffs.cols*_dist_coeffs.rows; i++)
      src_dist_coeffs(0,i) = _dist_coeffs.at<double>(0,i);
  }
  plk->setSourceCameraParameter(src_intrinsic,src_dist_coeffs);
}

/**
 * setTargetCameraParameter
 */
void ProjLKPoseTrackerLM::setTargetCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs)
{
  tgt_dist_coeffs = cv::Mat_<double>();
  if (_intrinsic.type() != CV_64F)
    _intrinsic.convertTo(tgt_intrinsic, CV_64F);
  else tgt_intrinsic = _intrinsic;
  if (!_dist_coeffs.empty())
  {
    tgt_dist_coeffs = cv::Mat_<double>::zeros(1,8);
    for (int i=0; i<_dist_coeffs.cols*_dist_coeffs.rows; i++)
      tgt_dist_coeffs(0,i) = _dist_coeffs.at<double>(0,i);
  }
  plk->setTargetCameraParameter(tgt_intrinsic,tgt_dist_coeffs);

  if (!_dist_coeffs.empty()) {
    lm_intrinsics.resize(9);
    lm_intrinsics[4] = tgt_dist_coeffs(0,0);
    lm_intrinsics[5] = tgt_dist_coeffs(0,1);
    lm_intrinsics[6] = tgt_dist_coeffs(0,5);
    lm_intrinsics[7] = tgt_dist_coeffs(0,2);
    lm_intrinsics[8] = tgt_dist_coeffs(0,3);
  } else lm_intrinsics.resize(4);
  lm_intrinsics[0] = tgt_intrinsic(0,0);
  lm_intrinsics[1] = tgt_intrinsic(1,1);
  lm_intrinsics[2] = tgt_intrinsic(0,2);
  lm_intrinsics[3] = tgt_intrinsic(1,2);
}



}












