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


#include <v4r/camera_tracking_and_mapping/TSFOptimizeBundle.hh>
#include <v4r/camera_tracking_and_mapping/BACostFunctions.hpp>
#include <v4r/keypoints/impl/invPose.hpp>


namespace v4r
{


using namespace std;




/************************************************************************************
 * Constructor/Destructor
 */
TSFOptimizeBundle::TSFOptimizeBundle(const Parameter &p)
  : const_all_intrinsics(true)
{
  setParameter(p);
}


TSFOptimizeBundle::~TSFOptimizeBundle()
{
}


/**
 * @brief TSFOptimizeBundle::convertPosesToRt
 * @param poses
 */
void TSFOptimizeBundle::convertPosesToRt(const std::vector<TSFFrame::Ptr> &map)
{
  Eigen::Matrix3d R;
  poses_Rt.resize(map.size());
  for (unsigned i=0; i<map.size(); i++)
  {
    R = map[i]->pose.topLeftCorner<3, 3>().cast<double>();
    ceres::RotationMatrixToAngleAxis(&R(0,0), &poses_Rt[i](0));
    poses_Rt[i].tail<3>() = map[i]->pose.block<3,1>(0, 3).cast<double>();
  }
}

/**
 * @brief TSFOptimizeBundle::convertPosesFromRt
 * @param poses
 */
void TSFOptimizeBundle::convertPosesFromRt(std::vector<TSFFrame::Ptr> &map)
{
  if (map.size()!=poses_Rt.size())
    return;

  Eigen::Matrix3d R;
  for (unsigned i=0; i<poses_Rt.size(); i++)
  {
    ceres::AngleAxisToRotationMatrix(&poses_Rt[i](0), &R(0,0));
    map[i]->pose.topLeftCorner<3, 3>() = R.cast<float>();
    map[i]->pose.block<3,1>(0, 3) = poses_Rt[i].tail<3>().cast<float>();
  }
}

/**
 * @brief TSFOptimizeBundle::convertPosesFromRt
 * @param map
 */
void TSFOptimizeBundle::convertPosesFromRtRGB(std::vector<TSFFrame::Ptr> &map)
{
  Eigen::Matrix3d R;
  if (map.size()==poses_Rt_RGB.size())
  {
    Eigen::Matrix4f pose_rgb, inv_pose;
    for (unsigned i=0; i<poses_Rt_RGB.size(); i++)
    {
      pose_rgb.setIdentity();
      ceres::AngleAxisToRotationMatrix(&poses_Rt_RGB[i](0), &R(0,0));
      pose_rgb.topLeftCorner<3, 3>() = R.cast<float>();
      pose_rgb.block<3,1>(0, 3) = poses_Rt_RGB[i].tail<3>().cast<float>();
      invPose(map[i]->pose, inv_pose);
      map[i]->delta_cloud_rgb_pose = pose_rgb*inv_pose;
    }
  }
  else
  {
    for (unsigned i=0; i<map.size(); i++)
    {
      map[i]->delta_cloud_rgb_pose.setIdentity();
      ceres::AngleAxisToRotationMatrix(&delta_pose[0], &R(0,0));
      map[i]->delta_cloud_rgb_pose.topLeftCorner<3, 3>() = R.cast<float>();
      map[i]->delta_cloud_rgb_pose.block<3,1>(0, 3) = delta_pose.tail<3>().cast<float>();
    }
  }
}

/**
 * @brief TSFOptimizeBundle::convertPoints
 */
void TSFOptimizeBundle::convertPoints(const std::vector<TSFFrame::Ptr> &map)
{
  points3d.resize(map.size());
  for (unsigned i=0; i<map.size(); i++)
  {
    const std::vector<Eigen::Vector3f> &pts = map[i]->points3d;
    points3d[i].resize(pts.size());
    for (unsigned j=0; j<points3d[i].size(); j++)
      points3d[i][j] = pts[j].cast<double>();
  }
}

/**
 * @brief TSFOptimizeBundle::optimizePoseRobustLossLM
 * @param dst_im
 * @param dst_3d
 */
void TSFOptimizeBundle::optimizePoses(std::vector<TSFFrame::Ptr> &map)
{
  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);

  if (lm_intrinsics.size()==4) // no distortions
  {
    for (unsigned i=0; i<map.size(); i++)
    {
      TSFFrame &frame = *map[i];
      double *cam0 = &poses_Rt[i][0];
      for (unsigned j=0; j<frame.points3d.size(); j++)
      {
        if (frame.projections[j].size()==0)
          continue;
        Eigen::Vector3d &pt3 = points3d[i][j];
        const Eigen::Vector3f &n0 = frame.normals[j];
        for (unsigned k=0; k<frame.projections[j].size(); k++)
        {
          const cv::Point2f &im_pt = frame.projections[j][k].second;
          double *cam1 = &poses_Rt[frame.projections[j][k].first][0];
          problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction< ReprojectionErrorGlobalPoseCamViewData, 2, 4, 6, 6, 3 >(
                  new ReprojectionErrorGlobalPoseCamViewData(im_pt.x,im_pt.y)),
                    (param.use_robust_loss?new ceres::CauchyLoss(param.loss_scale):NULL), &lm_intrinsics[0], cam0, cam1, &pt3[0]);
          const Eigen::Vector3f &pt3v = frame.projections[j][k].third;
          if (!isnan(pt3v[0]) && !isnan(pt3v[1]) && !isnan(pt3v[2]))
          {
            problem.AddResidualBlock(
                  new ceres::AutoDiffCostFunction< PointToPlaneErrorGlobalPoseCamViewData, 3, 6, 6 >(
                    new PointToPlaneErrorGlobalPoseCamViewData(pt3,n0.cast<double>(),pt3v.cast<double>(),param.depth_error_scale)),
                  (param.use_robust_loss?new ceres::CauchyLoss(param.loss_scale):NULL), cam0, cam1);
          }
        }
        //        problem.SetParameterBlockConstant(&pt3[0]);
      }
    }
  }
  else if (lm_intrinsics.size()==9) // radial distortions
  {
    for (unsigned i=0; i<map.size(); i++)
    {
      TSFFrame &frame = *map[i];
      double *cam0 = &poses_Rt[i][0];
      for (unsigned j=0; j<frame.points3d.size(); j++)
      {
        if (frame.projections[j].size()==0)
          continue;
        Eigen::Vector3d &pt3 = points3d[i][j];
        const Eigen::Vector3f &n0 = frame.normals[j];
        for (unsigned k=0; k<frame.projections[j].size(); k++)
        {
          const cv::Point2f &im_pt = frame.projections[j][k].second;
          double *cam1 = &poses_Rt[frame.projections[j][k].first][0];
          problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction< RadialDistortionReprojectionErrorGlobalPoseCamViewData, 2, 9, 6, 6, 3 >(
                  new RadialDistortionReprojectionErrorGlobalPoseCamViewData(im_pt.x,im_pt.y)),
                    (param.use_robust_loss?new ceres::CauchyLoss(param.loss_scale):NULL), &lm_intrinsics[0], cam0, cam1, &pt3[0]);
          const Eigen::Vector3f &pt3v = frame.projections[j][k].third;
          if (!isnan(pt3v[0]) && !isnan(pt3v[1]) && !isnan(pt3v[2]) && !isnan(n0[0]) && !isnan(n0[1]) && !isnan(n0[2]))
          {
            problem.AddResidualBlock(
                  new ceres::AutoDiffCostFunction< PointToPlaneErrorGlobalPoseCamViewData, 3, 6, 6 >(
                    new PointToPlaneErrorGlobalPoseCamViewData(pt3,n0.cast<double>(),pt3v.cast<double>(),param.depth_error_scale)),
                  (param.use_robust_loss?new ceres::CauchyLoss(param.loss_scale):NULL), cam0, cam1);
          }
        }
        //        problem.SetParameterBlockConstant(&pt3[0]);
      }
    }
  }

  if (const_all_intrinsics)
  {
    problem.SetParameterBlockConstant(&lm_intrinsics[0]);
  }
  else
  {
    if (const_intrinsics.size()>0)
    {
      ceres::SubsetParameterization *subset_parameterization =
            new ceres::SubsetParameterization(9, const_intrinsics);
      problem.SetParameterization(&lm_intrinsics[0], subset_parameterization);
    }
  }


    // Configure the solver.
  ceres::Solver::Options options;
  options.use_nonmonotonic_steps = true;
  options.preconditioner_type = ceres::SCHUR_JACOBI;
  options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.use_inner_iterations = true;
  options.max_num_iterations = 100;

//  options.minimizer_progress_to_stdout = false;
  options.minimizer_progress_to_stdout = true;

  // Solve!
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << "Final report:\n" << summary.FullReport();
}


/**
 * @brief TSFOptimizeBundle::optimizeCloudPosesRGBPoses
 * @param map
 */
void TSFOptimizeBundle::optimizeCloudPosesRGBPoses(std::vector<TSFFrame::Ptr> &map)
{
  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);

  poses_Rt_RGB = poses_Rt;

  if (lm_intrinsics.size()==4) // no distortions
  {
    for (unsigned i=0; i<map.size(); i++)
    {
      TSFFrame &frame = *map[i];
      double *cam0pc = &poses_Rt[i][0];
      double *cam0RGB = &poses_Rt_RGB[i][0];
      for (unsigned j=0; j<frame.points3d.size(); j++)
      {
        if (frame.projections[j].size()==0)
          continue;
        Eigen::Vector3d &pt3RGB = points3d[i][j];
        const Eigen::Vector3f &pt3pc = frame.points3d[j];
        const Eigen::Vector3f &n0pc = frame.normals[j];
        for (unsigned k=0; k<frame.projections[j].size(); k++)
        {
          const cv::Point2f &im_pt = frame.projections[j][k].second;
          double *cam1pc = &poses_Rt[frame.projections[j][k].first][0];
          double *cam1RGB = &poses_Rt_RGB[frame.projections[j][k].first][0];
          problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction< ReprojectionErrorGlobalPoseCamViewData, 2, 4, 6, 6, 3 >(
                  new ReprojectionErrorGlobalPoseCamViewData(im_pt.x,im_pt.y)),
                    (param.use_robust_loss?new ceres::CauchyLoss(param.loss_scale):NULL), &lm_intrinsics[0], cam0RGB, cam1RGB, &pt3RGB[0]);
          const Eigen::Vector3f &pt3v = frame.projections[j][k].third;
          if (!isnan(pt3v[0]) && !isnan(pt3v[1]) && !isnan(pt3v[2]) && !isnan(n0pc[0]) && !isnan(n0pc[1]) && !isnan(n0pc[2]))
          {
            problem.AddResidualBlock(
                  new ceres::AutoDiffCostFunction< PointToPlaneErrorGlobalPoseCamViewData, 3, 6, 6 >(
                    new PointToPlaneErrorGlobalPoseCamViewData(pt3pc.cast<double>(),n0pc.cast<double>(),pt3v.cast<double>(),param.depth_error_scale)),
                  (param.use_robust_loss?new ceres::CauchyLoss(param.loss_scale):NULL), cam0pc, cam1pc);
          }
          if (!isnan(n0pc[0]) && !isnan(n0pc[1]) && !isnan(n0pc[2]))
          {
            problem.AddResidualBlock(
                  new ceres::AutoDiffCostFunction< PointToPlaneErrorGlobalPoseCamViewDataOptiPt1, 3, 6, 6, 3 >(
                    new PointToPlaneErrorGlobalPoseCamViewDataOptiPt1(pt3pc.cast<double>(),n0pc.cast<double>(),param.depth_error_scale)),
                  (param.use_robust_loss?new ceres::CauchyLoss(param.loss_scale):NULL), cam0pc, cam0RGB, &pt3RGB[0]);
          }
        }
//        problem.SetParameterBlockConstant(&pt3RGB[0]);
      }
//      problem.SetParameterBlockConstant(cam0pc);
    }
  }
  else if (lm_intrinsics.size()==9) // radial distortions
  {
    for (unsigned i=0; i<map.size(); i++)
    {
      TSFFrame &frame = *map[i];
      double *cam0pc = &poses_Rt[i][0];
      double *cam0RGB = &poses_Rt_RGB[i][0];
      for (unsigned j=0; j<frame.points3d.size(); j++)
      {
        if (frame.projections[j].size()==0)
          continue;
        Eigen::Vector3d &pt3RGB = points3d[i][j];
        const Eigen::Vector3f &pt3pc = frame.points3d[j];
        const Eigen::Vector3f &n0pc = frame.normals[j];
        for (unsigned k=0; k<frame.projections[j].size(); k++)
        {
          const cv::Point2f &im_pt = frame.projections[j][k].second;
          double *cam1pc = &poses_Rt[frame.projections[j][k].first][0];
          double *cam1RGB = &poses_Rt_RGB[frame.projections[j][k].first][0];
          problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction< RadialDistortionReprojectionErrorGlobalPoseCamViewData, 2, 9, 6, 6, 3 >(
                  new RadialDistortionReprojectionErrorGlobalPoseCamViewData(im_pt.x,im_pt.y)),
                    (param.use_robust_loss?new ceres::CauchyLoss(param.loss_scale):NULL), &lm_intrinsics[0], cam0RGB, cam1RGB, &pt3RGB[0]);
          const Eigen::Vector3f &pt3v = frame.projections[j][k].third;
          if (!isnan(pt3v[0]) && !isnan(pt3v[1]) && !isnan(pt3v[2]) && !isnan(n0pc[0]) && !isnan(n0pc[1]) && !isnan(n0pc[2]))
          {
            problem.AddResidualBlock(
                  new ceres::AutoDiffCostFunction< PointToPlaneErrorGlobalPoseCamViewData, 3, 6, 6 >(
                    new PointToPlaneErrorGlobalPoseCamViewData(pt3pc.cast<double>(),n0pc.cast<double>(),pt3v.cast<double>(),param.depth_error_scale)),
                  (param.use_robust_loss?new ceres::CauchyLoss(param.loss_scale):NULL), cam0pc, cam1pc);
          }
          if (!isnan(n0pc[0]) && !isnan(n0pc[1]) && !isnan(n0pc[2]))
          {
            problem.AddResidualBlock(
                  new ceres::AutoDiffCostFunction< PointToPlaneErrorGlobalPoseCamViewDataOptiPt1, 3, 6, 6, 3 >(
                    new PointToPlaneErrorGlobalPoseCamViewDataOptiPt1(pt3pc.cast<double>(),n0pc.cast<double>(),param.depth_error_scale)),
                  (param.use_robust_loss?new ceres::CauchyLoss(param.loss_scale):NULL), cam0pc, cam0RGB, &pt3RGB[0]);
          }
        }
//        problem.SetParameterBlockConstant(&pt3RGB[0]);
      }
//      problem.SetParameterBlockConstant(cam0pc);
    }
  }

  if (const_all_intrinsics)
  {
    problem.SetParameterBlockConstant(&lm_intrinsics[0]);
  }
  else
  {
    if (const_intrinsics.size()>0)
    {
      ceres::SubsetParameterization *subset_parameterization =
            new ceres::SubsetParameterization(9, const_intrinsics);
      problem.SetParameterization(&lm_intrinsics[0], subset_parameterization);
    }
  }

    // Configure the solver.
  ceres::Solver::Options options;
  options.use_nonmonotonic_steps = true;
  options.preconditioner_type = ceres::SCHUR_JACOBI;
  options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.use_inner_iterations = true;
  options.max_num_iterations = 100;

//  options.minimizer_progress_to_stdout = false;
  options.minimizer_progress_to_stdout = true;

  // Solve!
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << "Final report:\n" << summary.FullReport();
}

/**
 * @brief TSFOptimizeBundle::optimizeCloudPosesRGBPoses
 * @param map
 */
void TSFOptimizeBundle::optimizeCloudPosesDeltaRGBPose(std::vector<TSFFrame::Ptr> &map)
{
  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);

  if (lm_intrinsics.size()==4) // no distortions
  {
    for (unsigned i=0; i<map.size(); i++)
    {
      TSFFrame &frame = *map[i];
      double *cam0pc = &poses_Rt[i][0];
      for (unsigned j=0; j<frame.points3d.size(); j++)
      {
        if (frame.projections[j].size()==0)
          continue;
        Eigen::Vector3d &pt3RGB = points3d[i][j];
        const Eigen::Vector3f &pt3pc = frame.points3d[j];
        const Eigen::Vector3f &n0pc = frame.normals[j];
        for (unsigned k=0; k<frame.projections[j].size(); k++)
        {
          const cv::Point2f &im_pt = frame.projections[j][k].second;
          double *cam1pc = &poses_Rt[frame.projections[j][k].first][0];
          problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction< ReprojectionErrorGlobalPoseDeltaPoseCamViewData, 2, 4, 6, 6, 6, 3 >(
                  new ReprojectionErrorGlobalPoseDeltaPoseCamViewData(im_pt.x,im_pt.y)),
                    (param.use_robust_loss?new ceres::CauchyLoss(param.loss_scale):NULL), &lm_intrinsics[0], cam0pc, cam1pc, &delta_pose[0], &pt3RGB[0]);
          const Eigen::Vector3f &pt3v = frame.projections[j][k].third;
          if (!isnan(pt3v[0]) && !isnan(pt3v[1]) && !isnan(pt3v[2]) && !isnan(n0pc[0]) && !isnan(n0pc[1]) && !isnan(n0pc[2]))
          {
            problem.AddResidualBlock(
                  new ceres::AutoDiffCostFunction< PointToPlaneErrorGlobalPoseCamViewData, 3, 6, 6 >(
                    new PointToPlaneErrorGlobalPoseCamViewData(pt3pc.cast<double>(),n0pc.cast<double>(),pt3v.cast<double>(),param.depth_error_scale)),
                  (param.use_robust_loss?new ceres::CauchyLoss(param.loss_scale):NULL), cam0pc, cam1pc);
          }
          if (!isnan(n0pc[0]) && !isnan(n0pc[1]) && !isnan(n0pc[2]))
          {
            problem.AddResidualBlock(
                  new ceres::AutoDiffCostFunction< PointToPlaneErrorGlobalPoseDeltaPoseCamViewDataOptiPt1, 3, 6, 3 >(
                    new PointToPlaneErrorGlobalPoseDeltaPoseCamViewDataOptiPt1(pt3pc.cast<double>(),n0pc.cast<double>(),param.depth_error_scale)),
                  (param.use_robust_loss?new ceres::CauchyLoss(param.loss_scale):NULL), &delta_pose[0], &pt3RGB[0]);
          }
        }
//        problem.SetParameterBlockConstant(&pt3RGB[0]);
      }
//      problem.SetParameterBlockConstant(cam0pc);
    }

  }
  else if (lm_intrinsics.size()==9) // radial distortions
  {
    for (unsigned i=0; i<map.size(); i++)
    {
      TSFFrame &frame = *map[i];
      double *cam0pc = &poses_Rt[i][0];
      for (unsigned j=0; j<frame.points3d.size(); j++)
      {
        if (frame.projections[j].size()==0)
          continue;
        Eigen::Vector3d &pt3RGB = points3d[i][j];
        const Eigen::Vector3f &pt3pc = frame.points3d[j];
        const Eigen::Vector3f &n0pc = frame.normals[j];
        for (unsigned k=0; k<frame.projections[j].size(); k++)
        {
          const cv::Point2f &im_pt = frame.projections[j][k].second;
          double *cam1pc = &poses_Rt[frame.projections[j][k].first][0];
          problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction< RadialDistortionReprojectionErrorGlobalPoseDeltaPoseCamViewData, 2, 9, 6, 6, 6, 3 >(
                  new RadialDistortionReprojectionErrorGlobalPoseDeltaPoseCamViewData(im_pt.x,im_pt.y)),
                    (param.use_robust_loss?new ceres::CauchyLoss(param.loss_scale):NULL), &lm_intrinsics[0], cam0pc, cam1pc, &delta_pose[0], &pt3RGB[0]);
          const Eigen::Vector3f &pt3v = frame.projections[j][k].third;
          if (!isnan(pt3v[0]) && !isnan(pt3v[1]) && !isnan(pt3v[2]) && !isnan(n0pc[0]) && !isnan(n0pc[1]) && !isnan(n0pc[2]))
          {
            problem.AddResidualBlock(
                  new ceres::AutoDiffCostFunction< PointToPlaneErrorGlobalPoseCamViewData, 3, 6, 6 >(
                    new PointToPlaneErrorGlobalPoseCamViewData(pt3pc.cast<double>(),n0pc.cast<double>(),pt3v.cast<double>(),param.depth_error_scale)),
                  (param.use_robust_loss?new ceres::CauchyLoss(param.loss_scale):NULL), cam0pc, cam1pc);
          }
          if (!isnan(n0pc[0]) && !isnan(n0pc[1]) && !isnan(n0pc[2]))
          {
            problem.AddResidualBlock(
                  new ceres::AutoDiffCostFunction< PointToPlaneErrorGlobalPoseDeltaPoseCamViewDataOptiPt1, 3, 6, 3 >(
                    new PointToPlaneErrorGlobalPoseDeltaPoseCamViewDataOptiPt1(pt3pc.cast<double>(),n0pc.cast<double>(),param.depth_error_scale)),
                  (param.use_robust_loss?new ceres::CauchyLoss(param.loss_scale):NULL), &delta_pose[0], &pt3RGB[0]);
          }
        }
//        problem.SetParameterBlockConstant(&pt3RGB[0]);
      }
//      problem.SetParameterBlockConstant(cam0pc);
    }
  }

  if (const_all_intrinsics)
  {
    problem.SetParameterBlockConstant(&lm_intrinsics[0]);
  }
  else
  {
    if (const_intrinsics.size()>0)
    {
      ceres::SubsetParameterization *subset_parameterization =
            new ceres::SubsetParameterization(9, const_intrinsics);
      problem.SetParameterization(&lm_intrinsics[0], subset_parameterization);
    }
  }

    // Configure the solver.
  ceres::Solver::Options options;
  options.use_nonmonotonic_steps = true;
  options.preconditioner_type = ceres::SCHUR_JACOBI;
  options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.use_inner_iterations = true;
  options.max_num_iterations = 100;

//  options.minimizer_progress_to_stdout = false;
  options.minimizer_progress_to_stdout = true;

  // Solve!
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << "Final report:\n" << summary.FullReport();
}




/***************************************************************************************/


/**
 * @brief TSFOptimizeBundle::optimize
 * @param poses
 */
void TSFOptimizeBundle::optimize(std::vector<TSFFrame::Ptr> &map)
{
  if (map.size()<2)
  {
    cout<<"No data available! Can not bundle "<<map.size()<<" frames!"<<endl;
    return;
  }

  delta_pose.setZero();
  convertPosesToRt(map);
  convertPoints(map);

  cout<<"start opti. of "<<map.size()<<" frames"<<endl;
  cout<<"intrinsics: ";
  for (unsigned i=0; i<lm_intrinsics.size(); i++)
    cout<<lm_intrinsics[i]<<" ";
  cout<<endl;

  optimizePoses(map);

  if (param.optimize_delta_cloud_rgb_pose_global)
  {
    optimizeCloudPosesDeltaRGBPose(map);
  }
  else if (param.optimize_delta_cloud_rgb_pose)
  {
    optimizeCloudPosesRGBPoses(map);
  }

  cout<<"intrinsics opti.: ";
  for (unsigned i=0; i<lm_intrinsics.size(); i++)
    cout<<lm_intrinsics[i]<<" ";
  cout<<endl;
  cout<<"new delta pose: "<<delta_pose.transpose()<<endl;

  convertPosesFromRt(map);
  convertPosesFromRtRGB(map);

//  Eigen::Matrix3d Rrgb, Rpc;
//  Eigen::Matrix4d pose_rgb(Eigen::Matrix4d::Identity()), pose_pc(Eigen::Matrix4d::Identity()), inv_pose;
//  for (unsigned i=0; i<poses_Rt.size(); i++)
//  {
//    ceres::AngleAxisToRotationMatrix(&poses_Rt[i](0), &Rpc(0,0));
//    pose_pc.topLeftCorner<3, 3>() = Rpc;
//    pose_pc.block<3,1>(0, 3) = poses_Rt[i].tail<3>();
//    ceres::AngleAxisToRotationMatrix(&poses_Rt_RGB[i](0), &Rrgb(0,0));
//    pose_rgb.topLeftCorner<3, 3>() = Rrgb;
//    pose_rgb.block<3,1>(0, 3) = poses_Rt_RGB[i].tail<3>();

//    v4r::invPose(pose_pc,inv_pose);
//    cout<<i<<":"<<endl;
//    cout<<(pose_rgb*inv_pose)<<endl;
//  }
}

/**
 * @brief TSFOptimizeBundle::getCameraParameter
 * @param _intrinsic
 * @param _dist_coeffs
 */
void TSFOptimizeBundle::getCameraParameter(cv::Mat &_intrinsic, cv::Mat &_dist_coeffs)
{
  _dist_coeffs = cv::Mat();
  _intrinsic = cv::Mat();
  if (lm_intrinsics.size()>=4)
  {
    _intrinsic = cv::Mat_<double>(3,3);
    _intrinsic.at<double>(0,0) = lm_intrinsics[0];
    _intrinsic.at<double>(1,1) = lm_intrinsics[1];
    _intrinsic.at<double>(0,2) = lm_intrinsics[2];
    _intrinsic.at<double>(1,2) = lm_intrinsics[3];
  }
  if (lm_intrinsics.size()==9)
  {
    _dist_coeffs = cv::Mat_<double>(1,5);
    _dist_coeffs.at<double>(0,0) = lm_intrinsics[4];
    _dist_coeffs.at<double>(0,1) = lm_intrinsics[5];
    _dist_coeffs.at<double>(0,5) = lm_intrinsics[6];
    _dist_coeffs.at<double>(0,2) = lm_intrinsics[7];
    _dist_coeffs.at<double>(0,3) = lm_intrinsics[8];
  }
}

/**
 * setTargetCameraParameter
 */
void TSFOptimizeBundle::setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs)
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
 * @brief TSFOptimizeBundle::setParameter
 * @param p
 */
void TSFOptimizeBundle::setParameter(const Parameter &p)
{
  param = p;

  const_intrinsics.clear();
  const_all_intrinsics = true;
  if (lm_intrinsics.size()==0)
    return;
  if (!param.optimize_focal_length)
  {
    const_intrinsics.push_back(0);
    const_intrinsics.push_back(1);
  }
  if (!param.optimize_principal_point)
  {
    const_intrinsics.push_back(2);
    const_intrinsics.push_back(3);
  }
  if (lm_intrinsics.size()==9 && !param.optimize_radial_k1)
  {
    const_intrinsics.push_back(4);
  }
  if (lm_intrinsics.size()==9 && !param.optimize_radial_k2)
  {
    const_intrinsics.push_back(5);
  }
  if (lm_intrinsics.size()==9 && !param.optimize_radial_k3)
  {
    const_intrinsics.push_back(6);
  }
  if (lm_intrinsics.size()==9 && !param.optimize_tangential_p1)
  {
    const_intrinsics.push_back(7);
  }
  if (lm_intrinsics.size()==9 && !param.optimize_tangential_p2)
  {
    const_intrinsics.push_back(8);
  }
  const_all_intrinsics = ((lm_intrinsics.size()==4&&const_intrinsics.size()==4)||(lm_intrinsics.size()==9&&const_intrinsics.size()==9)?true:false);
}

}












