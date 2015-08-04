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


#include <v4r/reconstruction/ProjBundleAdjuster.h>
#include <v4r/keypoints/impl/invPose.hpp>
#include <v4r/keypoints/impl/triple.hpp>
#include <v4r/reconstruction/impl/ReprojectionError.hpp>

namespace v4r 
{

using namespace std;


/************************************************************************************
 * Constructor/Destructor
 */
ProjBundleAdjuster::ProjBundleAdjuster(const Parameter &p)
 : param(p)
{ 
  sqr_depth_inl_dist = param.depth_inl_dist*param.depth_inl_dist;
}

ProjBundleAdjuster::~ProjBundleAdjuster()
{
}


/**
 * getDataToBundle
 */
void ProjBundleAdjuster::getCameras(const Object &data, std::vector<Camera> &cameras)
{
  cameras.clear();
  Eigen::Matrix3d R;
  Eigen::Vector3d t;

  cameras.resize(data.cameras.size());

  for (unsigned i=0; i<cameras.size(); i++)
  {
      getR(data.cameras[i], R);
      getT(data.cameras[i], t);
      ceres::RotationMatrixToAngleAxis(&R(0,0), &cameras[i].pose_Rt(0));                
      cameras[i].pose_Rt.tail<3>() = t;
      cameras[i].idx = i;
  } 
}

/**
 * setBundledData
 */
void ProjBundleAdjuster::setCameras(const std::vector<Camera> &cameras, Object &data)
{
  Eigen::Matrix3d R;
  Eigen::Vector3d t;

  for (unsigned i=0; i<cameras.size(); i++)
  {
    const Camera &cam = cameras[i];

    ceres::AngleAxisToRotationMatrix(&cam.pose_Rt(0), &R(0,0));
    t = cam.pose_Rt.tail<3>();

    setPose(R,t, data.cameras[i]);
  }
}


/**
 * bundle
 */
void ProjBundleAdjuster::bundle(Object &data, std::vector<Camera> &cameras)
{
  if (cameras.size()<2)
    return;

  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);

  Eigen::Matrix3f poseR;
  Eigen::Vector3f poset;

  double *intrinsics = 0;
  int num_cam_param = 0;
  std::set<int> idx_cams;
  std::set<int>::iterator it;
  std::vector<int> constant_intrinsics;

  if (data.camera_parameter.size()==1)
  {
    intrinsics = &data.camera_parameter[0][0];
    num_cam_param = data.camera_parameter[0].size();
  }

  if (param.optimize_intrinsic && !param.optimize_dist_coeffs)
  {
    constant_intrinsics.push_back(4);
    constant_intrinsics.push_back(5);
    constant_intrinsics.push_back(6);
    constant_intrinsics.push_back(7);
    constant_intrinsics.push_back(8);
  }

  //std::vector<int> projections_per_point(data.points.size(), 0);

  for (unsigned v=0; v<data.views.size(); v++)
  {
    ObjectView &view = *data.views[v];

    for (unsigned i=0; i<view.projs.size(); i++)
    {
      Eigen::Vector3d &eig_pt3 = view.getPt(i).pt;
      double *pt3 = &eig_pt3[0];
      const std::vector< triple<int, cv::Point2f, Eigen::Vector3f> > &projs = view.projs[i];

      //projections_per_point[view.points[i]]+=projs.size();
      if (projs.size() < 2) continue;

      for (unsigned j=0; j<projs.size(); j++)
      {
        const triple<int, cv::Point2f, Eigen::Vector3f> &p = projs[j];
        double *pose_Rt = &cameras[p.first].pose_Rt[0];
        poseR = data.cameras[cameras[p.first].idx].topLeftCorner<3,3>();
        poset = data.cameras[cameras[p.first].idx].block<3,1>(0,3);

        if (data.camera_parameter.size() > 1)
        {
          intrinsics = &data.camera_parameter[p.first][0];
          num_cam_param = data.camera_parameter[p.first].size();
          idx_cams.insert(p.first);
        }

        if (num_cam_param==4) {
          if ( param.use_depth_prior && !isnan(p.third) && p.third[2]<param.depth_cut_off && 
               (poseR*eig_pt3.cast<float>()+poset - p.third).squaredNorm() < sqr_depth_inl_dist ) {
            problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction< NoDistortionReprojectionAndDepthError, 3, 4, 6, 3 >(
              new NoDistortionReprojectionAndDepthError(p.second.x,p.second.y,p.third[2],param.depth_error_weight)), 0/*new ceres::CauchyLoss(0.1)*/,intrinsics,pose_Rt,pt3);
          } else {
            problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction< NoDistortionReprojectionError, 2, 4, 6, 3 >(
              new NoDistortionReprojectionError(p.second.x, p.second.y)), 0/*new ceres::CauchyLoss(0.1)*/,intrinsics, pose_Rt, pt3);
          }
        } else if (num_cam_param==9) {
          if ( param.use_depth_prior && !isnan(p.third) && p.third[2]<param.depth_cut_off && 
               (poseR*eig_pt3.cast<float>()+poset - p.third).squaredNorm() < sqr_depth_inl_dist )           {
            problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction< RadialDistortionReprojectionAndDepthError, 3, 9, 6, 3 >(
              new RadialDistortionReprojectionAndDepthError(p.second.x, p.second.y, p.third[2],param.depth_error_weight)), 0/*new ceres::CauchyLoss(0.1)*/, intrinsics, pose_Rt, pt3);
          } else {
            problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction< RadialDistortionReprojectionError, 2, 9, 6, 3 >(
              new RadialDistortionReprojectionError(p.second.x, p.second.y)), 0/*new ceres::CauchyLoss(0.1)*/, intrinsics, pose_Rt, pt3);
          }
        }
      }
    }
  }

  //std::cout << "Number of 3D points:" << projections_per_point.size() << std::endl;
  //for(size_t i=0; i < projections_per_point.size(); i++)
  //{
  //    std::cout << projections_per_point[i] << " ";
  //}
  //std::cout << std::endl;

  if (data.camera_parameter.size()==1) 
  {
    if (param.optimize_intrinsic) {
      if (num_cam_param==9 && !param.optimize_dist_coeffs) {
        ceres::SubsetParameterization *subset_parameterization = 
              new ceres::SubsetParameterization(9, constant_intrinsics);
        problem.SetParameterization(intrinsics, subset_parameterization);
      } 
    } else problem.SetParameterBlockConstant(intrinsics);
  }
  else
  {
    for (it=idx_cams.begin(); it!=idx_cams.end(); it++)
    {
      if (param.optimize_intrinsic) { 
        if (num_cam_param==9 && !param.optimize_dist_coeffs) {
          ceres::SubsetParameterization *subset_parameterization =
                new ceres::SubsetParameterization(9, constant_intrinsics);
          problem.SetParameterization(&data.camera_parameter[*it][0], subset_parameterization);
        }
      } else problem.SetParameterBlockConstant(&data.camera_parameter[*it][0]);
    }
  }
 
  // Configure the solver.
  ceres::Solver::Options options;
  options.use_nonmonotonic_steps = true;
  options.preconditioner_type = ceres::SCHUR_JACOBI;
  options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.use_inner_iterations = true;
  options.max_num_iterations = 100;

  if (!dbg.empty()) 
    options.minimizer_progress_to_stdout = true;
  else options.minimizer_progress_to_stdout = false;

  // Solve!
  ceres::Solver::Summary summary;

  ceres::Solve(options, &problem, &summary);

  if (!dbg.empty()) {
    std::cout << "Final report:\n" << summary.FullReport();
  }
}

/**
 * TODO: that was a test for Kinect calibration
 */
/*static const bool param.use_depth_prior = true;  // HACK!!!!!!!!!!!!!!
static const double depth_error_weight = 100.;//30.;
void ProjBundleAdjuster::bundle(Object &data, std::vector<Camera> &cameras, std::vector< std::vector<Eigen::Vector3d> > &points)
{
  if (cameras.size()<2)
    return;

  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);

  Eigen::Matrix3f poseR;
  Eigen::Vector3f poset;

Eigen::Matrix3d R;
Eigen::Vector3d t;
Eigen::Matrix4f delta_pose(Eigen::Matrix4f::Identity());
Eigen::Matrix<double, 6, 1> delta_Rt;
getR(delta_pose, R);
getT(delta_pose, t);
ceres::RotationMatrixToAngleAxis(&R(0,0), &delta_Rt(0));
delta_Rt.tail<3>() = t;
cout<<"delta_pose="<<endl;
cout<<delta_pose<<endl;


  double *intrinsics = 0;
  int num_cam_param = 0;
  std::set<int> idx_cams;
  std::set<int>::iterator it;
  std::vector<int> constant_intrinsics;

  if (data.camera_parameter.size()==1)
  {
    intrinsics = &data.camera_parameter[0][0];
    num_cam_param = data.camera_parameter[0].size();
  }

  if (param.optimize_intrinsic && !param.optimize_dist_coeffs)
  {
    constant_intrinsics.push_back(4);
    constant_intrinsics.push_back(5);
    constant_intrinsics.push_back(6);
    constant_intrinsics.push_back(7);
    constant_intrinsics.push_back(8);
  }

  for (unsigned v=0; v<data.views.size(); v++)
  {
    ObjectView &view = *data.views[v];
    std::vector<Eigen::Vector3d> &pts = points[v];

    for (unsigned i=0; i<view.projs.size(); i++)
    {
      Eigen::Vector3d &eig_pt3 = pts[i];
      double *pt3 = &eig_pt3[0];
      const std::vector< triple<int, cv::Point2f, Eigen::Vector3f> > &projs = view.projs[i];
      //const Eigen::Vector3f &normal = view.normals[i];

      if (projs.size() < 2) continue;

      for (unsigned j=0; j<projs.size(); j++)
      {
        const triple<int, cv::Point2f, Eigen::Vector3f> &p = projs[j];
        double *pose_Rt = &cameras[p.first].pose_Rt[0];
        poseR = data.cameras[cameras[p.first].idx].topLeftCorner<3,3>();
        poset = data.cameras[cameras[p.first].idx].block<3,1>(0,3);

        if (data.camera_parameter.size() > 1)
        {
          intrinsics = &data.camera_parameter[p.first][0];
          num_cam_param = data.camera_parameter[p.first].size();
          idx_cams.insert(p.first);
        }

        if (num_cam_param==4) {
          if ( param.use_depth_prior && !isnan(p.third) && 
               (poseR*eig_pt3.cast<float>()+poset - p.third).squaredNorm() < 0.01*0.01 ) {
            problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction< NoDistortionReprojectionAndDepthError, 3, 4, 6, 3 >(
              new NoDistortionReprojectionAndDepthError(p.second.x,p.second.y,p.third.norm(),depth_error_weight)), NULL,intrinsics,pose_Rt,pt3);
*/
            //problem.AddResidualBlock(
            //  new ceres::AutoDiffCostFunction< NoDistortionReprojectionAndPointPlaneError, 3, 4, 6, 6, 3 >(
            //  new NoDistortionReprojectionAndPointPlaneError(p.second.x,p.second.y,p.third[0],p.third[1],p.third[2],
            //          normal[0],normal[1],normal[2],depth_error_weight)), NULL /*new ceres::CauchyLoss(0.5)*/,intrinsics,pose_Rt,&delta_Rt[0],pt3);
/*          } else {
            problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction< NoDistortionReprojectionError, 2, 4, 6, 3 >(
              new NoDistortionReprojectionError(p.second.x, p.second.y)), NULL,intrinsics, pose_Rt, pt3);
          }
        } else if (num_cam_param==9) {
          problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction< RadialDistortionReprojectionError, 2, 9, 6, 3 >(
            new RadialDistortionReprojectionError(p.second.x, p.second.y)), NULL, intrinsics, pose_Rt, pt3);
        }
      }
    }
  }

//problem.SetParameterBlockConstant(&delta_Rt[0]);

  if (data.camera_parameter.size()==1) 
  {
    if (param.optimize_intrinsic) {
      if (num_cam_param==9 && !param.optimize_dist_coeffs) {
        ceres::SubsetParameterization *subset_parameterization = 
              new ceres::SubsetParameterization(9, constant_intrinsics);
        problem.SetParameterization(intrinsics, subset_parameterization);
      } 
    } else problem.SetParameterBlockConstant(intrinsics);
  }
  else
  {
    for (it=idx_cams.begin(); it!=idx_cams.end(); it++)
    {
      if (param.optimize_intrinsic) { 
        if (num_cam_param==9 && !param.optimize_dist_coeffs) {
          ceres::SubsetParameterization *subset_parameterization =
                new ceres::SubsetParameterization(9, constant_intrinsics);
          problem.SetParameterization(&data.camera_parameter[*it][0], subset_parameterization);
        }
      } else problem.SetParameterBlockConstant(&data.camera_parameter[*it][0]);
    }
  }
 
  // Configure the solver.
  ceres::Solver::Options options;
  options.use_nonmonotonic_steps = true;
  options.preconditioner_type = ceres::SCHUR_JACOBI;
  options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.use_inner_iterations = true;
  options.max_num_iterations = 100;

  if (!dbg.empty()) 
    options.minimizer_progress_to_stdout = true;
  else options.minimizer_progress_to_stdout = false;

  // Solve!
  ceres::Solver::Summary summary;

  ceres::Solve(options, &problem, &summary);

  if (!dbg.empty()) {
    std::cout << "Final report:\n" << summary.FullReport();
  }
ceres::AngleAxisToRotationMatrix(&delta_Rt(0), &R(0,0));
t = delta_Rt.tail<3>();
setPose(R,t, delta_pose);
cout<<"delta_pose="<<endl;
cout<<delta_pose<<endl;
}
*/



/***************************************************************************************/

/**
 * optimize
 */
void ProjBundleAdjuster::optimize(Object &data)
{
  if (!dbg.empty()) cout<<"-- [ProjBundleAdjuster::bundle] debug out --"<<endl;

  getCameras(data, cameras);

  if (cameras.size()<2) return;

  if (!dbg.empty() && param.optimize_intrinsic) {
    for (unsigned i=0; i<data.camera_parameter[0].size(); i++)
      cout<<data.camera_parameter[0][i]<<" ";
    cout<<endl;
  }

  bundle(data, cameras);

  setCameras(cameras, data); 

  if (!dbg.empty()) cout<<"[ProjBundleAdjuster::optimize] Number of cameras to bundle: "<<cameras.size()<<endl;
  
  if (!dbg.empty() && param.optimize_intrinsic) {
    for (unsigned i=0; i<data.camera_parameter[0].size(); i++)
      cout<<data.camera_parameter[0][i]<<" ";
    cout<<endl;
  }

}






}












