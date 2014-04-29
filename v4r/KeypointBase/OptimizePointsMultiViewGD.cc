/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#include "OptimizePointsMultiViewGD.hh"
#include <opencv2/highgui/highgui.hpp>
#include "v4r/KeypointTools/invPose.hpp"
#include "v4r/KeypointTools/ScopeTime.hpp"
#include "v4r/KeypointTools/rotation.h"


namespace kp 
{

using namespace std;


/********************** OptimizePointsMultiViewGD ************************
 * Constructor/Destructor
 */
OptimizePointsMultiViewGD::OptimizePointsMultiViewGD(const Parameter &p)
 : param(p), have_im_pts(false), have_depth(false), have_points(false), 
   have_normals(false), have_im_indices(false)
{
  cos_view_ray_normal_offset = cos(param.view_ray_normal_offset*M_PI/180);
  mv_views.resize(1);
  mv_views[0].idx = 0;

  im_pts.reset(new std::vector<cv::Point2f>);
  depth.reset(new std::vector<double>);
  points.reset(new std::vector<Eigen::Vector3f>);
  normals.reset(new std::vector<Eigen::Vector3f>);
  converged.reset(new std::vector<bool>);
  im_indices.reset(new std::vector< std::vector<int> >);
  view_rays.reset(new std::vector<Eigen::Vector3f>);
}

OptimizePointsMultiViewGD::~OptimizePointsMultiViewGD()
{
}




/************************** PRIVATE ************************/


/**
 * optimizeSequentialDim
 */
bool OptimizePointsMultiViewGD::optimizeSequentialDim(MultiViewPatchError &mpe, Eigen::Vector3d &rot, double &ddepth)
{
  int z;
  double delta_a = param.init_delta_rot;
  double delta_s = param.init_delta_trans; 
  double tmp_ddepth, sv_ddepth, sv_err, err, sv_err1, start_err;
  Eigen::Vector3d tmp_rot, sv_rot;

  sv_ddepth = ddepth = 0.;
  sv_rot = rot = Eigen::Vector3d::Zero();

  mpe(rot, ddepth, start_err);
  sv_err1 = start_err;

  for (z=0; z<param.max_iter; z++)
  {
    sv_err = sv_err1;

    for (int k=-1; k<=1; k+=2)
    {
      // test rotations
      for (int j=0; j<3; j++)
      {
        tmp_ddepth = ddepth;
        tmp_rot = rot;
        tmp_rot[j] += k*delta_a;

        mpe(tmp_rot, tmp_ddepth, err);

        if (err < sv_err)
        {
          sv_ddepth = tmp_ddepth;
          sv_rot = tmp_rot;
          sv_err = err;
        }
      }    
      // test depth
      tmp_ddepth = ddepth + k*delta_s;
      tmp_rot = rot;

      mpe(tmp_rot, tmp_ddepth, err);

      if (err < sv_err)
      {
        sv_ddepth = tmp_ddepth;
        sv_rot = tmp_rot;
        sv_err = err;
      }
    }

    if (sv_err < sv_err1)
    {
      rot = sv_rot;
      ddepth = sv_ddepth;
      sv_err1 = sv_err;
    }
    else
    {
      delta_a *= 0.5;
      delta_s *= 0.5;

      if (delta_a < param.convergence_rot && delta_s < param.convergence_trans)
      {
        break;
      }
    }
  }

  if ( sv_err1 < start_err && z < param.max_iter)
  {
    return true;
  }
  return false;
}

/**
 * InitData
 */
void OptimizePointsMultiViewGD::InitData(std::vector<cv::Point2f> &r_im_pts, std::vector<double> &r_depth, std::vector<Eigen::Vector3f> &r_points, std::vector<Eigen::Vector3f> &r_normals, std::vector<Eigen::Vector3f> &r_view_rays, std::vector< std::vector<int> > &r_im_indices)
{
  if (!(have_im_pts && have_depth) && !have_points)
    throw std::runtime_error("[OptimizePointsMultiViewGD::InitData] Invalid data! You need ether 3D points or image points and the depth!"); 

  r_view_rays.clear();

  if (have_im_pts && have_depth)
  {
    if (r_im_pts.size() != r_depth.size())
      throw std::runtime_error("[OptimizePointsMultiViewGD::InitData] Number of depth points != Number of image points!");

    // init 3d points
    if (!have_points || r_points.size() != r_depth.size())
    {
      r_view_rays.resize(r_im_pts.size());
      r_points.resize(r_im_pts.size());
      for (unsigned i=0; i<r_im_pts.size(); i++)
      {
        cv::Point2f &pt = r_im_pts[i];
        r_view_rays[i] = -Eigen::Vector3f((pt.x-C(0,2))/C(0,0),(pt.y-C(1,2))/C(1,1),1.).normalized();
        r_points[i] = -r_view_rays[i]*r_depth[i];
      }
    }
  }

  if (have_points)
  {
    // init image points
    if (!have_im_pts || r_im_pts.size()!=r_points.size())
    {
      r_im_pts.resize(r_points.size());
      for (unsigned i=0; i<r_points.size(); i++)
      {
        Eigen::Vector3f &pt3 = r_points[i];
        r_im_pts[i] = cv::Point2f(pt3[0]/pt3[2]*C(0,0)+C(0,2), pt3[1]/pt3[2]*C(1,1)+C(1,2));
      }
    }
    // init depth
    if (!have_depth || r_depth.size()!=r_points.size())
    {
      r_depth.resize(r_points.size());
      for (unsigned i=0; i<r_points.size(); i++)
      {
        r_depth[i] = r_points[i].norm();
      }
    }
  }

  // init view rays
  if (r_view_rays.size()!=r_points.size())
  {
    r_view_rays.resize(r_points.size());
    for (unsigned i=0; i<r_points.size(); i++)
    {
      r_view_rays[i] = -r_points[i].normalized();
    }
  }

  // init normals
  if (!have_normals || r_normals.size()!=r_view_rays.size())
  {
    r_normals.resize(r_view_rays.size());
    for (unsigned i=0; i<r_view_rays.size(); i++)
    {
      r_normals[i] = r_view_rays[i];
    }
  }

  // clar visibility
  if (!have_im_indices || r_im_indices.size()!=r_points.size())
  {
    r_im_indices.clear();
    r_im_indices.resize(r_points.size());
  }
}


/************************** PUBLIC *************************/

/**
 * clearFrames
 */
void OptimizePointsMultiViewGD::clearFrames()
{
  mv_views.resize(1);
}

/**
 * addFrame
 */
int OptimizePointsMultiViewGD::addFrame(const cv::Mat_<unsigned char> &image, const Eigen::Matrix4f &delta_pose)
{
  mv_views.push_back(MVView());
  
  mv_views.back().idx = mv_views.size()-1;
  mv_views.back().image = image;
  getR(delta_pose, mv_views.back().R);
  getT(delta_pose, mv_views.back().t);

  return mv_views.size()-2;
}

/**
 * setImagePoints
 */
void OptimizePointsMultiViewGD::setImagePoints(SmartPtr< std::vector<cv::Point2f> > &_im_pts) 
{
  im_pts=_im_pts;
  have_im_pts=true;
}

/**
 * setDepth
 */
void OptimizePointsMultiViewGD::setDepth(SmartPtr< std::vector<double> > &_depth) 
{
  depth=_depth;
  have_depth=true;
}

/**
 * setPoints
 */
void OptimizePointsMultiViewGD::setPoints(SmartPtr< std::vector<Eigen::Vector3f> > &_points) 
{ 
  points=_points;
  have_points=true;
}

/**
 * setNormals
 */
void OptimizePointsMultiViewGD::setNormals(SmartPtr< std::vector<Eigen::Vector3f> > &_normals) 
{
  normals=_normals;
  have_normals=true;
}

/**
 * setImageIndices
 */
void OptimizePointsMultiViewGD::setImageIndices(SmartPtr< std::vector< std::vector<int> > > &_idx) 
{
  im_indices=_idx;
  have_im_indices=true;
}

/**
 * optimize
 */
void OptimizePointsMultiViewGD::optimize(const cv::Mat_<unsigned char> &image)
{
  if (mv_views.size()<2)
    return;

  std::vector<cv::Point2f> &r_im_pts = *im_pts;
  std::vector<double> &r_depth = *depth;
  std::vector<Eigen::Vector3f> &r_points = *points;
  std::vector<Eigen::Vector3f> &r_normals = *normals;
  std::vector<bool> &r_converged = *converged;
  std::vector< std::vector<int> > &r_im_indices = *im_indices;
  std::vector<Eigen::Vector3f> &r_view_rays = *view_rays;

  InitData(r_im_pts, r_depth, r_points, r_normals, r_view_rays, r_im_indices);

  r_converged.clear();
  r_converged.resize(r_depth.size(), false);

  mv_views[0].image = image;
  mv_views[0].R.setIdentity();
  mv_views[0].t.setZero();

  Eigen::Vector3d vrx, n_new;

  MVPoint mv_point;
  Eigen::Vector3d rot;
  double ddepth;
  std::vector<unsigned char> patch0(param.patch_size.width*param.patch_size.height);
  std::vector<unsigned char> patch1(param.patch_size.width*param.patch_size.height);
  std::vector<MVView*> ptr_mv_views;
  std::vector<double> residuals;

  // optimize points
  for (unsigned j=0; j<r_im_pts.size(); j++)
  {
    const cv::Point2f &pt = r_im_pts[j];

    if (pt.x<param.patch_size.width || pt.y<param.patch_size.height || 
        pt.x>=image.cols-param.patch_size.width || 
        pt.y>=image.rows-param.patch_size.height)
      continue;

    mv_point = MVPoint(pt, r_points[j], r_normals[j], r_view_rays[j]);

    // select views (test visibility, test error)
    ptr_mv_views.clear();
    ptr_mv_views.push_back(&mv_views[0]);
    if (have_im_indices)
    {
      for (unsigned i=0; i<r_im_indices[j].size(); i++)
      {
        MVView *ptr_mvx = &mv_views[r_im_indices[j][i]+1];
        vrx = ptr_mvx->R * mv_point.vr;
        if (mv_point.n.dot(vrx) > cos_view_ray_normal_offset)
          ptr_mv_views.push_back(ptr_mvx);
      }
    }
    else
    {
      for (unsigned i=1; i<mv_views.size(); i++)
      {
        MVView *ptr_mvx = &mv_views[i];
        vrx = ptr_mvx->R * mv_point.vr;
        if (mv_point.n.dot(vrx) > cos_view_ray_normal_offset)
          ptr_mv_views.push_back(ptr_mvx);
      }
    }

    MultiViewPatchError mpe_select = MultiViewPatchError( ptr_mv_views, mv_point, C, invC,
            param.patch_size.height, param.patch_size.width, &patch0[0], &patch1[0] );
    
    mpe_select(rot, ddepth, residuals);

    unsigned z=0;
    for (unsigned i=0; i<ptr_mv_views.size(); i++) {
      if (residuals[i] < param.max_residual) {
        ptr_mv_views[z] = ptr_mv_views[i];
        z++;
      }
    }

    if (z<2) continue;
    ptr_mv_views.resize(z);

    // create cost function
    MultiViewPatchError mpe = MultiViewPatchError( ptr_mv_views, mv_point, C, invC, 
            param.patch_size.height, param.patch_size.width, &patch0[0], &patch1[0] );

    // simple optimizer
    if(optimizeSequentialDim(mpe, rot, ddepth))
    {
      kp::AngleAxisRotatePoint(&rot[0], &mv_point.n[0], &n_new[0]);
      r_depth[j] -= ddepth;
      r_normals[j] = Eigen::Vector3f(n_new[0], n_new[1], n_new[2]);
      r_points[j] = -r_view_rays[j]*r_depth[j];
      
      r_converged[j] = true;

      mpe(rot, ddepth, residuals);

      r_im_indices[j].clear();
      for (unsigned i=0; i<residuals.size(); i++) {
        if (residuals[i] > param.max_residual_optimized) {
          r_converged[j] = false;
          break;
        }
        else r_im_indices[j].push_back(ptr_mv_views[i]->idx-1);
      }
    }
  }

  have_im_pts= have_depth= have_points= have_normals= have_im_indices = false;
}

/**
 * setCameraParameter
 */
void OptimizePointsMultiViewGD::setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs)
{
  cv::Mat_<double> intrinsic;

  if (!_dist_coeffs.empty())
    throw std::runtime_error("[OptimizePointsMultiViewGD::setCameraParameter] Image distortions are not supported!");
  if (_intrinsic.rows!=3 || _intrinsic.cols!=3)
    throw std::runtime_error("[OptimizePointsMultiViewGD::setCameraParameter] Invalid intrinsic! Need a 3x3 matrix");

  if (_intrinsic.type() != CV_64F)
    _intrinsic.convertTo(intrinsic, CV_64F);
  else intrinsic = _intrinsic;

  for (int v=0; v<3; v++)
    for (int u=0; u<3; u++)
      C(v,u) = intrinsic(v,u);

  invC = C.inverse();
}



} //-- THE END --






