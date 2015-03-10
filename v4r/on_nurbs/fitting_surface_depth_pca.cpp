#include "fitting_surface_depth_pca.h"
#include <pcl/common/common.h>

using namespace pcl;
using namespace on_nurbs;

template<typename PointT> ON_NurbsSurface
FittingSurfaceDepthPCA<PointT>::getSurface3D()
{
  if (!nurbs_done_)
    initCompute();
  if (!nurbs_done_)
    PCL_THROW_EXCEPTION (InitFailedException,
                         "[pcl::on_nurbs::FittingSurfaceDepthPCA::reconstruct] FittingSurfaceDepthPCA ON_NurbsSurface failed");

  ON_NurbsSurface nurbs_3d;

  // increase dimension
  IncreaseDimension(m_nurbs, nurbs_3d, 3);

  double gx[nurbs_3d.CVCount(0)];
  double gy[nurbs_3d.CVCount(1)];
  m_nurbs.GetGrevilleAbcissae(0, gx);
  m_nurbs.GetGrevilleAbcissae(1, gy);

  // reconstruct (i.e. reproject from eigen space)
  PointT p1, p2;
  for(int j=0; j<nurbs_3d.CVCount(1); j++)
  {
    for(int i=0; i<nurbs_3d.CVCount(0); i++)
    {
      ON_3dPoint cp;
      m_nurbs.GetCV(i,j,cp);

      p1.x = gx[i];
      p1.y = gy[j];
      p1.z = cp.x;

      PCA<PointT>::reconstruct(p1,p2);

      cp.x = p2.x;
      cp.y = p2.y;
      cp.z = p2.z;

      nurbs_3d.SetCV(i,j,cp);
    }
  }

  return nurbs_3d;
}


template<typename PointT> bool
FittingSurfaceDepthPCA<PointT>::initCompute ()
{
  if(!Base::initCompute ())
  {
    PCL_THROW_EXCEPTION (InitFailedException, "[pcl::on_nurbs::FittingSurfaceDepthPCA::initCompute] failed (pcl::Base)");
    return (false);
  }

  // project cloud into eigen space and compute ROI
  Eigen::MatrixXd points(indices_->size(), 3);
  PointT p2;
  Eigen::Vector2f proj_min(DBL_MAX,DBL_MAX), proj_max(DBL_MIN,DBL_MIN), proj_del;
  for(size_t i=0; i<indices_->size(); i++)
  {
    const PointT& p1 = input_->at(indices_->at(i));
    PCA<PointT>::project(p1, p2);
    points(i,0) = p2.x;
    points(i,1) = p2.y;
    points(i,2) = p2.z;

    if(p2.x<proj_min(0))
      proj_min(0) = p2.x;
    if(p2.x>proj_max(0))
      proj_max(0) = p2.x;
    if(p2.y<proj_min(1))
      proj_min(1) = p2.y;
    if(p2.y>proj_max(1))
      proj_max(1) = p2.y;
  }
  proj_del = proj_max - proj_min;

  // init surface, solver
  initSurface(order_, cps_x_, cps_y_, ROI(proj_min(0),proj_min(1),proj_del(0),proj_del(1)));
  initSolver(points);

  // solve
  solve(points.col(2));

  nurbs_done_ = true;
  return true;
}

template class pcl::on_nurbs::FittingSurfaceDepthPCA<pcl::PointXYZRGBL>;
