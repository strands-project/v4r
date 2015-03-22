#include "fitting_surface_depth_pca.h"
#include <pcl/common/common.h>

using namespace pcl;
using namespace on_nurbs;

template<typename PointT> void
FittingSurfaceDepthPCA<PointT>::setInputCloud (const PointCloudConstPtr &cloud)
{
  pca.setInputCloud (cloud);
  pca_done_ = false;
  nurbs_done_ = false;
  proj_done_ = false;
}

template<typename PointT> void
FittingSurfaceDepthPCA<PointT>::setIndices (const IndicesPtr &indices)
{
  pca.setIndices(indices);
  pca_done_ = false;
  nurbs_done_ = false;
  proj_done_ = false;
}

template<typename PointT> void
FittingSurfaceDepthPCA<PointT>::setIndices (const IndicesConstPtr &indices)
{
  pca.setIndices(indices);
  pca_done_ = false;
  nurbs_done_ = false;
  proj_done_ = false;
}

template<typename PointT> void
FittingSurfaceDepthPCA<PointT>::setIndices (const PointIndicesConstPtr &indices)
{
  pca.setIndices(indices);
  pca_done_ = false;
  nurbs_done_ = false;
  proj_done_ = false;
}

template<typename PointT> void
FittingSurfaceDepthPCA<PointT>::setIndices (size_t row_start, size_t col_start, size_t nb_rows, size_t nb_cols)
{
  pca.setIndices(row_start, col_start, nb_rows, nb_cols);
  pca_done_ = false;
  nurbs_done_ = false;
  proj_done_ = false;
}

template<typename PointT> void
FittingSurfaceDepthPCA<PointT>::setParameter(int order, int cps_x, int cps_y)
{
  order_ = order;
  cps_x_ = cps_x;
  cps_y_ = cps_y;
  nurbs_done_ = false;
  proj_done_ = false;
}

template<typename PointT> void
FittingSurfaceDepthPCA<PointT>::flipEigenSpace()
{
  if(!pca_done_)
    computePCA();

  eigenvectors_.col(1) *= -1.0;
  eigenvectors_.col(2) *= -1.0;
}

template<typename PointT> void
FittingSurfaceDepthPCA<PointT>::project(const PointT& input, PointT& projection)
{
  if(!pca_done_)
    computePCA();

  Eigen::Vector3f demean_input = input.getVector3fMap () - mean_.head<3>();
  projection.getVector3fMap () = eigenvectors_.transpose() * demean_input;
}

template<typename PointT> void
FittingSurfaceDepthPCA<PointT>::reconstruct(const PointT& projection, PointT& input)
{
  if(!pca_done_)
    computePCA();

  input.getVector3fMap() = eigenvectors_ * projection.getVector3fMap();
  input.getVector3fMap() += mean_.head<3>();
}

template<typename PointT> Eigen::Vector4f
FittingSurfaceDepthPCA<PointT>::getMean()
{
  if(!pca_done_)
    computePCA();
  return mean_;
}

template<typename PointT> Eigen::Matrix3f
FittingSurfaceDepthPCA<PointT>::getEigenVectors()
{
  if(!pca_done_)
    computePCA();
  return eigenvectors_;
}

template<typename PointT> ON_NurbsSurface&
FittingSurfaceDepthPCA<PointT>::getSurface ()
{
  if(!pca_done_)
    computePCA();
  if(!proj_done_)
    computeProjection();
  if(!proj_done_)
    PCL_THROW_EXCEPTION (InitFailedException,
                         "[pcl::on_nurbs::FittingSurfaceDepthPCA::getSurface] FittingSurfaceDepthPCA computeProjection() failed");
  if (!nurbs_done_)
    computeNurbsSurface();
  if (!nurbs_done_)
    PCL_THROW_EXCEPTION (InitFailedException,
                         "[pcl::on_nurbs::FittingSurfaceDepthPCA::getSurface] FittingSurfaceDepthPCA computeNurbsSurface() failed");
  return m_nurbs;
}

template<typename PointT> ON_NurbsSurface
FittingSurfaceDepthPCA<PointT>::getSurface3D()
{
  if(!pca_done_)
    computePCA();
  if(!proj_done_)
    computeProjection();
  if(!proj_done_)
    PCL_THROW_EXCEPTION (InitFailedException,
                         "[pcl::on_nurbs::FittingSurfaceDepthPCA::getSurface3D] FittingSurfaceDepthPCA computeProjection() failed");
  if (!nurbs_done_)
    computeNurbsSurface();
  if (!nurbs_done_)
    PCL_THROW_EXCEPTION (InitFailedException,
                         "[pcl::on_nurbs::FittingSurfaceDepthPCA::getSurface3D] FittingSurfaceDepthPCA ON_NurbsSurface failed");

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

      reconstruct(p1,p2);

      cp.x = p2.x;
      cp.y = p2.y;
      cp.z = p2.z;

      nurbs_3d.SetCV(i,j,cp);
    }
  }

  return nurbs_3d;
}

template<typename PointT> Eigen::VectorXd
FittingSurfaceDepthPCA<PointT>::getError ()
{
  if(!pca_done_)
    computePCA();
  if(!proj_done_)
    computeProjection();
  if(!proj_done_)
    PCL_THROW_EXCEPTION (InitFailedException,
                         "[pcl::on_nurbs::FittingSurfaceDepthPCA::getError] FittingSurfaceDepthPCA computeProjection() failed");
  if (!nurbs_done_)
    computeNurbsSurface();
  if (!nurbs_done_)
    PCL_THROW_EXCEPTION (InitFailedException,
                         "[pcl::on_nurbs::FittingSurfaceDepthPCA::getError] FittingSurfaceDepthPCA computeNurbsSurface() failed");

  return FittingSurfaceDepth::GetError(points_.col(2));
}

template<typename PointT> const FittingSurfaceDepth::ROI&
FittingSurfaceDepthPCA<PointT>::getROI()
{
  if(!pca_done_)
    computePCA();
  if (!proj_done_)
    computeProjection();
  if (!proj_done_)
    PCL_THROW_EXCEPTION (InitFailedException,
                         "[pcl::on_nurbs::FittingSurfaceDepthPCA::getROI] FittingSurfaceDepthPCA computeProjection() failed");
  return roi_pc_;
}

template<typename PointT> const typename FittingSurfaceDepthPCA<PointT>::PointCloud&
FittingSurfaceDepthPCA<PointT>::getProjectedCloud()
{
  if(!pca_done_)
    computePCA();
  if (!proj_done_)
    computeProjection();
  if (!proj_done_)
    PCL_THROW_EXCEPTION (InitFailedException,
                         "[pcl::on_nurbs::FittingSurfaceDepthPCA::getProjectedCloud] FittingSurfaceDepthPCA computeProjection() failed");
  return cloud_pc_;
}

template<typename PointT> bool
FittingSurfaceDepthPCA<PointT>::computePCA ()
{
  mean_ = pca.getMean();
  eigenvectors_ = pca.getEigenVectors();
  pca_done_ = true;
  return true;
}

template<typename PointT> bool
FittingSurfaceDepthPCA<PointT>::computeProjection ()
{
  // project cloud into eigen space and compute ROI
  points_.resize(pca.getIndices()->size(), 3);
  cloud_pc_.resize(pca.getIndices()->size());

  Eigen::Vector2f proj_min(DBL_MAX,DBL_MAX), proj_max(DBL_MIN,DBL_MIN), proj_del;
  for(size_t i=0; i<pca.getIndices()->size(); i++)
  {
    const PointT& p1 = pca[i];
    PointT& p2 = cloud_pc_.at(i);
    p2 = p1; // copy attributes other than XYZ

    project(p1, p2);

    points_(i,0) = p2.x;
    points_(i,1) = p2.y;
    points_(i,2) = p2.z;

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
  roi_pc_ = ROI(proj_min(0),proj_min(1),proj_del(0),proj_del(1));

  proj_done_ = true;
  return true;
}

template<typename PointT> bool
FittingSurfaceDepthPCA<PointT>::computeNurbsSurface ()
{
  // init surface, solver
  initSurface(order_, cps_x_, cps_y_, roi_pc_);
  initSolver(points_);

  // solve
  solve(points_.col(2));

  nurbs_done_ = true;
  return true;
}

template class pcl::on_nurbs::FittingSurfaceDepthPCA<pcl::PointXYZ>;
template class pcl::on_nurbs::FittingSurfaceDepthPCA<pcl::PointXYZRGB>;
template class pcl::on_nurbs::FittingSurfaceDepthPCA<pcl::PointXYZRGBL>;
