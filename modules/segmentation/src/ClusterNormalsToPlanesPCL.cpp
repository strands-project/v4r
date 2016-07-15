/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#include <v4r/common/normals.h>
#include <v4r/segmentation/ClusterNormalsToPlanesPCL.h>
#include <pcl/features/normal_3d.h>

namespace v4r
{

using namespace std;

template<typename PointT>
void
ClusterNormalsToPlanesPCL<PointT>::clusterNormals(const typename pcl::PointCloud<PointT>::Ptr &cloud, const pcl::PointCloud<pcl::Normal> &normals, size_t idx, typename ClusterNormalsToPlanesPCL<PointT>::Plane &plane)
{
  mask_[idx] = false;

  plane.init(cloud->points[idx].getVector3fMap(), normals.points[idx].getNormalVector3fMap(), idx);
  
  size_t queue_idx = 0;
  int width = cloud->width;
  int height = cloud->height;

  std::vector<int> n_ind(4);

  queue_.resize(1);
  queue_[0] = idx;

  
  // start clustering
  while ( queue_.size() > queue_idx)
  {
    // extract current index
    idx = queue_.at(queue_idx);
    queue_idx++;

    n_ind[0] = idx-1;
    n_ind[1] = idx+1;
    n_ind[2] = idx+width;
    n_ind[3] = idx-width;

    for(unsigned i=0; i<n_ind.size(); i++)
    {
      int u = n_ind[i] % width;
      int v = n_ind[i] / width;

      if ( (v < 0) || (u < 0) || (v >= height) || (u >= width) )
        continue;

      idx = n_ind[i];

      // not valid or not used point
      if (!(mask_[idx]))
        continue;

      const Eigen::Vector3f &n = normals.points[idx].getNormalVector3fMap();
      const Eigen::Vector3f &pt = cloud->points[idx].getVector3fMap();

      float cosa = plane.normal.dot(n);
      float dist = fabs(plane.normal.dot(pt - plane.pt));

      // we can add this point to the plane
      if ( (cosa > cos_rad_thr_angle) && (dist < param.inlDist) )
      {
        mask_[idx] = false;
        plane.add(pt, n, idx);
        queue_.push_back(idx);
        plane.normal.normalize();
      }
    }
  }
}


template<typename PointT>
void
ClusterNormalsToPlanesPCL<PointT>::clusterNormalsUnorganized(const typename pcl::PointCloud<PointT>::Ptr &cloud, const pcl::PointCloud<pcl::Normal> &normals, size_t idx, typename ClusterNormalsToPlanesPCL<PointT>::Plane &plane)
{
    mask_[idx] = false;

    plane.init(cloud->points[idx].getVector3fMap(), normals.points[idx].getNormalVector3fMap(), idx);

    size_t queue_idx = 0;
    int width = cloud->width;
    int height = cloud->height;

    queue_.resize(1);
    queue_[0] = idx;

    if( !flann_index) {
        size_t rows = cloud->points.size ();
        size_t cols = 3; // XYZ

        flann::Matrix<float> flann_data ( new float[rows * cols], rows, cols );

        for ( size_t i = 0; i < rows; i++)
        {
            flann_data.ptr () [i * cols + 0] = cloud->points[i].x;
            flann_data.ptr () [i * cols + 1] = cloud->points[i].y;
            flann_data.ptr () [i * cols + 2] = cloud->points[i].z;
        }
        flann_index.reset (new flann::Index<DistT> ( flann_data, flann::KDTreeIndexParams ( 4 ) ) );
        flann_index->buildIndex ();
    }

    // start clustering
    while ( queue_.size() > queue_idx )
    {
      // extract current index
      idx = queue_.at(queue_idx);
      queue_idx++;

      const PointT &seed_pt = cloud->points[idx];
      float *data = new float[3];
      data[0] = seed_pt.x;
      data[1] = seed_pt.y;
      data[2] = seed_pt.z;

      flann::Matrix<float> p = flann::Matrix<float> ( new float[3], 1, 3 );
      memcpy ( &p.ptr () [0], &data[0], p.cols * p.rows * sizeof ( float ) );

      flann::Matrix<int> indices = flann::Matrix<int> ( new int[param.K_], 1, param.K_ );
      flann::Matrix<float> distances = flann::Matrix<float> ( new float[param.K_], 1, param.K_);
      flann_index->knnSearch(p, indices, distances, param.K_, flann::SearchParams ( 128 ));
      delete[] p.ptr ();

      std::vector<int> n_ind (indices.cols);
      for(size_t k=0; k<indices.cols; k++)
          n_ind[k] = indices[0][k];

      for(unsigned i=0; i<n_ind.size(); i++)
      {
        int u = n_ind[i] % width;
        int v = n_ind[i] / width;

        if ( (v < 0) || (u < 0) || (v >= height) || (u >= width) )
          continue;

        idx = n_ind[i];

        // not valid or not used point
        if (!(mask_[idx]))
          continue;

        const Eigen::Vector3f &n = normals.points[idx].getNormalVector3fMap();
        const Eigen::Vector3f &pt = cloud->points[idx].getVector3fMap();

        float cosa = plane.normal.dot(n);
        float dist = fabs(plane.normal.dot(pt - plane.pt));

        // we can add this point to the plane
        if ( (cosa > cos_rad_thr_angle) && (dist < param.inlDist) )
        {
          mask_[idx] = false;
          plane.add(pt, n, idx);
          queue_.push_back(idx);
          plane.normal.normalize();
        }
      }
    }
}

/**
 * @brief ClusterNormalsToPlanesPCL<PointT>::smoothClustering
 * @param cloud
 * @param normals
 * @param idx
 * @param plane
 */
template<typename PointT>
void
ClusterNormalsToPlanesPCL<PointT>::smoothClustering(const typename pcl::PointCloud<PointT>::Ptr &cloud, const pcl::PointCloud<pcl::Normal> &normals, size_t idx, typename ClusterNormalsToPlanesPCL<PointT>::Plane &plane)
{
  plane.clear();

  if(pcl::isFinite(normals.points[idx]))
    return;

  mask_[idx] = false;

  plane.init(cloud->points[idx].getVector3fMap(), normals.points[idx].getNormalVector3fMap(), idx);

  size_t queue_idx = 0;
  int width = cloud->width;
  int height = cloud->height;

  std::vector<int> n4ind(4);

  queue_.resize(1);
  queue_[0] = idx;

  // start clustering
  while (queue_.size() > queue_idx)
  {
    // extract current index
    idx = queue_.at(queue_idx);
    queue_idx++;

    pcl::Normal na;
    na.getNormalVector3fMap();
    const Eigen::Vector3f &pt0 = cloud->points[idx].getVector3fMap();
    const Eigen::Vector3f &n0 = normals.points[idx].getNormalVector3fMap();

    n4ind[0] = idx-1;
    n4ind[1] = idx+1;
    n4ind[2] = idx+width;
    n4ind[3] = idx-width;

    for(unsigned i=0; i<n4ind.size(); i++)
    {
      int u = n4ind[i] % width;
      int v = n4ind[i] / width;

      if ( (v < 0) || (u < 0) || (v >= height) || (u >= width) )
        continue;

      idx = n4ind[i];

      // not valid or not used point
      if (!(mask_[idx]))
        continue;

      if(!pcl::isFinite(normals.points[idx]))
        continue;

      const Eigen::Vector3f &n = normals.points[idx].getNormalVector3fMap();
      const Eigen::Vector3f &pt = cloud->points[idx].getVector3fMap();

      float cosa = n0.dot(n);
      float dist = fabs(n0.dot(pt - pt0));

      // we can add this point to the plane
      if ( (cosa > cos_rad_thr_angle_smooth) && (dist < param.inlDistSmooth) )
      {
        mask_[idx] = false;
        plane.add(pt, n, idx);
        queue_.push_back(idx);
        plane.normal.normalize();
      }
    }
  }
}

/**
 * ClusterNormals
 */
template<typename PointT>
void
ClusterNormalsToPlanesPCL<PointT>::doClustering(const typename pcl::PointCloud<PointT>::Ptr &cloud, const pcl::PointCloud<pcl::Normal> &normals, std::vector<typename ClusterNormalsToPlanesPCL<PointT>::Plane::Ptr> &planes)
{
  mask_.clear();
  queue_.clear();
  mask_.resize(cloud->points.size(), true);
  queue_.reserve(cloud->points.size());

  planes.clear();

  for (size_t i = 0; i < cloud->points.size(); i++)
  {
    if( !pcl::isFinite(cloud->points[i]) || !pcl::isFinite(normals.points[i]) )
        mask_[i] = false;
  }

  // plane clustering
  for (size_t i=0; i<mask_.size(); i++)
  {
    if (mask_[i])
    {
        typename Plane::Ptr plane(new Plane(true));
        if(cloud->isOrganized())
            clusterNormals(cloud, normals, i, *plane);
        else
            clusterNormalsUnorganized(cloud, normals, i, *plane);

      if (plane->size() >= param.minPoints)
          planes.push_back(plane);
    }
  }

  // for the remaining points do a smooth clustering
  if (param.smooth_clustering)
  {
    mask_.clear();
    mask_.resize(cloud->points.size(), true);

    // mark nans
    for (size_t i = 0; i < cloud->points.size(); i++)
    {
      if(!pcl::isFinite(cloud->points[i]) || !pcl::isFinite(normals.points[i]) )
        mask_[i] = false;
    }

    // mark planes
    for (size_t i=0; i < planes.size(); i++)
    {
      const Plane &plane_tmp = *planes[i];
      for (size_t j=0; j < plane_tmp.indices.size(); j++)
          mask_[ plane_tmp.indices[j] ] = false;
    }

    // do smooth clustering
    for (size_t i=0; i<mask_.size(); i++)
    {
      if (mask_[i])
      {
        typename Plane::Ptr plane(new Plane(false));
        smoothClustering(cloud, normals, i, *plane);

        if (plane->size()>=param.minPointsSmooth)
            planes.push_back(plane);
      }
    }
  }
}


/************************** PUBLIC *************************/


/**
 * Compute
 */
template<typename PointT>
void
ClusterNormalsToPlanesPCL<PointT>::compute(const typename pcl::PointCloud<PointT>::Ptr &_cloud, const pcl::PointCloud<pcl::Normal> &_normals, std::vector<PlaneModel<PointT> > &_planes)
{
  std::vector<typename ClusterNormalsToPlanesPCL<PointT>::Plane::Ptr> planes;


  PlaneModel<PointT> pm;

  if ( ! _cloud->isOrganized() ) {
      // Create the filtering object: downsample the dataset using a leaf size of 1cm
      pcl::VoxelGrid<PointT> vg;
      typename pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
      vg.setInputCloud (_cloud);
      float leaf_size_ = 0.005f;
      vg.setLeafSize (leaf_size_, leaf_size_, leaf_size_);
      vg.filter (*cloud_filtered);
      pm.cloud_ = cloud_filtered;
      pcl::PointCloud<pcl::Normal>::Ptr normals_ds (new pcl::PointCloud<pcl::Normal>);
      computeNormals<PointT>(pm.cloud_, normals_ds, 2);
      doClustering(pm.cloud_, *normals_ds, planes);
  }
  else {
      pm.cloud_ = _cloud;
      doClustering(pm.cloud_, _normals, planes);
  }

  _planes.clear();

  if (param.least_squares_refinement) {
      for (size_t i=0; i<planes.size(); i++) {
          if( !planes[i]->is_plane )
              continue;
          pm.inliers_ = planes[i]->indices;
          float curvature;
          pcl::computePointNormal<PointT>(*pm.cloud_, pm.inliers_, pm.coefficients_, curvature);
//          model_coeff.normalize();
          _planes.push_back( pm );
      }
  }
}

/**
 * @brief compute
 * @param cloud
 * @param normals
 * @param x
 * @param y
 * @param plane
 */
template<typename PointT>
void
ClusterNormalsToPlanesPCL<PointT>::compute(const typename pcl::PointCloud<PointT>::Ptr &cloud, const pcl::PointCloud<pcl::Normal> &normals, int x, int y, PlaneModel<PointT> &pm)
{
  mask_.clear();
  mask_.resize(cloud->height*cloud->width,true);

  for (int i = 0; i < cloud->height*cloud->width; i++)
  {
    if(!pcl::isFinite(cloud->points[i]))
      mask_[i] = false;
  }

  size_t idx = y*cloud->width+x;
  typename ClusterNormalsToPlanesPCL<PointT>::Plane plane;

  if (idx < mask_.size() && mask_[idx])
  {
      if(cloud->isOrganized())
          clusterNormals(cloud, normals, idx, plane);
      else
          clusterNormalsUnorganized(cloud, normals, idx, plane);

      pm.cloud_ = cloud;
      pm.inliers_ = plane.indices;

        float curvature;
        pcl::computePointNormal<PointT>(*pm.cloud_, pm.inliers_, pm.coefficients_, curvature);
  }
}

template class V4R_EXPORTS ClusterNormalsToPlanesPCL<pcl::PointXYZRGB>;
template class V4R_EXPORTS ClusterNormalsToPlanesPCL<pcl::PointXYZ>;

} //-- THE END --

