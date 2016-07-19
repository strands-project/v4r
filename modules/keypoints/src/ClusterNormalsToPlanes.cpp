/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#include "v4r/keypoints/ClusterNormalsToPlanes.h"

namespace v4r
{

using namespace std;
  
/********************** ClusterNormalsToPlanes ************************
 * Constructor/Destructor
 */
ClusterNormalsToPlanes::ClusterNormalsToPlanes(const Parameter &_p)
 : param(_p)
{
  cos_rad_thr_angle = cos(param.thrAngle*M_PI/180.);
  cos_rad_thr_angle_smooth = cos(param.thrAngleSmooth*M_PI/180.);
}

ClusterNormalsToPlanes::~ClusterNormalsToPlanes()
{
}

/************************** PRIVATE ************************/


/**
 * ClusterNormals
 */
void ClusterNormalsToPlanes::clusterNormals(const v4r::DataMatrix2D<Eigen::Vector3f> &cloud, const v4r::DataMatrix2D<Eigen::Vector3f> &normals, int idx, Plane &plane)
{
  plane.clear();

  if(isnan(normals[idx]))
    return;

  mask[idx] = false;

  plane.init(cloud[idx], normals[idx], idx);
  
  int queue_idx = 0;
  int width = cloud.cols;
  int height = cloud.rows;

  std::vector<int> n4ind(4);

  queue.resize(1);
  queue[0] = idx;
  
  // start clustering
  while (((int)queue.size()) > queue_idx)
  {
    // extract current index
    idx = queue.at(queue_idx);
    queue_idx++;

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
      if (!(mask[idx]))
        continue;

      const Eigen::Vector3f &n = normals[idx];

      if(isnan(n))
        continue;

      const Eigen::Vector3f &pt = cloud[idx];

      float cosa = plane.normal.dot(n);
      float dist = fabs(plane.normal.dot(pt - plane.pt));

      // we can add this point to the plane
      if ( (cosa > cos_rad_thr_angle) && (dist < param.inlDist) )
      {
        mask[idx] = false;

        plane.add(pt, n, idx);

        queue.push_back(idx);

        plane.normal.normalize();
      }
    }
  }
}

/**
 * @brief ClusterNormalsToPlanes::smoothClustering
 * @param cloud
 * @param normals
 * @param idx
 * @param plane
 */
void ClusterNormalsToPlanes::smoothClustering(const v4r::DataMatrix2D<Eigen::Vector3f> &cloud, const v4r::DataMatrix2D<Eigen::Vector3f> &normals, int idx, Plane &plane)
{
  plane.clear();

  if(isnan(normals[idx]))
    return;

  mask[idx] = false;

  plane.init(cloud[idx], normals[idx], idx);

  int queue_idx = 0;
  int width = cloud.cols;
  int height = cloud.rows;

  std::vector<int> n4ind(4);

  queue.resize(1);
  queue[0] = idx;

  // start clustering
  while (((int)queue.size()) > queue_idx)
  {
    // extract current index
    idx = queue.at(queue_idx);
    queue_idx++;

    const Eigen::Vector3f &pt0 = cloud[idx];
    const Eigen::Vector3f &n0 = normals[idx];

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
      if (!(mask[idx]))
        continue;

      const Eigen::Vector3f &n = normals[idx];

      if(isnan(n))
        continue;

      const Eigen::Vector3f &pt = cloud[idx];

      float cosa = n0.dot(n);
      float dist = fabs(n0.dot(pt - pt0));

      // we can add this point to the plane
      if ( (cosa > cos_rad_thr_angle_smooth) && (dist < param.inlDistSmooth) )
      {
        mask[idx] = false;

        plane.add(pt, n, idx);

        queue.push_back(idx);

        plane.normal.normalize();
      }
    }
  }
}

/**
 * ClusterNormals
 */
void ClusterNormalsToPlanes::doClustering(const v4r::DataMatrix2D<Eigen::Vector3f> &cloud, const v4r::DataMatrix2D<Eigen::Vector3f> &normals, std::vector<Plane::Ptr> &planes)
{
  mask.clear();
  mask.resize(cloud.rows*cloud.cols,true);

  queue.reserve(cloud.rows*cloud.cols);

  Plane::Ptr plane(new Plane(true));
  planes.clear();

  for (int i = 0; i < cloud.rows*cloud.cols; i++)
  {
    if(isnan(cloud[i]))
      mask[i] = false;
  }

  // plane clustering
  for (unsigned i=0; i<mask.size(); i++)
  {
    if (mask[i])
    {
      clusterNormals(cloud, normals, i, *plane);

      if ((int)plane->size()>=param.minPoints)
      {
        planes.push_back(plane);
        plane.reset(new Plane(true));
      }
    }
  }

  // for the remaining points do a smooth clustering
  if (param.smooth_clustering)
  {
    mask.clear();
    mask.resize(cloud.rows*cloud.cols,true);

    // mark nans
    for (int i = 0; i < cloud.rows*cloud.cols; i++)
    {
      if(isnan(cloud[i]))
        mask[i] = false;
    }
    // mark planes
    for (unsigned i=0; i<planes.size(); i++)
    {
      Plane &plane_tmp = *planes[i];
      for (unsigned j=0; j<plane_tmp.size(); j++)
        mask[plane_tmp.indices[j]] = false;
    }

    // do smooth clustering
    plane.reset(new Plane(false));

    for (unsigned i=0; i<mask.size(); i++)
    {
      if (mask[i])
      {
        smoothClustering(cloud, normals, i, *plane);

        if ((int)plane->size()>=param.minPointsSmooth)
        {
          planes.push_back(plane);
          plane.reset(new Plane(false));
        }
      }
    }
  }
}

/**
 * @brief ClusterNormalsToPlanes::computeLeastSquarePlanes
 * @param _cloud
 * @param planes
 */
void ClusterNormalsToPlanes::computeLeastSquarePlanes(const v4r::DataMatrix2D<Eigen::Vector3f> &cloud, std::vector<Plane::Ptr> &planes)
{
  Eigen::Vector3f n;

  for (unsigned i=0; i<planes.size(); i++)
  {
    Plane &plane = *planes[i];

    if (!plane.is_plane)
      continue;

    pest.estimatePlaneLS(cloud.data, plane.indices, plane.pt, n);

    if (plane.normal.dot(n) >= 0)
      plane.normal = n;
    else plane.normal = -n;
  }
}



/************************** PUBLIC *************************/


/**
 * Compute
 */
void ClusterNormalsToPlanes::compute(const v4r::DataMatrix2D<Eigen::Vector3f> &_cloud, const v4r::DataMatrix2D<Eigen::Vector3f> &_normals, std::vector<Plane::Ptr> &planes)
{
  planes.clear();

  doClustering(_cloud, _normals, planes);
  
  if (param.least_squares_refinement) computeLeastSquarePlanes(_cloud, planes);
}

/**
 * @brief compute
 * @param cloud
 * @param normals
 * @param x
 * @param y
 * @param plane
 */
void ClusterNormalsToPlanes::compute(const v4r::DataMatrix2D<Eigen::Vector3f> &cloud, const v4r::DataMatrix2D<Eigen::Vector3f> &normals, int x, int y, Plane &plane)
{
  mask.clear();
  mask.resize(cloud.rows*cloud.cols,true);

  for (int i = 0; i < cloud.rows*cloud.cols; i++)
  {
    if(isnan(cloud[i]))
      mask[i] = false;
  }

  Eigen::Vector3f n;
  int idx = y*cloud.cols+x;
  plane.clear();

  if (idx < (int)mask.size() && mask[idx])
  {
    clusterNormals(cloud, normals, idx, plane);

    pest.estimatePlaneLS(cloud.data, plane.indices, plane.pt, n);

    if (plane.normal.dot(n) >= 0)
      plane.normal = n;
    else plane.normal = -n;
  }
}

} //-- THE END --

