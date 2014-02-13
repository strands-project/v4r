/*
 * registration_utils.h
 *
 *  Created on: Nov 6, 2013
 *      Author: aitor
 */

#ifndef FAAT_PCL_REGISTRATION_UTILS_H_
#define FAAT_PCL_REGISTRATION_UTILS_H_

#include <pcl/octree/octree.h>
#include <pcl/filters/voxel_grid.h>
#include <numeric>

namespace faat_pcl
{
  namespace registration_utils
  {
    template<class PointType>
    void
    computeOverlapMatrix (std::vector<typename pcl::PointCloud<PointType>::Ptr> & clouds,
                             std::vector<std::vector<bool> > & A,
                             float inlier = 0.01f,
                             bool fast_overlap=true,
                             float ff = 0.3f)
    {
      if (fast_overlap)
      {
        PointType min_pt_all, max_pt_all;
        min_pt_all.x = min_pt_all.y = min_pt_all.z = std::numeric_limits<float>::max ();
        max_pt_all.x = max_pt_all.y = max_pt_all.z = (std::numeric_limits<float>::max () - 0.001f) * -1;

        for (size_t i = 0; i < clouds.size (); i++)
        {
          PointType min_pt, max_pt;
          pcl::getMinMax3D (*clouds[i], min_pt, max_pt);
          if (min_pt.x < min_pt_all.x)
            min_pt_all.x = min_pt.x;

          if (min_pt.y < min_pt_all.y)
            min_pt_all.y = min_pt.y;

          if (min_pt.z < min_pt_all.z)
            min_pt_all.z = min_pt.z;

          if (max_pt.x > max_pt_all.x)
            max_pt_all.x = max_pt.x;

          if (max_pt.y > max_pt_all.y)
            max_pt_all.y = max_pt.y;

          if (max_pt.z > max_pt_all.z)
            max_pt_all.z = max_pt.z;
        }

        float res_occupancy_grid_ = 0.01f;
        int size_x, size_y, size_z;
        size_x = static_cast<int> (std::ceil (std::abs (max_pt_all.x - min_pt_all.x) / res_occupancy_grid_)) + 1;
        size_y = static_cast<int> (std::ceil (std::abs (max_pt_all.y - min_pt_all.y) / res_occupancy_grid_)) + 1;
        size_z = static_cast<int> (std::ceil (std::abs (max_pt_all.z - min_pt_all.z) / res_occupancy_grid_)) + 1;

        for (size_t i = 0; i < clouds.size (); i++)
        {
          std::vector<int> complete_cloud_occupancy_by_RM_;
          complete_cloud_occupancy_by_RM_.resize (size_x * size_y * size_z, 0);
          for (size_t k = 0; k < clouds[i]->points.size (); k++)
          {
            int pos_x, pos_y, pos_z;
            pos_x = static_cast<int> (std::floor ((clouds[i]->points[k].x - min_pt_all.x) / res_occupancy_grid_));
            pos_y = static_cast<int> (std::floor ((clouds[i]->points[k].y - min_pt_all.y) / res_occupancy_grid_));
            pos_z = static_cast<int> (std::floor ((clouds[i]->points[k].z - min_pt_all.z) / res_occupancy_grid_));
            int idx = pos_z * size_x * size_y + pos_y * size_x + pos_x;
            complete_cloud_occupancy_by_RM_[idx] = 1;
          }

          int total_points_i = std::accumulate (complete_cloud_occupancy_by_RM_.begin (), complete_cloud_occupancy_by_RM_.end (), 0);

          std::vector<int> complete_cloud_occupancy_by_RM_j;
          complete_cloud_occupancy_by_RM_j.resize (size_x * size_y * size_z, 0);

          for (size_t j = i; j < clouds.size (); j++)
          {
            int overlap = 0;
            std::map<int, bool> banned;
            std::map<int, bool>::iterator banned_it;

            for (size_t k = 0; k < clouds[j]->points.size (); k++)
            {
              int pos_x, pos_y, pos_z;
              pos_x = static_cast<int> (std::floor ((clouds[j]->points[k].x - min_pt_all.x) / res_occupancy_grid_));
              pos_y = static_cast<int> (std::floor ((clouds[j]->points[k].y - min_pt_all.y) / res_occupancy_grid_));
              pos_z = static_cast<int> (std::floor ((clouds[j]->points[k].z - min_pt_all.z) / res_occupancy_grid_));
              int idx = pos_z * size_x * size_y + pos_y * size_x + pos_x;
              banned_it = banned.find (idx);
              if (banned_it == banned.end ())
              {
                complete_cloud_occupancy_by_RM_j[idx] = 1;
                if (complete_cloud_occupancy_by_RM_[idx] > 0)
                  overlap++;
                banned[idx] = true;
              }
            }

            float total_points_j = std::accumulate (complete_cloud_occupancy_by_RM_j.begin (), complete_cloud_occupancy_by_RM_j.end (), 0);

            float ov_measure_1 = overlap / static_cast<float> (total_points_j);
            float ov_measure_2 = overlap / static_cast<float> (total_points_i);
            if (!(ov_measure_1 > ff || ov_measure_2 > ff))
            {
              A[i][j] = false;
              A[j][i] = false;
            }
          }
        }
      }
      else
      {
        std::vector<typename pcl::PointCloud<PointType>::Ptr> clouds2 = clouds;

        {
          std::vector<typename pcl::PointCloud<PointType>::Ptr> clouds;
          clouds.resize(clouds2.size());
          for(size_t i=0; i < clouds.size(); i++)
          {
            float voxel_grid_size = 0.005f;
            pcl::VoxelGrid<PointType> grid_;
            grid_.setInputCloud (clouds2[i]);
            grid_.setLeafSize (voxel_grid_size, voxel_grid_size, voxel_grid_size);
            clouds[i].reset(new pcl::PointCloud<PointType>);
            grid_.filter (*clouds[i]);
          }

          std::vector<int> pointIdxNKNSearch;
          std::vector<float> pointNKNSquaredDistance;

          for (size_t i = 0; i < clouds.size (); i++)
          {
            A[i][i] = false;
            pcl::octree::OctreePointCloudSearch<PointType> octree (0.003);
            octree.setInputCloud (clouds[i]);
            octree.addPointsFromInputCloud ();

            for (size_t j = i; j < clouds.size (); j++)
            {
              //compute overlap
              int overlap = 0;
              for (size_t kk = 0; kk < clouds[j]->points.size (); kk++)
              {
                if (pcl_isnan (clouds[j]->points[kk].x))
                  continue;

                if (octree.nearestKSearch (clouds[j]->points[kk], 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
                {
                  float d = sqrt (pointNKNSquaredDistance[0]);
                  if (d < inlier)
                  {
                    overlap++;
                  }
                }
              }

              float ov_measure_1 = overlap / static_cast<float> (clouds[j]->points.size ());
              float ov_measure_2 = overlap / static_cast<float> (clouds[i]->points.size ());
              if (!(ov_measure_1 > ff || ov_measure_2 > ff))
              {
                A[i][j] = false;
                A[j][i] = false;
              }
            }
          }
        }
      }
    }
  }
}
#endif /* REGISTRATION_UTILS_H_ */
