/*
 * objectness_3D.H
 *
 *  Created on: Jul 27, 2012
 *      Author: aitor
 */

#include "faat_pcl/segmentation/objectness_3d/cuda/cuda_objectness_3D.h"
#include "faat_pcl/segmentation/objectness_3d/objectness_3D.h"
#include "faat_pcl/segmentation/objectness_3d/bbox_optimizer.h"
#include <boost/unordered_map.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include "pcl/recognition/hv/occlusion_reasoning.h"
#include "pcl/filters/voxel_grid.h"
#include "pcl/common/time.h"
//#include "faat_pcl/segmentation/objectness_3d/cuda/cuda_bbox_optimizer_wrapper.h"

namespace faat_pcl
{
  namespace segmentation
  {
    inline void
    visualizeIVData (int * data, int GRIDSIZE_X, int GRIDSIZE_Y, int GRID_SIZE_Z)
    {
      pcl::visualization::PCLVisualizer vis ("visualizeIVData");

      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
      for (int x = 0; x < GRIDSIZE_X; x++)
      {
        for (int y = 0; y < GRIDSIZE_Y; y++)
        {
          for (int z = 0; z < GRID_SIZE_Z; z++)
          {
            int idx = z * GRIDSIZE_X * GRIDSIZE_Y + y * GRIDSIZE_X + x;
            if (data[idx])
            {
              pcl::PointXYZ p;
              p.getVector3fMap () = Eigen::Vector3f (x, y, z);
              cloud->push_back (p);
            }
          }
        }
      }

      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> random_handler (cloud, 255, 0, 0);
      vis.addPointCloud<pcl::PointXYZ> (cloud, random_handler, "original points");
      vis.addCoordinateSystem (100);
      vis.spin ();
    }

    inline void
    dilateIVData (int * data, int GRIDSIZE_X, int GRIDSIZE_Y, int GRID_SIZE_Z, int iterations = 1, int ws2 = 2)
    {
      for (size_t it = 0; it < iterations; it++)
      {
        int * dilated = new int[GRIDSIZE_X * GRIDSIZE_Y * GRID_SIZE_Z];
        for (int x = 0; x < (GRIDSIZE_X * GRIDSIZE_Y * GRID_SIZE_Z); x++)
        {
          dilated[x] = 0;
        }

        for (int x = ws2; x < (GRIDSIZE_X - ws2); x++)
        {
          for (int y = ws2; y < (GRIDSIZE_Y - ws2); y++)
          {
            for (int z = ws2; z < (GRID_SIZE_Z - ws2); z++)
            {
              int idx_orig = z * GRIDSIZE_X * GRIDSIZE_Y + y * GRIDSIZE_X + x;
              if (data[idx_orig] > 0)
              {
                dilated[idx_orig] = data[idx_orig];
              }
              else
              {
                bool found = false;
                for (int wx = (x - ws2); wx < (x + ws2); wx++)
                {
                  for (int wy = (y - ws2); wy < (y + ws2); wy++)
                  {
                    for (int wz = (z - ws2); wz < (z + ws2); wz++)
                    {
                      int idx = wz * GRIDSIZE_X * GRIDSIZE_Y + wy * GRIDSIZE_X + wx;
                      if (data[idx] > 0)
                      {
                        found = true;
                      }
                    }
                  }
                }

                if (found)
                {
                  dilated[idx_orig] = 1;
                }
              }
            }
          }
        }

        memcpy (data, dilated, sizeof(int) * GRIDSIZE_X * GRIDSIZE_Y * GRID_SIZE_Z);
        delete[] dilated;
      }
    }

    inline void
    erodeIVData (int * data, int GRIDSIZE_X, int GRIDSIZE_Y, int GRID_SIZE_Z, int iterations = 1, int ws2 = 2)
    {
      for (size_t it = 0; it < iterations; it++)
      {
        int * eroded = new int[GRIDSIZE_X * GRIDSIZE_Y * GRID_SIZE_Z];
        for (int x = 0; x < (GRIDSIZE_X * GRIDSIZE_Y * GRID_SIZE_Z); x++)
        {
          eroded[x] = 0;
        }

        for (int x = ws2; x < (GRIDSIZE_X - ws2); x++)
        {
          for (int y = ws2; y < (GRIDSIZE_Y - ws2); y++)
          {
            for (int z = ws2; z < (GRID_SIZE_Z - ws2); z++)
            {
              int idx_orig = z * GRIDSIZE_X * GRIDSIZE_Y + y * GRIDSIZE_X + x;
              if (data[idx_orig] == 0)
              {
                eroded[idx_orig] = data[idx_orig];
              }
              else
              {
                bool found = true;
                for (int wx = (x - ws2); wx < (x + ws2); wx++)
                {
                  for (int wy = (y - ws2); wy < (y + ws2); wy++)
                  {
                    for (int wz = (z - ws2); wz < (z + ws2); wz++)
                    {
                      int idx = wz * GRIDSIZE_X * GRIDSIZE_Y + wy * GRIDSIZE_X + wx;
                      if (data[idx] <= 0)
                      {
                        found = false;
                      }
                    }
                  }
                }

                if (!found)
                {
                  eroded[idx_orig] = 0;
                }
                else
                {
                  eroded[idx_orig] = data[idx_orig];
                }
              }
            }
          }
        }

        memcpy (data, eroded, sizeof(int) * GRIDSIZE_X * GRIDSIZE_Y * GRID_SIZE_Z);
        delete[] eroded;
      }
    }

    template<typename PointInT>
      void
      Objectness3D<PointInT>::getObjectIndices (std::vector<pcl::PointIndices> & indices, PointInTPtr & cloud)
      {

        PointInTPtr cloud_no_plane (new pcl::PointCloud<PointInT> ());
        pcl::transformPointCloud (*cloud, *cloud_no_plane, transform_to_plane);
        float plane_threshold = 0.0075f;
        for (size_t i = 0; i < cloud_no_plane->points.size (); i++)
        {
          //check that the edge point is inside of the VOI
          if (!pcl::isFinite (cloud_no_plane->points[i]))
            continue;

          Eigen::Vector3f p = cloud_no_plane->points[i].getVector3fMap ();
          if (p[0] > max_x || p[1] > max_y || p[2] > max_z)
          {
            cloud_no_plane->points[i].x = std::numeric_limits<float>::quiet_NaN ();
            cloud_no_plane->points[i].y = std::numeric_limits<float>::quiet_NaN ();
            cloud_no_plane->points[i].z = std::numeric_limits<float>::quiet_NaN ();
          }
          if (p[0] < min_x || p[1] < min_y || p[2] < (min_z + plane_threshold))
          {
            cloud_no_plane->points[i].x = std::numeric_limits<float>::quiet_NaN ();
            cloud_no_plane->points[i].y = std::numeric_limits<float>::quiet_NaN ();
            cloud_no_plane->points[i].z = std::numeric_limits<float>::quiet_NaN ();
          }

          if (p[2] < plane_threshold)
          {
            cloud_no_plane->points[i].x = std::numeric_limits<float>::quiet_NaN ();
            cloud_no_plane->points[i].y = std::numeric_limits<float>::quiet_NaN ();
            cloud_no_plane->points[i].z = std::numeric_limits<float>::quiet_NaN ();
          }

        }
        for (size_t i = 0; i < final_boxes_.size (); i++)
        {
          //PointInTPtr cloud_for_iv (new pcl::PointCloud<PointInT> (*cloud_));
          //rotate the point cloud based on the bbox angle
          BBox bb = final_boxes_[i];
          int v = bb.angle;
          Eigen::Affine3f incr_rot_trans;
          incr_rot_trans.setIdentity ();

          /*Eigen::Vector4f minxyz, maxxyz;
          minxyz[0] = min_x + (bb.x) * resolution - resolution;
          minxyz[1] = min_y + (bb.y) * resolution - resolution;
          minxyz[2] = min_z + (bb.z) * resolution - resolution;
          minxyz[3] = 1.f;

          maxxyz[0] = min_x + (bb.sx + bb.x) * resolution + resolution;
          maxxyz[1] = min_y + (bb.sy + bb.y) * resolution + resolution;
          maxxyz[2] = min_z + (bb.sz + bb.z) * resolution + resolution;
          maxxyz[3] = 1.f;*/

          Eigen::Vector4f minxyz, maxxyz;
          minxyz[0] = min_x + (bb.x) * resolution - resolution / 2.f;
          minxyz[1] = min_y + (bb.y) * resolution - resolution / 2.f;
          minxyz[2] = min_z + (bb.z) * resolution - resolution / 2.f;
          minxyz[3] = 1.f;

          maxxyz[0] = min_x + (bb.sx + bb.x) * resolution + resolution / 2.f;
          maxxyz[1] = min_y + (bb.sy + bb.y) * resolution + resolution / 2.f;
          maxxyz[2] = min_z + (bb.sz + bb.z) * resolution + resolution / 2.f;
          maxxyz[3] = 1.f;

          if (v != 0)
          {

            float rot_rads = pcl::deg2rad (static_cast<float> (angle_incr_ * v));
            incr_rot_trans = Eigen::Affine3f (Eigen::AngleAxisf (static_cast<float> (rot_rads), Eigen::Vector3f::UnitZ ()));
          }

          pcl::CropBox<PointInT> cb;
          cb.setInputCloud (cloud_no_plane);
          cb.setMin (minxyz);
          cb.setMax (maxxyz);
          cb.setTransform (incr_rot_trans);
          std::vector<int> inside_box;
          cb.filter (inside_box);
          pcl::PointIndices pi;
          pi.indices = inside_box;
          indices.push_back (pi);
        }
      }

    template<typename PointInT>
      void
      Objectness3D<PointInT>::optimizeBoundingBoxes (PointInTPtr & cloud_for_iv, std::vector<BBox> & bounding_boxes,
                                                     Eigen::Matrix4f & transform_to_plane)
      {

        std::vector<float> free_space_vector;

        for (size_t i = 0; i < bounding_boxes.size (); i++)
        {
          BBox bb = bounding_boxes[i];
          //bounding_boxes[i].score = 1.f;
          int occupancy_val;
          rivs_occupancy[bb.angle]->getRectangleFromCorner (bb.x, bb.y, bb.z, bb.sx, bb.sy, bb.sz, occupancy_val);

          int occupancy_val_full;
          rivs_occupancy_complete_[bb.angle]->getRectangleFromCorner (bb.x, bb.y, bb.z, bb.sx, bb.sy, bb.sz, occupancy_val_full);

          int occluded_val;
          rivs_occluded[bb.angle]->getRectangleFromCorner (bb.x, bb.y, bb.z, bb.sx, bb.sy, bb.sz, occluded_val);
          float vol_outer = static_cast<float> (bb.sx * bb.sy * bb.sz);
          float free_space = vol_outer - occluded_val - occupancy_val;
          float free_space_score = (vol_outer - free_space) / vol_outer;

          //bounding_boxes[i].score *= (occupancy_val + 1.f * occluded_val) / static_cast<float>(bb.sx * bb.sy * bb.sz);
          //bounding_boxes[i].score += 0.25f * free_space_score;

          int occupancy_val_proj;
          int occluded_val_proj;

          projected_occluded_[bb.angle]->getRectangleFromCorner (bb.x, bb.y, bb.sz, bb.sx, bb.sy, 1, occluded_val_proj);
          projected_occupancy_[bb.angle]->getRectangleFromCorner (bb.x, bb.y, bb.sz, bb.sx, bb.sy, 1, occupancy_val_proj);

          //get info of the inner bounding box
          /*BBox bb_shrinked;
           shrink_bbox (bb, bb_shrinked);
           int inner_edges;
           rivs[bb.angle]->getRectangleFromCorner (bb_shrinked.x, bb_shrinked.y, bb_shrinked.z, bb_shrinked.sx, bb_shrinked.sy, bb_shrinked.sz,
           inner_edges);*/

          float free_space2 = (bb.sx * bb.sy - occluded_val_proj - occupancy_val_proj) * bb.sz;
          float free_space2_score = (vol_outer - free_space2) / vol_outer;
          //float combi = (free_space + free_space2) / 2.f;
          //combi = free_space_inner + free_space2;
          //combi = free_space2 * free_space;
          //std::cout << combi << " " << free_space2 << " " << free_space << std::endl;

          //combi = (free_space_score * free_space2_score) * occupancy_val_full;
          //combi = free_space2;
          //std::cout << combi << " " << free_space2_score << " " << free_space_score << std::endl;
          //combi = free_space_score / 2.f * occupancy_val_full;
          //combi = 0;
          //combi = (free_space / 4.f);
          float combi = (1.f - free_space_score) * occupancy_val_full;
          combi = free_space;
          //combi = (1.f - free_space2_score) * occupancy_val_full;
          //combi += (1.f - free_space2_score) * occupancy_val_full;
          //combi /= 2.f;
          //combi = 0;
          free_space_vector.push_back (combi);
        }

        PointInTPtr cloud_for_bbo2 (new pcl::PointCloud<PointInT>);

        for (size_t i = 0; i < cloud_for_iv->points.size (); i++)
        {
          if (!pcl::isFinite (cloud_for_iv->points[i]))
            continue;

          Eigen::Vector3f p = cloud_for_iv->points[i].getVector3fMap ();

          if (p[2] >= (0.01f))
            cloud_for_bbo2->points.push_back (cloud_for_iv->points[i]);
        }

        PointInTPtr cloud_for_bbo (new pcl::PointCloud<PointInT>);
        pcl::VoxelGrid<PointInT> voxel_grid;
        voxel_grid.setInputCloud (cloud_for_bbo2);
        voxel_grid.setLeafSize (0.005f, 0.005f, 0.005f);
        voxel_grid.filter (*cloud_for_bbo);

        /*pcl::visualization::PCLVisualizer vis ("cloud_for_bbo");
        vis.addPointCloud<PointInT> (cloud_for_bbo, "edges");
        vis.addCoordinateSystem (0.3f);
        vis.spin ();*/

        std::vector<bool> mask;
        /*if (do_cuda_)
        {
          faat_pcl::cuda::segmentation::CudaBBoxOptimizerWrapper<PointInT> bbo_cuda (0.5f);
          bbo_cuda.setCloud (cloud_for_bbo);
          bbo_cuda.setMinMaxValues (min_x, max_x, min_y, max_y, min_z, max_z);
          bbo_cuda.setResolution (resolution);
          bbo_cuda.setAngleIncr (angle_incr_);
          bbo_cuda.addModels (bounding_boxes, free_space_vector);

          {
            pcl::ScopeTime t ("Optimization cuda");
            bbo_cuda.optimize ();
          }

          bbo_cuda.getMask(mask);
        }
        else
        {*/
        BBoxOptimizer<PointInT> bbo (0.1f);
        bbo.setOptType(opt_type_);
        bbo.setCloud (cloud_for_bbo);
        bbo.setMinMaxValues (min_x, max_x, min_y, max_y, min_z, max_z);
        bbo.setResolution (resolution);
        bbo.angle_incr_ = angle_incr_;

        {
          pcl::ScopeTime t ("Adding models to the optimizer");
          bbo.addModels (bounding_boxes, free_space_vector);
        }

        {
          pcl::ScopeTime t ("Optimization");
          bbo.optimize ();
        }

        bbo.getMask (mask);
        //}

        std::vector<BBox> after_opt;
        std::vector<float> free_space_vector_after_opt;

        for (size_t i = 0; i < mask.size (); i++)
        {
          if (mask[i])
          {
            std::cout << "This hypothesis remains active..." << std::endl;
            after_opt.push_back (bounding_boxes[i]);
            free_space_vector_after_opt.push_back (free_space_vector[i]);
          }
        }

        free_space_vector = free_space_vector_after_opt;
        bounding_boxes = after_opt;

        bool extend_boxes = false;
        if (extend_boxes)
        {
          std::vector<float> extended_free_space_vector;
          std::vector<BBox> extended_bounding_boxes;

          for (size_t i = 0; i < bounding_boxes.size (); i++)
          {
            extended_bounding_boxes.push_back (bounding_boxes[i]);
            extended_free_space_vector.push_back (free_space_vector[i]);
          }

          for (size_t i = 0; i < bounding_boxes.size (); i++)
          {
            {
              BBox bb = bounding_boxes[i];
              bb.x--;
              extended_bounding_boxes.push_back (bb);
              extended_free_space_vector.push_back (free_space_vector[i]);
            }

            {
              BBox bb = bounding_boxes[i];
              bb.y--;
              extended_bounding_boxes.push_back (bb);
              extended_free_space_vector.push_back (free_space_vector[i]);
            }

            {
              BBox bb = bounding_boxes[i];
              bb.x++;
              extended_bounding_boxes.push_back (bb);
              extended_free_space_vector.push_back (free_space_vector[i]);
            }

            {
              BBox bb = bounding_boxes[i];
              bb.y++;
              extended_bounding_boxes.push_back (bb);
              extended_free_space_vector.push_back (free_space_vector[i]);
            }

            {
              BBox bb = bounding_boxes[i];
              if (bb.z > 1)
              {
                bb.z--;
                extended_bounding_boxes.push_back (bb);
                extended_free_space_vector.push_back (free_space_vector[i]);
              }
            }

            {
              BBox bb = bounding_boxes[i];
              bb.z++;
              extended_bounding_boxes.push_back (bb);
              extended_free_space_vector.push_back (free_space_vector[i]);
            }

            {
              BBox bb = bounding_boxes[i];
              bb.sz++;
              extended_bounding_boxes.push_back (bb);
              extended_free_space_vector.push_back (free_space_vector[i]);

              bb.sz++;
              extended_bounding_boxes.push_back (bb);
              extended_free_space_vector.push_back (free_space_vector[i]);
            }

            {
              BBox bb = bounding_boxes[i];
              bb.sx++;
              extended_bounding_boxes.push_back (bb);
              extended_free_space_vector.push_back (free_space_vector[i]);
            }

            {
              BBox bb = bounding_boxes[i];
              bb.sy++;
              extended_bounding_boxes.push_back (bb);
              extended_free_space_vector.push_back (free_space_vector[i]);
            }
          }

          BBoxOptimizer<PointInT> bbo (1.f);
          bbo.setCloud (cloud_for_bbo);

          {
            pcl::ScopeTime t ("Adding models to the optimizer");
            bbo.addModels (extended_bounding_boxes, extended_free_space_vector);
          }
          bbo.setMinMaxValues (min_x, max_x, min_y, max_y, min_z, max_z);
          bbo.setResolution (resolution);
          bbo.angle_incr_ = angle_incr_;

          {
            pcl::ScopeTime t ("Optimization");
            bbo.optimize ();
          }

          std::vector<bool> mask;
          bbo.getMask (mask);

          std::vector<BBox> after_opt;
          for (size_t i = 0; i < mask.size (); i++)
          {
            if (mask[i])
            {
              std::cout << "This hypothesis remains active..." << std::endl;
              after_opt.push_back (extended_bounding_boxes[i]);
            }
          }

          bounding_boxes = after_opt;

          std::cout << "Extended bounding boxes:" << bounding_boxes.size () << std::endl;
        }
      }

    template<typename PointInT>
      void
      Objectness3D<PointInT>::createRotatedIV (PointInTPtr & cloud_for_iv_original, PointInTPtr & occluded_cloud_transformed_back_original)
      {

        projected_occupancy_.clear ();
        projected_occluded_.clear ();
        rivs_occupancy.clear ();
        rivs_occupancy_complete_.clear ();
        rivs.clear ();
        rivs_occluded.clear ();
        rivs_full.clear();

        //edges_heat_maps.clear ();
        rivhistograms.clear ();
        riv_color_histograms_.clear();
        riv_points_color_histogram_.clear();
        riv_squared_color_histograms_.clear();
        npoints_label_.clear();

        int num_ivs = (90 / angle_incr_); //number of integral volumes to compute
        rivs.resize (num_ivs);
        rivs_occupancy.resize (num_ivs);
        rivs_occluded.resize (num_ivs);
        projected_occupancy_.resize (num_ivs);
        projected_occluded_.resize (num_ivs);
        rivhistograms.resize (num_ivs);
        rivs_occupancy_complete_.resize (num_ivs);
        riv_color_histograms_.resize(num_ivs);
        riv_points_color_histogram_.resize(num_ivs);
        riv_squared_color_histograms_.resize(num_ivs);
        rivs_full.resize(num_ivs);
        npoints_label_.resize(num_ivs);

        int GRIDSIZE_X = (int)((max_x - min_x) / resolution);
        int GRIDSIZE_Y = (int)((max_y - min_y) / resolution);
        int GRIDSIZE_Z = (int)((max_z - min_z) / resolution);

        if (smooth_labels_cloud_)
        {
          for (int v = 0; v < num_ivs; v++)
          {
            rivhistograms[v].resize ((max_label_ + 1));
          }
        }

        for (int v = 0; v < num_ivs; v++)
        {
          riv_color_histograms_[v].resize(ycolor_size_ * ucolor_size_ * vcolor_size_); //r,g,b
          riv_squared_color_histograms_[v].resize(ycolor_size_ * ucolor_size_ * vcolor_size_); //r,g,b
        }

//#pragma omp parallel for schedule(dynamic, 1) num_threads(omp_get_num_procs())
        for (int v = 0; v < num_ivs; v++)
        {

          PointInTPtr cloud_for_iv(new pcl::PointCloud<PointInT>);
          PointInTPtr occluded_cloud_transformed_back(new pcl::PointCloud<PointInT>);
          pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_smooth(new pcl::PointCloud<pcl::PointXYZL>);

          if (v != 0)
          {

            float rot_rads = pcl::deg2rad (static_cast<float> (angle_incr_ * v));
            Eigen::Affine3f incr_rot_trans = Eigen::Affine3f (Eigen::AngleAxisf (static_cast<float> (rot_rads), Eigen::Vector3f::UnitZ ()));

            //edges_ are just indices to the clouds, so do not rotate
            pcl::transformPointCloud (*cloud_for_iv_original, *cloud_for_iv, incr_rot_trans);
            pcl::transformPointCloud (*occluded_cloud_transformed_back_original, *occluded_cloud_transformed_back, incr_rot_trans);

            if (smooth_labels_cloud_)
              pcl::transformPointCloud (*smooth_labels_cloud_, *cloud_smooth, incr_rot_trans);
          }
          else
          {
            occluded_cloud_transformed_back.reset(new pcl::PointCloud<PointInT> (*occluded_cloud_transformed_back_original));
            cloud_for_iv.reset (new pcl::PointCloud<PointInT> (*cloud_for_iv_original));
            if (smooth_labels_cloud_)
              cloud_smooth.reset (new pcl::PointCloud<pcl::PointXYZL> (*smooth_labels_cloud_));
          }

          int * full_data = new int[GRIDSIZE_X * GRIDSIZE_Y * GRIDSIZE_Z];
          int * label_data = new int[GRIDSIZE_X * GRIDSIZE_Y * GRIDSIZE_Z];
          int * occupancy = new int[GRIDSIZE_X * GRIDSIZE_Y * GRIDSIZE_Z];
          int * occupancy_complete = new int[GRIDSIZE_X * GRIDSIZE_Y * GRIDSIZE_Z];
          int * occluded = new int[GRIDSIZE_X * GRIDSIZE_Y * GRIDSIZE_Z];

          int size_z_proj = GRIDSIZE_Z;
          int * occupancy_projected = new int[GRIDSIZE_X * GRIDSIZE_Y * size_z_proj];
          int * occluded_projected = new int[GRIDSIZE_X * GRIDSIZE_Y * size_z_proj];

          for (int i = 0; i < (GRIDSIZE_X * GRIDSIZE_Y * GRIDSIZE_Z); i++)
          {
            label_data[i] = 0;
            occupancy[i] = 0;
            occupancy_complete[i] = 0;
            occluded[i] = 0;
            full_data[i] = 1;
          }

          for (int i = 0; i < (GRIDSIZE_X * GRIDSIZE_Y * size_z_proj); i++)
          {
            occupancy_projected[i] = 0;
            occluded_projected[i] = 0;
          }

          for (size_t i = 0; i < occluded_cloud_transformed_back->points.size (); i++)
          {
            Eigen::Vector3f p = occluded_cloud_transformed_back->points[i].getVector3fMap ();
            if (!pcl::isFinite (occluded_cloud_transformed_back->points[i]))
              continue;

            if (p[0] > max_x || p[1] > max_y || p[2] > max_z)
              continue;
            if (p[0] < min_x || p[1] < min_y || p[2] < min_z)
              continue;

            int xx = std::floor (((p[0] - min_x) / (max_x - min_x)) * GRIDSIZE_X);
            int yy = std::floor (((p[1] - min_y) / (max_y - min_y)) * GRIDSIZE_Y);
            int zz = std::floor (((p[2] - min_z) / (max_z - min_z)) * GRIDSIZE_Z);
            int idx = zz * GRIDSIZE_X * GRIDSIZE_Y + yy * GRIDSIZE_X + xx;
            assert (idx < (GRIDSIZE_X * GRIDSIZE_Y * GRIDSIZE_Z));
            assert (idx >= 0);
            occluded[idx] = 1;

            if (p[2] > 0.01f)
            {
              //zz = 1;
              //occluded_projected[zz * GRIDSIZE_X * GRIDSIZE_Y + yy * GRIDSIZE_X + xx] = 1;
              for (int z = zz; z < size_z_proj; z++)
                occluded_projected[z * GRIDSIZE_X * GRIDSIZE_Y + yy * GRIDSIZE_X + xx] = 1;
            }
          }

          for (size_t i = 0; i < edges_.size (); i++)
          {
            //check that the edge point is inside of the VOI
            Eigen::Vector3f p = cloud_for_iv->points[edges_[i]].getVector3fMap ();
            if (p[0] > max_x || p[1] > max_y || p[2] > max_z)
              continue;
            if (p[0] < min_x || p[1] < min_y || p[2] < min_z)
              continue;

            int xx = std::floor (((p[0] - min_x) / (max_x - min_x)) * GRIDSIZE_X);
            int yy = std::floor (((p[1] - min_y) / (max_y - min_y)) * GRIDSIZE_Y);
            int zz = std::floor (((p[2] - min_z) / (max_z - min_z)) * GRIDSIZE_Z);

            int idx = zz * GRIDSIZE_X * GRIDSIZE_Y + yy * GRIDSIZE_X + xx;
            assert (idx < (GRIDSIZE_X * GRIDSIZE_Y * GRIDSIZE_Z));
            if (idx < 0)
              continue;

            label_data[idx] = 1;
          }

          if (v != 0)
          {
            //visualizeIVData (occluded, GRIDSIZE_X, GRIDSIZE_Y, GRIDSIZE_Z);
            //dilateIVData (occluded, GRIDSIZE_X, GRIDSIZE_Y, GRIDSIZE_Z, 1, 1);
            //erodeIVData (occluded, GRIDSIZE_X, GRIDSIZE_Y, GRIDSIZE_Z, 1, 1);
            //visualizeIVData (occluded, GRIDSIZE_X, GRIDSIZE_Y, GRIDSIZE_Z);
          }

          //fill occupancy
          for (size_t i = 0; i < cloud_for_iv->points.size (); i++)
          {
            //check that the edge point is inside of the VOI
            if (!pcl::isFinite (cloud_for_iv->points[i]))
              continue;

            Eigen::Vector3f p = cloud_for_iv->points[i].getVector3fMap ();
            if (p[0] > max_x || p[1] > max_y || p[2] > max_z)
              continue;
            if (p[0] < min_x || p[1] < min_y || p[2] < (min_z + 0.0f))
              continue;

            int xx = std::floor (((p[0] - min_x) / (max_x - min_x)) * GRIDSIZE_X);
            int yy = std::floor (((p[1] - min_y) / (max_y - min_y)) * GRIDSIZE_Y);
            int zz = std::floor (((p[2] - min_z) / (max_z - min_z)) * GRIDSIZE_Z);
            int idx = zz * GRIDSIZE_X * GRIDSIZE_Y + yy * GRIDSIZE_X + xx;
            assert (idx < (GRIDSIZE_X * GRIDSIZE_Y * GRIDSIZE_Z));
            assert (idx >= 0);
            occupancy[idx] = 1;
            occluded[idx] = 0;

            if (p[2] > 0.01f)
            {

              for (int z = zz; z < size_z_proj; z++)
              {
                occluded_projected[z * GRIDSIZE_X * GRIDSIZE_Y + yy * GRIDSIZE_X + xx] = 0;
                occupancy_projected[z * GRIDSIZE_X * GRIDSIZE_Y + yy * GRIDSIZE_X + xx] = 1;
              }
            }

          }

          rivs_full[v].reset (new IntegralVolume (full_data, GRIDSIZE_X, GRIDSIZE_Y, GRIDSIZE_Z));
          rivs[v].reset (new IntegralVolume (label_data, GRIDSIZE_X, GRIDSIZE_Y, GRIDSIZE_Z));
          rivs_occupancy[v].reset (new IntegralVolume (occupancy, GRIDSIZE_X, GRIDSIZE_Y, GRIDSIZE_Z));
          rivs_occluded[v].reset (new IntegralVolume (occluded, GRIDSIZE_X, GRIDSIZE_Y, GRIDSIZE_Z));
          projected_occupancy_[v].reset (new IntegralVolume (occupancy_projected, GRIDSIZE_X, GRIDSIZE_Y, size_z_proj));
          projected_occluded_[v].reset (new IntegralVolume (occluded_projected, GRIDSIZE_X, GRIDSIZE_Y, size_z_proj));

          if (smooth_labels_cloud_)
          {
            for (size_t i = 0; i < cloud_smooth->points.size (); i++)
            {
              Eigen::Vector3f p = cloud_smooth->points[i].getVector3fMap ();
              if (p[0] > max_x || p[1] > max_y || p[2] > max_z)
                continue;
              if (p[0] < min_x || p[1] < min_y || p[2] < min_z)
                continue;

              int xx = std::floor (((p[0] - min_x) / (max_x - min_x)) * GRIDSIZE_X);
              int yy = std::floor (((p[1] - min_y) / (max_y - min_y)) * GRIDSIZE_Y);
              int zz = std::floor (((p[2] - min_z) / (max_z - min_z)) * GRIDSIZE_Z);
              int idx = zz * GRIDSIZE_X * GRIDSIZE_Y + yy * GRIDSIZE_X + xx;
              occupancy_complete[idx]++;
            }

            rivs_occupancy_complete_[v].reset (new IntegralVolume (occupancy_complete, GRIDSIZE_X, GRIDSIZE_Y, GRIDSIZE_Z));

            npoints_label_[v].resize (max_label_ + 1, 0);

            int * label_data_smooth = new int[GRIDSIZE_X * GRIDSIZE_Y * GRIDSIZE_Z];
            for (int b = 0; b < (max_label_ + 1); b++)
            {
              for (int i = 0; i < (GRIDSIZE_X * GRIDSIZE_Y * GRIDSIZE_Z); i++)
                label_data_smooth[i] = 0;

              int n_points = 0;
              for (size_t i = 0; i < cloud_smooth->points.size (); i++)
              {
                if (static_cast<int> (cloud_smooth->points[i].label) != b)
                  continue;

                Eigen::Vector3f p = cloud_smooth->points[i].getVector3fMap ();
                if (p[0] > max_x || p[1] > max_y || p[2] > max_z)
                  continue;
                if (p[0] < min_x || p[1] < min_y || p[2] < min_z)
                  continue;

                int xx = std::floor (((p[0] - min_x) / (max_x - min_x)) * GRIDSIZE_X);
                int yy = std::floor (((p[1] - min_y) / (max_y - min_y)) * GRIDSIZE_Y);
                int zz = std::floor (((p[2] - min_z) / (max_z - min_z)) * GRIDSIZE_Z);
                int idx = zz * GRIDSIZE_X * GRIDSIZE_Y + yy * GRIDSIZE_X + xx;
                if(label_data_smooth[idx] == 0)
                  n_points++;

                label_data_smooth[idx] = 1; //++;
              }

              npoints_label_[v][b] = n_points;
              table_plane_label_ = 0;
              int max_points = -1;
              for (size_t i = 1; i < npoints_label_[v].size (); i++)
              {
                if (npoints_label_[v][i] > max_points)
                {
                  max_points = npoints_label_[v][i];
                  table_plane_label_ = static_cast<unsigned int> (i);
                }
              }

              //std::cout << "N points:" << n_points << " " << b << std::endl;
              //visualizeIVData(label_data, GRIDSIZE_X, GRIDSIZE_Y, GRIDSIZE_Z);
              rivhistograms[v][b].reset (new IntegralVolume (label_data_smooth, GRIDSIZE_X, GRIDSIZE_Y, GRIDSIZE_Z));
            }

            delete[] label_data_smooth;
          }

          int * color_points_data = new int[GRIDSIZE_X * GRIDSIZE_Y * GRIDSIZE_Z];
          std::vector<int *> color_data_quantized;
          color_data_quantized.resize(ycolor_size_ * vcolor_size_ * ucolor_size_);
          for(size_t b = 0; b < ycolor_size_ * vcolor_size_ * ucolor_size_; b++) {
            color_data_quantized[b] = new int[GRIDSIZE_X * GRIDSIZE_Y * GRIDSIZE_Z];
            for(size_t jj=0; jj < GRIDSIZE_X * GRIDSIZE_Y * GRIDSIZE_Z; jj++) {
              color_data_quantized[b][jj] = 0;
            }
          }

          //for(size_t b=0; b < 3; b++) {
          //int * color_data = new int[GRIDSIZE_X * GRIDSIZE_Y * GRIDSIZE_Z];
          //int * color_data_sq = new int[GRIDSIZE_X * GRIDSIZE_Y * GRIDSIZE_Z];
          std::vector<int> points_per_hist;
          points_per_hist.resize(ycolor_size_ * ucolor_size_ * vcolor_size_);
          for(size_t i=0; i < points_per_hist.size(); i++)
            points_per_hist[i] = 0;

          for (size_t i = 0; i < cloud_for_iv->points.size (); i++) {
            if (!pcl::isFinite (cloud_for_iv->points[i]))
              continue;

            Eigen::Vector3f p = cloud_for_iv->points[i].getVector3fMap ();
            if (p[0] > max_x || p[1] > max_y || p[2] > max_z)
              continue;
            if (p[0] < min_x || p[1] < min_y || p[2] < (min_z + 0.0f))
              continue;

            if (p[2] < (min_z + start_z_ * resolution + 0.01f))
              continue;

            int xx = std::floor (((p[0] - min_x) / (max_x - min_x)) * GRIDSIZE_X);
            int yy = std::floor (((p[1] - min_y) / (max_y - min_y)) * GRIDSIZE_Y);
            int zz = std::floor (((p[2] - min_z) / (max_z - min_z)) * GRIDSIZE_Z);
            int idx = zz * GRIDSIZE_X * GRIDSIZE_Y + yy * GRIDSIZE_X + xx;
            assert (idx < (GRIDSIZE_X * GRIDSIZE_Y * GRIDSIZE_Z));
            assert (idx >= 0);

            //compute yuv position
            uint32_t rgb = *reinterpret_cast<int*> (&cloud_for_iv->points[i].rgb);
            uint8_t rm = (rgb >> 16) & 0x0000ff;
            uint8_t gm = (rgb >> 8) & 0x0000ff;
            uint8_t bm = (rgb) & 0x0000ff;

            float ym = 0.257f * rm + 0.504f * gm + 0.098f * bm + 16; //between 16 and 235
            float um = -(0.148f * rm) - (0.291f * gm) + (0.439f * bm) + 128;
            float vm = (0.439f * rm) - (0.368f * gm) - (0.071f * bm) + 128;

            int idx_y = (ym - 16) / (220 / ycolor_size_);
            int idx_u = (um) / (256 / ucolor_size_);
            int idx_v = (vm) / (256 / vcolor_size_);

            int idx_yuv = idx_y * (ucolor_size_ * vcolor_size_) + idx_u * (vcolor_size_) + idx_v;
            color_data_quantized[idx_yuv][idx] += 1;
            color_points_data[idx] += 1;

            points_per_hist[idx_yuv]++;
            /*if(b == 0) {
              color_data[idx] += cloud_for_iv->points[i].r;
              color_data_sq[idx] += cloud_for_iv->points[i].r * cloud_for_iv->points[i].r;
              color_points_data[idx] += 1;
            } else if ( b==1) {
              color_data[idx] += cloud_for_iv->points[i].g;
              color_data_sq[idx] += cloud_for_iv->points[i].g * cloud_for_iv->points[i].g;
            } else {
              color_data[idx] += cloud_for_iv->points[i].b;
              color_data_sq[idx] += cloud_for_iv->points[i].b * cloud_for_iv->points[i].b;
            }*/
          }

          /*riv_color_histograms_[v][b].reset (new IntegralVolume (color_data, GRIDSIZE_X, GRIDSIZE_Y, GRIDSIZE_Z));
          riv_squared_color_histograms_[v][b].reset (new IntegralVolume (color_data_sq, GRIDSIZE_X, GRIDSIZE_Y, GRIDSIZE_Z));
          delete[] color_data;
          delete[] color_data_sq;
        }*/

          for(size_t b = 0; b < ycolor_size_ * vcolor_size_ * ucolor_size_; b++) {
            /*if(points_per_hist[b] > 0) {
              std::cout << points_per_hist[b] << std::endl;
              visualizeIVData(color_data_quantized[b], GRIDSIZE_X, GRIDSIZE_Y, GRIDSIZE_Z);
            }*/
            riv_color_histograms_[v][b].reset (new IntegralVolume (color_data_quantized[b], GRIDSIZE_X, GRIDSIZE_Y, GRIDSIZE_Z));
          }

          riv_points_color_histogram_[v].reset(new IntegralVolume(color_points_data, GRIDSIZE_X, GRIDSIZE_Y, GRIDSIZE_Z));
          delete[] color_points_data;
          for(size_t b = 0; b < ycolor_size_ * vcolor_size_ * ucolor_size_; b++) {
            delete[] color_data_quantized[b];
          }

          //visualizeIVData(label_data, GRIDSIZE_X, GRIDSIZE_Y, GRIDSIZE_Z);
          //visualizeIVData(occupancy, GRIDSIZE_X, GRIDSIZE_Y, GRIDSIZE_Z);
          //visualizeIVData (occluded, GRIDSIZE_X, GRIDSIZE_Y, GRIDSIZE_Z);
          //visualizeIVData (occupancy_projected, GRIDSIZE_X, GRIDSIZE_Y, size_z_proj);
          //visualizeIVData (occluded_projected, GRIDSIZE_X, GRIDSIZE_Y, size_z_proj);

          delete[] label_data;
          delete[] occupancy;
          delete[] occluded;
          delete[] full_data;
          delete[] occluded_projected;
          delete[] occupancy_projected;
        }
      }

    template<typename PointInT>
    void
    Objectness3D<PointInT>::generateBoundingBoxesExhaustiveSearch (std::vector<BBox> & bounding_boxes, int zs, int z_limit)
    {

    }

    template<typename PointInT>
    void
    Objectness3D<PointInT>::generateBoundingBoxes (std::vector<BBox> & bounding_boxes)
    {

    }

    template<typename PointInT>
      void
      Objectness3D<PointInT>::evaluateBoundingBoxes (std::vector<BBox> & bounding_boxes, bool print_objectness)
      {
        //int valid = 0;
        std::vector<bool> valid_boxes;
        valid_boxes.resize (bounding_boxes.size (), false);
        int GRIDSIZE_X = (int)((max_x - min_x) / resolution);
        int GRIDSIZE_Y = (int)((max_y - min_y) / resolution);
        int GRIDSIZE_Z = (int)((max_z - min_z) / resolution);

#pragma omp parallel for schedule(static, 4) num_threads(omp_get_num_procs())
        for (int i = 0; i < bounding_boxes.size (); i++)
        {
          BBox * bb = &(bounding_boxes[i]);
          BBox bb_shrinked;
          shrink_bbox (bb, bb_shrinked);

          int outer_edges;
          rivs[bb->angle]->getRectangleFromCorner (bb->x, bb->y, bb->z, bb->sx, bb->sy, bb->sz, outer_edges);

          if (outer_edges <= 20)
            continue;

          int inner_edges;
          /*rivs[bb->angle]->getRectangleFromCorner (bb_shrinked.x, bb_shrinked.y, bb_shrinked.z, bb_shrinked.sx, bb_shrinked.sy, bb_shrinked.sz,
           inner_edges);*/

          int occupancy_val;
          int occupancy_val_inner;

          rivs_occupancy[bb->angle]->getRectangleFromCorner (bb->x, bb->y, bb->z, bb->sx, bb->sy, bb->sz, occupancy_val);
          rivs_occupancy[bb->angle]->getRectangleFromCorner (bb_shrinked.x, bb_shrinked.y, bb_shrinked.z, bb_shrinked.sx, bb_shrinked.sy,
                                                             bb_shrinked.sz, occupancy_val_inner);

          if (print_objectness)
          {
            std::cout << "Occupancy val:" << occupancy_val << std::endl;
          }

          if (occupancy_val == 0)
            continue;

          int occluded_val;
          int occluded_val_inner;

          rivs_occluded[bb->angle]->getRectangleFromCorner (bb->x, bb->y, bb->z, bb->sx, bb->sy, bb->sz, occluded_val);
          rivs_occluded[bb->angle]->getRectangleFromCorner (bb_shrinked.x, bb_shrinked.y, bb_shrinked.z, bb_shrinked.sx, bb_shrinked.sy,
                                                            bb_shrinked.sz, occluded_val_inner);

          if ((bb->sx * bb->sy * bb->sz - occluded_val) == 0) //all is occluded!
            continue;

          float out_perimeter = 4.f * (bb->sx + bb->sy + bb->sz) / 2.f;
          float in_perimeter = 4.f * (bb_shrinked.sx + bb_shrinked.sy + bb_shrinked.sz) / 2.f;

          /*int occupancy_val_proj;
          int occluded_val_proj;

          projected_occluded_[bb->angle]->getRectangleFromCorner (bb->x, bb->y, bb->sz, bb->sx, bb->sy, 1, occluded_val_proj);
          projected_occupancy_[bb->angle]->getRectangleFromCorner (bb->x, bb->y, bb->sz, bb->sx, bb->sy, 1, occupancy_val_proj);

          float free_space2 = (bb->sx * bb->sy - occluded_val_proj - occupancy_val_proj) * bb->sz;*/
          float free_space = bb->sx * bb->sy * bb->sz - occluded_val - occupancy_val;
          float free_space_inner = bb_shrinked.sx * bb_shrinked.sy * bb_shrinked.sz - occluded_val_inner - occupancy_val_inner;

          int faces_area = ((bb->sx * bb->sy) + (bb->sx * bb->sz) + (bb->sy * bb->sz)) * 2.f;
          int occluded_faces = getValueByFaces (bb, rivs_occluded[bb->angle]);

          if (print_objectness)
          {
            //std::cout << "sum_edges after occ:" << sum_edges << std::endl;
            std::cout << "occluded faces:" << faces_area << " " << occluded_faces << std::endl;
          }

          //assert(sum_edges > 0);
          in_perimeter = 0;
          inner_edges = 0;

          float vol_outer = static_cast<float> (bb->sx * bb->sy * bb->sz);
          float vol_shrinked = static_cast<float> (bb_shrinked.sx * bb_shrinked.sy * bb_shrinked.sz);

          //////////////////////////////////EXPANDED BBOX/////////////////////////////////////
          BBox bb_extended;
          bb_extended.sx = static_cast<int> (pcl_round (bb->sx * expand_factor_));
          bb_extended.sy = static_cast<int> (pcl_round (bb->sy * expand_factor_));
          bb_extended.sz = static_cast<int> (pcl_round (bb->sz * expand_factor_));

          bb_extended.x = bb->x - static_cast<int> (pcl_round ((bb_extended.sx - bb->sx) / 2.f));
          bb_extended.y = bb->y - static_cast<int> (pcl_round ((bb_extended.sy - bb->sy) / 2.f));
          bb_extended.z = bb->z - static_cast<int> (pcl_round ((bb_extended.sz - bb->sz) / 2.f));

          bb_extended.x = std::max (bb_extended.x, 1);
          bb_extended.y = std::max (bb_extended.y, 1);
          bb_extended.z = std::max (bb_extended.z, 1);

          bb_extended.sx = std::min (GRIDSIZE_X - 1, bb_extended.x + bb_extended.sx) - bb_extended.x;
          bb_extended.sy = std::min (GRIDSIZE_Y - 1, bb_extended.y + bb_extended.sy) - bb_extended.y;
          bb_extended.sz = std::min (GRIDSIZE_Z - 1, bb_extended.z + bb_extended.sz) - bb_extended.z;

          int expanded_edges;
          rivs[bb->angle]->getRectangleFromCorner (bb_extended.x, bb_extended.y, bb_extended.z, bb_extended.sx, bb_extended.sy, bb_extended.sz,
                                                   expanded_edges);
          //////////////////////////////////EXPANDED BBOX/////////////////////////////////////

          //float edges_score = outer_edges / ( (vol_outer - occluded_val) - (vol_shrinked - occluded_val_inner));
          //float edges_score = (outer_edges - inner_edges) / (sum_edges - in_perimeter);
          float edges_score = outer_edges / static_cast<float> (faces_area - occluded_faces);
          //float free_space2_score = (vol_outer - free_space2) / vol_outer;
          float occ_inner_score = (vol_shrinked - occluded_val_inner) / (vol_shrinked - occluded_val_inner + occupancy_val_inner);
          float free_space_score = (vol_outer - free_space) / vol_outer;
          float free_space_inner_score = (vol_shrinked - free_space_inner) / vol_shrinked;
          //float face_edges_score = 1.f - (sum_face_edges / static_cast<float> (bb->sx * bb->sy * 2 + bb->sx * bb->sz * 2 + bb->sz * bb->sy * 2));

          float clutter_score = 0.f;
          if (smooth_labels_cloud_)
          {
            /*std::vector<int> labels_inside, labels_outside;
             labels_inside.resize (max_label_ + 1, 0);
             labels_outside.resize (max_label_ + 1, 0);*/

            int occupancy_val_complete;
            rivs_occupancy_complete_[bb->angle]->getRectangleFromCorner (bb->x, bb->y, bb->z, bb->sx, bb->sy, bb->sz, occupancy_val_complete);

            int num_points_inside = 0;
            int v = bb->angle;
            int above_zero = 0;
            for (int j = 1; j < (max_label_ + 1); j++) //NOT ignoring label zero
            {
              //if (j == table_plane_label_) //ignore table plane label
              //continue;

              int val;
              rivhistograms[v][j]->getRectangleFromCorner (bb->x, bb->y, bb->z, bb->sx, bb->sy, bb->sz, val);
              //labels_inside[j] = val;
              num_points_inside += val;

              if (val > 0)
                above_zero++;

              //clutter_score += (labels_inside[j] / static_cast<float> (npoints_label_[j])) * labels_inside[j];
              clutter_score += (val / static_cast<float> (npoints_label_[v][j])) * val;
            }

            if (num_points_inside <= 0)
            {
              bb->score = 0;
              continue;
            }
            else
            {
              clutter_score /= static_cast<float> (num_points_inside);
              //bb->score += above_zero / static_cast<float>(labels_inside.size () - 1);
            }

            if (print_objectness)
            {
              std::cout << "***********************" << std::endl;
              std::cout << "volume:" << vol_outer << " sides:" << bb->sx << "," << bb->sy << "," << bb->sz << std::endl;
              //std::cout << "edges:" << (outer_edges - inner_edges) / (out_perimeter - in_perimeter) << std::endl;
              std::cout << "new edges:" << edges_score << std::endl;
              //std::cout << "free space 2:" << free_space2_score << std::endl;
              std::cout << "occupancy inner:" << occ_inner_score << std::endl;
              std::cout << "free space:" << free_space_score << std::endl;
              std::cout << "free space inner:" << free_space_inner_score << std::endl;
              std::cout << "clutter:" << clutter_score << " " << std::endl;
              //std::cout << "face edges score:" << face_edges_score << std::endl;
              std::cout << "num_points_inside:" << num_points_inside << std::endl;
              std::cout << "edges:" << outer_edges << " " << expanded_edges << std::endl;
              std::cout << "***********************" << std::endl << std::endl;
            }

            //if (pcl_isnan(bb->score))
            //bb->score = 0.f;
          }

          //bb->score = outer_edges;
          bb->score = (static_cast<float> (outer_edges) / static_cast<float> (expanded_edges)) * edges_score;
          bb->score += std::pow (clutter_score, 2);
          //bb->score = 2.5f * edges_score;
          bb->score = (static_cast<float> (outer_edges) / static_cast<float> (expanded_edges)) * edges_score;
          /*if(clutter_score != 0) {
           bb->score *= clutter_score;
           }*/

          //bb->score += 0.25f * free_space2_score;
          //bb->score += 0.25f * occ_inner_score;
          //bb->score += 0.5f * (free_space_inner_score * free_space_score);
          //bb->score += 1.f * static_cast<float> (outer_edges) / static_cast<float> (expanded_edges);
          bb->score += 3.f * clutter_score;
          //bb->score = 0.5f * edges_score * free_space2_score * free_space_score;
          //bb->score += 0.25f * free_space_score;
          //bb->score += 0.1 * face_edges_score;

          valid_boxes[i] = true;
        }

        if (!print_objectness)
        {
          int valid = 0;
          for (size_t i = 0; i < bounding_boxes.size (); i++)
          {
            if (valid_boxes[i])
            {
              bounding_boxes[valid] = bounding_boxes[i];
              valid++;
            }
          }
          bounding_boxes.resize (valid);
          std::cout << "Number of valid bounding boxes:" << valid << std::endl;
        }
      }

    template<typename PointInT>
    void Objectness3D<PointInT>::printSomeObjectnessValues(BBox & box) {
      int faces_area = ((box.sx * box.sy) + (box.sx * box.sz) + (box.sy * box.sz)) * 2.f;
      int vol_faces[6];
      vol_faces[0] = vol_faces[5] = box.sy * box.sz;
      vol_faces[1] = vol_faces[4] = box.sx * box.sz;
      vol_faces[2] = vol_faces[3] = box.sy * box.sx;

      bool visible_faces[6];
      for(int j=0; j < 6; j++)
        visible_faces[j] = true;

      //compute oriented normals for faces and check the dot product with viewpoint
      Eigen::Vector4f min_p4f, max_p4f;
      min_p4f[0] = min_x + box.x * resolution;
      min_p4f[1] = min_y + box.y * resolution;
      min_p4f[2] = min_z + box.z * resolution;
      min_p4f[3] = 0;
      max_p4f[0] = min_x + (box.sx + box.x) * resolution;
      max_p4f[1] = min_y + (box.sy + box.y) * resolution;
      max_p4f[2] = min_z + (box.sz + box.z) * resolution;
      max_p4f[3] = 0;

      Eigen::Vector3f vertices[8];
      vertices[0] = Eigen::Vector3f(min_p4f[0], min_p4f[1], min_p4f[2]);
      vertices[1] = Eigen::Vector3f(min_p4f[0], max_p4f[1], min_p4f[2]);
      vertices[2] = Eigen::Vector3f(min_p4f[0], min_p4f[1], max_p4f[2]);
      vertices[3] = Eigen::Vector3f(min_p4f[0], max_p4f[1], max_p4f[2]);
      vertices[4] = Eigen::Vector3f(max_p4f[0], min_p4f[1], min_p4f[2]);
      vertices[5] = Eigen::Vector3f(max_p4f[0], max_p4f[1], min_p4f[2]);
      vertices[6] = Eigen::Vector3f(max_p4f[0], min_p4f[1], max_p4f[2]);
      vertices[7] = Eigen::Vector3f(max_p4f[0], max_p4f[1], max_p4f[2]);

      //compute normals and orient them properly
      Eigen::Vector3f normals[6];
      Eigen::Vector3f faces_centroid[6];
      faces_centroid[0] = (vertices[0] + vertices[3]) / 2.f;
      faces_centroid[1] = (vertices[0] + vertices[6]) / 2.f;
      faces_centroid[2] = (vertices[0] + vertices[5]) / 2.f;
      faces_centroid[3] = (vertices[2] + vertices[7]) / 2.f;
      faces_centroid[4] = (vertices[7] + vertices[1]) / 2.f;
      faces_centroid[5] = (vertices[7] + vertices[4]) / 2.f;

      Eigen::Vector3f center = BBoxCenter(box);

      if (box.angle != 0)
      {
        float rot_angle = pcl::deg2rad (static_cast<float> (box.angle * angle_incr_ * -1.f));
        Eigen::Affine3f rot_trans = Eigen::Affine3f (Eigen::AngleAxisf (static_cast<float> (rot_angle), Eigen::Vector3f::UnitZ ()));

        min_p4f = rot_trans * min_p4f;
        max_p4f = rot_trans * max_p4f;

        for(size_t i = 0; i < 6; i++) {
          faces_centroid[i] = rot_trans * faces_centroid[i];
        }

        for(size_t i = 0; i < 8; i++) {
          vertices[i] = rot_trans * vertices[i];
       }
      }

      for(int j=0; j < 6; j++) {
        normals[j] = faces_centroid[j] - center;
        normals[j].normalize();
      }

      //check that normals are consistenly oriented...
      Eigen::Vector3f vp(vpx_, vpy_, vpz_);
      int sum_v = 0;
      faces_area = 0;
      for(int j=0; j < 6; j++) {
        Eigen::Vector3f c_vp = (vp - faces_centroid[j]);
        c_vp.normalize();
        float dot_p = normals[j].dot(c_vp);
        //std::cout << "dot product:" << dot_p << std::endl;
        visible_faces[j] = dot_p > 0.02f;
        if(visible_faces[j]) {
          sum_v++;
          faces_area += vol_faces[j];
        }
      }

      int occluded_faces = getValueByFaces (&box, rivs_occluded[box.angle], visible_faces);

      int occupancy_val;
      rivs_occupancy[box.angle]->getRectangleFromCorner (box.x, box.y, box.z, box.sx, box.sy, box.sz, occupancy_val);

      int occluded_val;
      rivs_occluded[box.angle]->getRectangleFromCorner (box.x, box.y, box.z, box.sx, box.sy, box.sz, occluded_val);
      float vol_outer = static_cast<float> (box.sx * box.sy * box.sz);
      /*float free_space = vol_outer - occluded_val - occupancy_val;
      float free_space_score = (vol_outer - free_space) / vol_outer;*/

      std::cout << "**************** ONE BOX ********************" << std::endl;
      std::cout << vol_faces[0] << " " << vol_faces[1] << " " << vol_faces[2] << std::endl;
      std::cout << box.sx << " " << box.sy << " " << box.sz << std::endl;
      std::cout << "Number of visible faces:" << sum_v << " " << faces_area << " " << occluded_faces << " visible area from faces:" << faces_area - occluded_faces << std::endl;
      std::cout << "occupancy:" << occupancy_val << " occuded_val:" << occluded_val << " " << vol_outer << std::endl;
      std::cout << "angle:" << box.angle * angle_incr_ << std::endl;
      int outer_edges;
      rivs[box.angle]->getRectangleFromCorner (box.x, box.y, box.z, box.sx, box.sy, box.sz, outer_edges);

      int GRIDSIZE_X = (int)((max_x - min_x) / resolution);
      int GRIDSIZE_Y = (int)((max_y - min_y) / resolution);
      int GRIDSIZE_Z = (int)((max_z - min_z) / resolution);

      BBox bb_extended;
      bb_extended.sx = static_cast<int> (pcl_round (box.sx * expand_factor_));
      bb_extended.sy = static_cast<int> (pcl_round (box.sy * expand_factor_));
      bb_extended.sz = static_cast<int> (pcl_round (box.sz * expand_factor_));

      bb_extended.x = box.x - static_cast<int> (pcl_round ((bb_extended.sx - box.sx) / 2.f));
      bb_extended.y = box.y - static_cast<int> (pcl_round ((bb_extended.sy - box.sy) / 2.f));
      bb_extended.z = box.z - static_cast<int> (pcl_round ((bb_extended.sz - box.sz) / 2.f));

      bb_extended.x = std::max (bb_extended.x, 1);
      bb_extended.y = std::max (bb_extended.y, 1);
      bb_extended.z = std::max (bb_extended.z, 1);

      bb_extended.sx = std::min (GRIDSIZE_X - 1, bb_extended.x + bb_extended.sx) - bb_extended.x;
      bb_extended.sy = std::min (GRIDSIZE_Y - 1, bb_extended.y + bb_extended.sy) - bb_extended.y;
      bb_extended.sz = std::min (GRIDSIZE_Z - 1, bb_extended.z + bb_extended.sz) - bb_extended.z;
      bb_extended.angle = box.angle;

      int expanded_edges;
      rivs[bb_extended.angle]->getRectangleFromCorner (bb_extended.x, bb_extended.y, bb_extended.z,
                                                       bb_extended.sx, bb_extended.sy, bb_extended.sz, expanded_edges);

      float edges_score = (float)outer_edges / (float)( (faces_area - occluded_faces));

      BBox bb_shrinked;
      bb_shrinked.sx = std::max(std::min (std::max (static_cast<int> (floor (box.sx * shrink_factor_x)), 1), box.sx - 2),1);
      bb_shrinked.sy = std::max(std::min (std::max (static_cast<int> (floor (box.sy * shrink_factor_y)), 1), box.sy - 2),1);
      bb_shrinked.sz = std::max(std::min (std::max (static_cast<int> (floor (box.sz * shrink_factor_z)), 1), box.sz - 2),1);

      bb_shrinked.x = box.x + std::max (static_cast<int> (floor ((box.sx - bb_shrinked.sx) / 2.f)), 1);
      bb_shrinked.y = box.y + std::max (static_cast<int> (floor ((box.sy - bb_shrinked.sy) / 2.f)), 1);
      bb_shrinked.z = box.z + std::max (static_cast<int> (floor ((box.sz - bb_shrinked.sz) / 2.f)), 1);

      float vol_shrinked = static_cast<float> (bb_shrinked.sx * bb_shrinked.sy * bb_shrinked.sz);
      int occupancy_val_inner;
      rivs_occupancy[box.angle]->getRectangleFromCorner (bb_shrinked.x, bb_shrinked.y, bb_shrinked.z, bb_shrinked.sx, bb_shrinked.sy,
                                                         bb_shrinked.sz, occupancy_val_inner);

      int inner_edges;
      rivs[box.angle]->getRectangleFromCorner (bb_shrinked.x, bb_shrinked.y, bb_shrinked.z, bb_shrinked.sx, bb_shrinked.sy,
                                                           bb_shrinked.sz, inner_edges);

      std::cout << "Outer edges:" << outer_edges << " inner:" << inner_edges << " expanded:" << expanded_edges << std::endl;
      float score = ((float)outer_edges / (float)expanded_edges) * edges_score;
      std::cout << "Occupancy val inner:" << occupancy_val_inner << " vol shrinked:" << vol_shrinked << " " << vol_outer << std::endl;
      std::cout << score << std::endl;

      //clutter score
      int num_points_inside = 0;
      int v = box.angle;
      int above_zero = 0;
      float clutter_score = 0;
      for (int j = 1; j < (max_label_ + 1); j++) //NOT ignoring label zero
      {
        int val;
        rivhistograms[v][j]->getRectangleFromCorner (box.x, box.y, box.z, box.sx, box.sy, box.sz, val);
        //labels_inside[j] = val;
        if(val == 0)
          continue;

        num_points_inside += val;

        if (val > 0)
          above_zero++;

        /*float sc = (val / (float)npoints_label_[v][j]) * (val / (float)(occupancy_val));
        if((val / (float)npoints_label_[v][j]) < 0.95f)
         sc *= -1.f;

        clutter_score += sc;*/
        //clutter_score += (val / (float)npoints_label_[j]) * (val / (float)(occupancy_val));
        std::cout << val << " " << npoints_label_[v][j] << " " << occupancy_val << " " << j << std::endl;
        //clutter_score += (labels_inside[j] / static_cast<float> (npoints_label_[j])) * labels_inside[j];
        clutter_score += (val / static_cast<float> (npoints_label_[v][j])) * val;
      }

      clutter_score /= static_cast<float>(num_points_inside);
      std::cout << "Clutter score:" << clutter_score << " >zero:" << above_zero << std::endl;

      /*int npoints_color;
      riv_points_color_histogram_[v]->getRectangleFromCorner (box.x, box.y, box.z, box.sx, box.sy, box.sz, npoints_color);
      int colors[3];
      int colors_squared_sum[3];
      for(int j=0; j < 3; j++) {
        riv_color_histograms_[v][j]->getRectangleFromCorner (box.x, box.y, box.z, box.sx, box.sy, box.sz, colors[j]);
        colors[j] /= static_cast<float>(npoints_color);

        riv_squared_color_histograms_[v][j]->getRectangleFromCorner (box.x, box.y, box.z, box.sx, box.sy, box.sz, colors_squared_sum[j]);
        colors_squared_sum[j] /= static_cast<float>(npoints_color);
      }

      std::cout << "Color average: (" << colors[0] << "," << colors[1] << "," << colors[2] << ") " << npoints_color << std::endl;
      std::cout << "Color variances: (" << colors_squared_sum[0] - colors[0]*colors[0] << "," << colors_squared_sum[1] - colors[1]*colors[1] << "," << colors_squared_sum[2] - colors[2]*colors[2] << ")" << std::endl;*/

      /*pcl::visualization::PCLVisualizer vis("vis");
      //add spheres on the corners of the box
      pcl::PointXYZ p1, p2, p3, p4, p5, p6;
      p1.getVector3fMap () = faces_centroid[0];
      p2.getVector3fMap () = faces_centroid[1];
      p3.getVector3fMap () = faces_centroid[2];
      p4.getVector3fMap () = faces_centroid[3];
      p5.getVector3fMap () = faces_centroid[4];
      p6.getVector3fMap () = faces_centroid[5];

      vis.addSphere<pcl::PointXYZ> (p1, 0.01, 0, 255, 255, "sphere_1");
      vis.addSphere<pcl::PointXYZ> (p2, 0.01, 0, 255, 255, "sphere_2");
      vis.addSphere<pcl::PointXYZ> (p3, 0.01, 0, 255, 255, "sphere_3");
      vis.addSphere<pcl::PointXYZ> (p4, 0.01, 0, 255, 255, "sphere_4");
      vis.addSphere<pcl::PointXYZ> (p5, 0.01, 0, 255, 255, "sphere_5");
      vis.addSphere<pcl::PointXYZ> (p6, 0.01, 0, 255, 255, "sphere_6");

      std::stringstream box_name;
      box_name << "box";
      visBBox(vis, box, box_name);
      vis.addCoordinateSystem(0.1f);

      {
          pcl::PointXYZ p1, p2, p3, p4, p5, p6, p7, p8;
          p1.getVector3fMap () = vertices[0];
          p2.getVector3fMap () = vertices[1];
          p3.getVector3fMap () = vertices[2];
          p4.getVector3fMap () = vertices[3];
          p5.getVector3fMap () = vertices[4];
          p6.getVector3fMap () = vertices[5];
          p7.getVector3fMap () = vertices[6];
          p8.getVector3fMap () = vertices[7];

          vis.addSphere<pcl::PointXYZ> (p1, 0.01, 255, 255, 255, "sphere_11");
          vis.addSphere<pcl::PointXYZ> (p2, 0.01, 255, 255, 255, "sphere_22");
          vis.addSphere<pcl::PointXYZ> (p3, 0.01, 255, 255, 255, "sphere_33");
          vis.addSphere<pcl::PointXYZ> (p4, 0.01, 255, 255, 255, "sphere_44");
          vis.addSphere<pcl::PointXYZ> (p5, 0.01, 255, 255, 255, "sphere_55");
          vis.addSphere<pcl::PointXYZ> (p6, 0.01, 255, 255, 255, "sphere_66");
          vis.addSphere<pcl::PointXYZ> (p7, 0.01, 255, 255, 255, "sphere_77");
          vis.addSphere<pcl::PointXYZ> (p8, 0.01, 255, 255, 255, "sphere_88");
        }

      //show viewpoint too
      {
        pcl::PointXYZ p1;
        p1.getVector3fMap () = vp;
        vis.addSphere<pcl::PointXYZ> (p1, 0.01, 255, 0, 255, "viewpoint");
      }

      vis.spin ();*/
    }

    template<typename PointInT>
      void
      Objectness3D<PointInT>::computeObjectness (bool compute_voi)
      {

        PointInTPtr cloud_for_iv (new pcl::PointCloud<PointInT> ());

        if (table_plane_set_)
        {
          //transform point cloud to be on the plane
          transformToBeCenteredOnPlane (table_plane_, transform_to_plane);

          pcl::transformPointCloud (*input_cloud, *cloud_for_iv, transform_to_plane);

          int GRIDSIZE_X, GRIDSIZE_Y, GRIDSIZE_Z;

          if (compute_voi)
          {
            PointInT min_pt, max_pt;
            PointInTPtr cloud_no_plane (new pcl::PointCloud<PointInT> (*cloud_for_iv));

            float plane_threshold = 0.02f;
            for (size_t i = 0; i < cloud_no_plane->points.size (); i++)
            {
              //check that the edge point is inside of the VOI
              if (!pcl::isFinite (cloud_no_plane->points[i]))
                continue;

              Eigen::Vector3f p = cloud_no_plane->points[i].getVector3fMap ();
              if (p[2] < plane_threshold)
              {
                cloud_no_plane->points[i].x = std::numeric_limits<float>::quiet_NaN ();
                cloud_no_plane->points[i].y = std::numeric_limits<float>::quiet_NaN ();
                cloud_no_plane->points[i].z = std::numeric_limits<float>::quiet_NaN ();
              }
            }

            Eigen::Vector4f centroid_above_plane;
            pcl::compute3DCentroid (*cloud_no_plane, centroid_above_plane);
            centroid_above_plane *= -1.f;

            Eigen::Matrix4f center_cloud;
            center_cloud.setIdentity ();
            center_cloud (0, 3) = centroid_above_plane[0];
            center_cloud (1, 3) = centroid_above_plane[1];

            pcl::transformPointCloud (*cloud_no_plane, *cloud_no_plane, center_cloud);
            pcl::transformPointCloud (*cloud_for_iv, *cloud_for_iv, center_cloud);

            Eigen::Matrix4f tmp;
            tmp = center_cloud * transform_to_plane;
            transform_to_plane = tmp;

            //transform viewpoint
            Eigen::Vector4f vp = Eigen::Vector4f(vpx_, vpy_, vpz_, 1);
            vp = transform_to_plane * vp;
            vpx_ = vp[0]; vpy_ = vp[1]; vpz_ = vp[2];

            std::cout << "Transformed viewpoint:" << vp << std::endl;

            int num_ivs = (90 / angle_incr_); //number of integral volumes to compute
            PointInTPtr cloud_no_plane_rotated (new pcl::PointCloud<PointInT> (*cloud_no_plane));
            for (int v = 0; v < num_ivs; v++)
            {

              if (v != 0)
              {

                float rot_rads = pcl::deg2rad (static_cast<float> (angle_incr_ * v));
                Eigen::Affine3f incr_rot_trans = Eigen::Affine3f (Eigen::AngleAxisf (static_cast<float> (rot_rads), Eigen::Vector3f::UnitZ ()));
                pcl::transformPointCloud (*cloud_no_plane, *cloud_no_plane_rotated, incr_rot_trans);
              }

              pcl::getMinMax3D (*cloud_no_plane_rotated, min_pt, max_pt);
              if (min_x > (min_pt.x))
                min_x = min_pt.x;

              if (min_y > (min_pt.y))
                min_y = min_pt.y;

              if (max_x < (max_pt.x))
                max_x = max_pt.x;

              if (max_y < (max_pt.y))
                max_y = max_pt.y;
            }

            float fac = 1.f;
            min_x -= resolution * fac;
            min_y -= resolution * fac;
            max_x += resolution * fac;
            max_y += resolution * fac;
            max_z = max_pt.z + resolution * fac;

            std::cout << "max_z" << max_z << std::endl;
          }

          GRIDSIZE_X = (int)((max_x - min_x) / resolution);
          GRIDSIZE_Y = (int)((max_y - min_y) / resolution);
          GRIDSIZE_Z = (int)((max_z - min_z) / resolution);

          //fill occluded cloud
          PointInTPtr occluded_cloud (new pcl::PointCloud<PointInT>);
          occluded_cloud->width = (GRIDSIZE_X * GRIDSIZE_Y * GRIDSIZE_Z);
          occluded_cloud->height = 1;
          occluded_cloud->points.resize ((GRIDSIZE_X * GRIDSIZE_Y * GRIDSIZE_Z));

          if (smooth_labels_cloud_)
          {

            max_label_ = 0;
            for (size_t i = 0; i < smooth_labels_cloud_->points.size (); i++)
            {
              if (smooth_labels_cloud_->points[i].label > max_label_)
              {
                max_label_ = smooth_labels_cloud_->points[i].label;
              }
            }

            std::cout << "max label:" << max_label_ << std::endl;

            /*npoints_label_.resize (max_label_ + 1, 0);

            for (size_t i = 0; i < smooth_labels_cloud_->points.size (); i++)
            {
              npoints_label_[smooth_labels_cloud_->points[i].label]++;
            }

            table_plane_label_ = 0;
            int max_points = -1;
            for (size_t i = 1; i < npoints_label_.size (); i++)
            {
              if (npoints_label_[i] > max_points)
              {
                max_points = npoints_label_[i];
                table_plane_label_ = static_cast<unsigned int> (i);
              }
            }*/

            pcl::transformPointCloud (*smooth_labels_cloud_, *smooth_labels_cloud_, transform_to_plane);

            pcl::PassThrough<pcl::PointXYZL> pass_;
            pass_.setFilterLimits (min_z, max_z);
            pass_.setFilterFieldName ("z");
            pass_.setInputCloud (smooth_labels_cloud_);
            pass_.filter (*smooth_labels_cloud_);
            pass_.setFilterLimits (min_x, max_x);
            pass_.setFilterFieldName ("x");
            pass_.setInputCloud (smooth_labels_cloud_);
            pass_.filter (*smooth_labels_cloud_);
            pass_.setFilterLimits (min_y, max_y);
            pass_.setFilterFieldName ("y");
            pass_.setInputCloud (smooth_labels_cloud_);
            pass_.filter (*smooth_labels_cloud_);

            /*if (visualize_)
            {
              pcl::visualization::PCLVisualizer vis ("smooth");
              pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZL> handler (smooth_labels_cloud_, "label");
              vis.addPointCloud<pcl::PointXYZL> (smooth_labels_cloud_, handler, "edges");
              vis.addCoordinateSystem (0.3f);
              vis.spin ();
            }*/
          }

          for (int xx = 0; xx < GRIDSIZE_X; xx++)
          {
            for (int yy = 0; yy < GRIDSIZE_Y; yy++)
            {
              for (int zz = 0; zz < GRIDSIZE_Z; zz++)
              {
                Eigen::Vector3f vec;
                float x = min_x + xx * resolution + resolution / 2.f;
                float y = min_y + yy * resolution + resolution / 2.f;
                float z = min_z + zz * resolution + resolution / 2.f;
                int idx = zz * GRIDSIZE_X * GRIDSIZE_Y + yy * GRIDSIZE_X + xx;
                vec = Eigen::Vector3f (x, y, z);
                occluded_cloud->points[idx].getVector3fMap () = vec;
              }
            }
          }

          /*pcl::visualization::PCLVisualizer vis("smooth");
           pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (cloud_no_plane);
           vis.addPointCloud<pcl::PointXYZRGB> (cloud_no_plane, handler, "rgb");
           vis.spin();*/

          PointInTPtr occluded_cloud_transformed (new pcl::PointCloud<PointInT>);
          Eigen::Matrix4f back_to_range_map;
          back_to_range_map = transform_to_plane.inverse ();
          pcl::transformPointCloud (*occluded_cloud, *occluded_cloud_transformed, back_to_range_map);
          PointInTPtr filtered = pcl::occlusion_reasoning::getOccludedCloud<PointInT, PointInT> (input_cloud, occluded_cloud_transformed, 525.f,
                                                                                                 0.01f, false);

          /*if (visualize_)
          {
            pcl::visualization::PCLVisualizer vis ("occlusion");
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (input_cloud);
            vis.addPointCloud<PointInT> (input_cloud, handler, "input cloud");
            pcl::visualization::PointCloudColorHandlerCustom<PointInT> handler_filtered (filtered, 255, 0, 0);
            vis.addPointCloud<PointInT> (filtered, handler_filtered, "filtered");
            vis.spin ();
          }*/

          PointInTPtr occluded_cloud_transformed_back (new pcl::PointCloud<PointInT>);
          pcl::transformPointCloud (*filtered, *occluded_cloud_transformed_back, transform_to_plane);

          {
            pcl::ScopeTime t ("Constructing RIV");
            createRotatedIV (cloud_for_iv, occluded_cloud_transformed_back);
          }

          //start sampling volumes and evaluate the bounding boxes
          srand (time (NULL));
          std::vector<BBox> bounding_boxes;
          if (do_z_)
          {
            int z_limit = (GRIDSIZE_Z - min_size_w_ - 1);
            int step_z = 1;
            std::vector<BBox> boxes;
            for (size_t z = start_z_; z < z_limit; z += step_z)
            {
              generateBoundingBoxesExhaustiveSearch (boxes, z, z);
              evaluateBoundingBoxes (boxes);

              std::vector<BBox>::iterator max_bbox = std::max_element (boxes.begin (), boxes.end (), BBoxless);
              float max_score = max_bbox->score;
              int valid = 0;

              for (std::vector<BBox>::iterator it = boxes.begin (); it != boxes.end (); ++it)
              {
                //it->score /= max_score;
                if ((it->score /*/ max_score*/) > max_score_threshold_)
                {
                  boxes[valid] = *it;
                  valid++;
                }
              }

              boxes.resize (valid);

              if (boxes.size () > num_wins_)
                std::sort (boxes.begin (), boxes.end (), sortBBoxes);

              boxes.resize (std::min (num_wins_, static_cast<int> (boxes.size ())));
              bounding_boxes.insert (bounding_boxes.end (), boxes.begin (), boxes.end ());
            }
          }
          else
          {
            faat_pcl::cuda::segmentation::Objectness3DCuda o3d_cuda (angle_incr_, expand_factor_, max_label_, GRIDSIZE_X, GRIDSIZE_Y, GRIDSIZE_Z, resolution,
                                                                     min_size_w_, max_size_w_, min_x, max_x, min_y, max_y, min_z, max_z);
            {
              o3d_cuda.addHistogramVolumes (rivhistograms);
              o3d_cuda.addEdgesIV (rivs);
              o3d_cuda.addOccupancyVolumes (rivs_occupancy);
              o3d_cuda.addFullVolumes (rivs_full);
              o3d_cuda.addOcclusionVolumes (rivs_occluded);
              o3d_cuda.setNPointsLabel (npoints_label_);
              o3d_cuda.setViewpoint(vpx_, vpy_, vpz_);
              //o3d_cuda.setMinMaxValues (min_x, max_x, min_y, max_y, min_z, max_z);
              o3d_cuda.addColorHistogramVolumes(riv_color_histograms_, riv_points_color_histogram_, ycolor_size_ * ucolor_size_ * vcolor_size_);
              o3d_cuda.setResolution (resolution);
              pcl::ScopeTime t ("evaluateBoundingBoxes cuda and adding volumes");
              o3d_cuda.generateAndComputeBoundingBoxesScore (bounding_boxes, max_score_threshold_);
              std::cout << bounding_boxes.size() << std::endl;
            }
          }

          if(best_wins_ == -1)
          {
            {
              pcl::ScopeTime t ("Filtering and sorting bboxes");
              std::vector<BBox>::iterator max_bbox = std::max_element (bounding_boxes.begin (), bounding_boxes.end (), BBoxless);
              float max_score = max_bbox->score;
              std::cout << "max score:" << max_score << std::endl;
              int valid = 0;
              for (std::vector<BBox>::iterator it = bounding_boxes.begin (); it != bounding_boxes.end (); ++it)
              {
                //it->score /= max_score;
                if ((it->score) > max_score_threshold_)
                {
                  bounding_boxes[valid] = *it;
                  valid++;
                }
              }

              bounding_boxes.resize (valid);
              std::cout << "bboxes over threshold:" << bounding_boxes.size () << std::endl;

                //group bounding boxes based on their center distance
                std::vector < std::vector<int> > clusters;
                std::vector < Eigen::Vector3f > clusters_mean;
                float large_radius = 0.05f;
                large_radius *= large_radius;
                for (size_t i = 0; i < bounding_boxes.size (); i++)
                {
                  Eigen::Vector3f center_ri = BBoxCenter (bounding_boxes[i]);
                  std::vector<bool> valid_in_cluster (clusters_mean.size (), false);
                  bool found = false;
                  for (size_t j = 0; j < clusters_mean.size (); j++)
                  {
                    float sq_norm = (clusters_mean[j] - center_ri).squaredNorm ();
                    if (sq_norm < large_radius)
                    {
                      //found one cluster, update cluster mean and append index
                      valid_in_cluster[j] = true;
                      found = true;
                    }
                  }

                  //no cluster found, create new cluster
                  if (!found)
                  {
                    std::vector < int > ind;
                    ind.push_back (static_cast<int>(i));
                    clusters.push_back (ind);
                    clusters_mean.push_back (center_ri);
                    continue;
                  }

                  //get the closest cluster and put if there
                  int idx = -1;
                  float closest_dist = std::numeric_limits<float>::max();
                  for (size_t j = 0; j < clusters_mean.size (); j++)
                  {
                    if(!valid_in_cluster[j])
                      continue;
                    float sq_norm = (clusters_mean[j] - center_ri).squaredNorm ();
                    if (sq_norm < closest_dist)
                    {
                      idx = static_cast<int>(j);
                      closest_dist = sq_norm;
                    }
                  }

                  clusters_mean[idx] = (clusters_mean[idx] * (static_cast<float> (clusters[idx].size ())) + center_ri)
                      / (static_cast<float> (clusters[idx].size ()) + 1.f);
                  clusters[idx].push_back (static_cast<int>(i));
                }

                std::cout << "Number of clusters:" << clusters.size () << std::endl;
                for (size_t i = 0; i < clusters.size (); i++)
                {
                  std::cout << clusters[i].size () << std::endl;
                }

                float voxel_size_nms = 0.01f;
                typename pcl::PointCloud<PointInT>::Ptr cloud_downsampled_nms;
                cloud_downsampled_nms.reset (new pcl::PointCloud<PointInT>);
                pcl::VoxelGrid<PointInT> voxel_grid;
                voxel_grid.setInputCloud (cloud_for_iv);
                voxel_grid.setDownsampleAllData (true);
                voxel_grid.setLeafSize (voxel_size_nms, voxel_size_nms, voxel_size_nms);
                voxel_grid.filter (*cloud_downsampled_nms);

                std::vector<BBox> merged_bounding_boxes;
    #pragma omp parallel for schedule(dynamic) num_threads(omp_get_num_procs())
                for (int i = 0; i < clusters.size (); i++)
                {
                  std::vector<BBox> cluster_boxes;
                  cluster_boxes.resize (clusters[i].size ());
                  for (size_t j = 0; j < cluster_boxes.size (); j++)
                  {
                    cluster_boxes[j] = bounding_boxes[clusters[i][j]];
                  }

                  if (cluster_boxes.size () > num_wins_)
                    std::sort (cluster_boxes.begin (), cluster_boxes.end (), sortBBoxes);

                  //visualizeBoundingBoxes(cloud_for_iv, cluster_boxes, transform_to_plane, false, false);
                  std::vector<BBox> maximas;
                  //nonMaximaSupression (cluster_boxes, num_wins_, maximas);
                  nonMaximaSupressionExplainedPoints (cluster_boxes, num_wins_, maximas, cloud_downsampled_nms, false);
                  std::cout << cluster_boxes.size() << " " << maximas.size() << " " << num_wins_ << std::endl;

                  //visualizeBoundingBoxes(cloud_for_iv, maximas, transform_to_plane, false, false);
    #pragma omp critical
                  {
                    merged_bounding_boxes.insert (merged_bounding_boxes.begin (), maximas.begin (), maximas.end ());
                  }
                }

                {
                  PointInTPtr cloud_for_bbo2 (new pcl::PointCloud<PointInT>);

                  for (size_t i = 0; i < cloud_for_iv->points.size (); i++)
                  {
                    if (!pcl::isFinite (cloud_for_iv->points[i]))
                      continue;

                    Eigen::Vector3f p = cloud_for_iv->points[i].getVector3fMap ();

                    if (p[2] >= 0.01f)
                      cloud_for_bbo2->points.push_back (cloud_for_iv->points[i]);
                  }

                  PointInTPtr cloud_for_bbo (new pcl::PointCloud<PointInT>);
                  pcl::VoxelGrid<PointInT> voxel_grid;
                  voxel_grid.setInputCloud (cloud_for_iv);
                  voxel_grid.setInputCloud (cloud_for_bbo2);
                  voxel_grid.setLeafSize (voxel_size_nms, voxel_size_nms, voxel_size_nms);
                  voxel_grid.filter (*cloud_for_bbo);

                  {
                    pcl::ScopeTime t("fine maxima supression");
                    std::vector<BBox> maximas;
                    nonMaximaSupressionExplainedPoints (merged_bounding_boxes, 3000, maximas, cloud_for_bbo, true);
                    std::cout << merged_bounding_boxes.size() << " " << maximas.size();
                    bounding_boxes = maximas;
                  }

                  if(visualize_)
                  {
                    /*for (size_t j = 0; j < maximas.size (); j++)
                    {
                      std::cout << "Score:" << maximas[j].score << std::endl;
                      printSomeObjectnessValues(maximas[j]);
                    }*/

                    visualizeBoundingBoxes(cloud_for_iv, bounding_boxes, transform_to_plane, false, false);
                  }
                }

                //bounding_boxes = merged_bounding_boxes;
                /*if (bounding_boxes.size () > num_wins_)
                  std::sort (bounding_boxes.begin (), bounding_boxes.end (), sortBBoxes);*/
              }

              /*std::vector<BBox> maximas;
              {
                pcl::ScopeTime t ("nonMaximaSupression of bboxes");
                nonMaximaSupression (bounding_boxes, num_wins_, maximas);
                bounding_boxes = maximas;
              }*/

              /*if (visualize_)
                visualizeBoundingBoxes (cloud_for_iv, bounding_boxes, transform_to_plane, false, false);*/

              if (do_optimize_)
                optimizeBoundingBoxes (cloud_for_iv, bounding_boxes, transform_to_plane);

              //evaluateBoundingBoxes (bounding_boxes, true);
              for(size_t jj=0; jj < bounding_boxes.size(); jj++)
                printSomeObjectnessValues(bounding_boxes[jj]);

              final_boxes_ = bounding_boxes;
              cloud_for_iv_ = cloud_for_iv;

              if (visualize_) {
                visualizeBoundingBoxes (cloud_for_iv, final_boxes_, transform_to_plane, false, false);

                /*pcl::visualization::PCLVisualizer vis ("occlusion");

                int v1, v2, v3;
                vis.createViewPort (0.0, 0.0, 0.33, 1.0, v1);
                vis.createViewPort (0.33, 0.0, 0.66, 1.0, v2);
                vis.createViewPort (0.66, 0.0, 1.0, 1.0, v3);

                pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (input_cloud);
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud_trans(new pcl::PointCloud<pcl::PointXYZRGB>);
                pcl::transformPointCloud (*input_cloud, *input_cloud_trans, transform_to_plane);
                vis.addPointCloud<PointInT> (input_cloud_trans, handler, "input cloud", v2);

                pcl::visualization::PointCloudColorHandlerCustom<PointInT> handler_filtered (filtered, 255, 0, 0);
                typename pcl::PointCloud<PointInT>::Ptr filtered_trans(new pcl::PointCloud<PointInT>);
                pcl::transformPointCloud (*filtered, *filtered_trans, transform_to_plane);
                vis.addPointCloud<PointInT> (filtered_trans, handler_filtered, "filtered", v2);

                pcl::visualization::PointCloudColorHandlerCustom<PointInT> random_handler (cloud_for_iv, 128, 128, 128);
                vis.addPointCloud<PointInT> (cloud_for_iv, random_handler, "points", v1);

                {
                  pcl::PointCloud<pcl::PointXYZL>::Ptr label_cloud (new pcl::PointCloud<pcl::PointXYZL>);
                  pcl::transformPointCloud (*label_cloud_, *label_cloud, transform_to_plane);
                  pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZL> handler (label_cloud, "label");
                  vis.addPointCloud<pcl::PointXYZL> (label_cloud, handler, "edges", v1);
                  vis.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 12, "edges");
                }

                {
                  pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZL> handler (smooth_labels_cloud_, "label");
                  vis.addPointCloud<pcl::PointXYZL> (smooth_labels_cloud_, handler, "smooth_labels", v3);
                }
                vis.spin ();*/
              }
          }
          else
          {
            std::vector<BBox>::iterator max_bbox = std::max_element (bounding_boxes.begin (), bounding_boxes.end (), BBoxless);
            float max_score = max_bbox->score;
            std::cout << "max score:" << max_score << std::endl;
            int valid = 0;
            for (std::vector<BBox>::iterator it = bounding_boxes.begin (); it != bounding_boxes.end (); ++it)
            {
              //it->score /= max_score;
              if ((it->score) > max_score_threshold_)
              {
                bounding_boxes[valid] = *it;
                valid++;
              }
            }

            bounding_boxes.resize(valid);
            std::sort(bounding_boxes.begin(), bounding_boxes.end(), sortBBoxes);
            bounding_boxes.resize (std::min(valid, best_wins_));
            final_boxes_ = bounding_boxes;
            if (visualize_) {

              for (size_t j = 0; j < final_boxes_.size (); j++)
              {
                std::cout << "Score:" << final_boxes_[j].score << std::endl;
                printSomeObjectnessValues(final_boxes_[j]);
              }

              visualizeBoundingBoxes (cloud_for_iv, final_boxes_, transform_to_plane, false, false);
            }
          }
        }
        else
        {
          cloud_for_iv.reset (new pcl::PointCloud<PointInT> (*input_cloud));
        }
      }

    template<typename PointInT>
      void
      Objectness3D<PointInT>::visualizeBoundingBoxes (PointInTPtr & cloud_for_iv, std::vector<BBox> & bounding_boxes,
                                                      Eigen::Matrix4f & transform_to_plane, bool vis_shrinked, bool vis_extended)
      {
        PointInTPtr grid (new pcl::PointCloud<PointInT> ());
        pcl::PassThrough<PointInT> pass_;
        pass_.setFilterLimits (min_z, max_z);
        pass_.setFilterFieldName ("z");
        pass_.setInputCloud (cloud_for_iv);
        pass_.filter (*grid);
        pass_.setFilterLimits (min_x, max_x);
        pass_.setFilterFieldName ("x");
        pass_.setInputCloud (grid);
        pass_.filter (*grid);
        pass_.setFilterLimits (min_y, max_y);
        pass_.setFilterFieldName ("y");
        pass_.setInputCloud (grid);
        pass_.filter (*grid);

        pcl::visualization::PCLVisualizer vis ("normals and points");

        bool do_viewports = false;

        if(do_viewports) {
          int v1, v2;
          vis.createViewPort (0.0, 0.0, 1.0, 0.5, v1);
          vis.createViewPort (0.0, 0.5, 1.0, 1.0, v2);

          {
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (input_cloud);
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud_trans(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::transformPointCloud (*input_cloud, *input_cloud_trans, transform_to_plane);
            vis.addPointCloud<PointInT> (input_cloud_trans, handler, "input cloud", v1);

            //pcl::visualization::PointCloudColorHandlerCustom<PointInT> random_handler (grid, 128, 128, 128);
            //vis.addPointCloud<PointInT> (grid, random_handler, "points");
          }

          pcl::PointCloud<pcl::PointXYZL>::Ptr label_cloud (new pcl::PointCloud<pcl::PointXYZL>);
          pcl::transformPointCloud (*label_cloud_, *label_cloud, transform_to_plane);

          {
            pcl::PassThrough<pcl::PointXYZL> pass_;
            pass_.setFilterLimits (min_z, max_z);
            pass_.setFilterFieldName ("z");
            pass_.setInputCloud (label_cloud);
            pass_.filter (*label_cloud);
            pass_.setFilterLimits (min_x, max_x);
            pass_.setFilterFieldName ("x");
            pass_.setInputCloud (label_cloud);
            pass_.filter (*label_cloud);
            pass_.setFilterLimits (min_y, max_y);
            pass_.setFilterFieldName ("y");
            pass_.setInputCloud (label_cloud);
            pass_.filter (*label_cloud);
          }

          pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZL> handler (label_cloud, "label");
          vis.addPointCloud<pcl::PointXYZL> (label_cloud, handler, "edges", v2);
          vis.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 12, "edges");
        } else {
          pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (input_cloud);
          pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud_trans(new pcl::PointCloud<pcl::PointXYZRGB>);
          pcl::transformPointCloud (*input_cloud, *input_cloud_trans, transform_to_plane);
          vis.addPointCloud<PointInT> (input_cloud_trans, handler, "input cloud");
        }

        //add the first bounding boxes
        std::vector<Eigen::Vector3f> colors_;
        colors_.push_back (Eigen::Vector3f (0.25f, 0.25f, 0.25f));
        colors_.push_back (Eigen::Vector3f (0.5f, 0.5f, 0.5f));
        colors_.push_back (Eigen::Vector3f (0.75f, 0.75f, 0.75f));
        colors_.push_back (Eigen::Vector3f (1.f, 1.f, 1.f));

        int GRIDSIZE_X = (int)((max_x - min_x) / resolution);
        int GRIDSIZE_Y = (int)((max_y - min_y) / resolution);
        int GRIDSIZE_Z = (int)((max_z - min_z) / resolution);

        for (size_t i = 0; i < static_cast<int> (bounding_boxes.size ()); i++)
        {
          std::stringstream name;
          name << "box_" << i;
          BBox bb = bounding_boxes[i];
          visBBox (vis, bb, name);

          if (vis_shrinked)
          {
            name << "shrinked";
            BBox bb_shrinked;
            shrink_bbox (bb, bb_shrinked);
            bb_shrinked.angle = bb.angle;
            visBBox (vis, bb_shrinked, name);
          }

          if (vis_extended)
          {
            BBox bb_extended;
            bb_extended.sx = static_cast<int> (pcl_round (bb.sx * expand_factor_));
            bb_extended.sy = static_cast<int> (pcl_round (bb.sy * expand_factor_));
            bb_extended.sz = static_cast<int> (pcl_round (bb.sz * expand_factor_));

            bb_extended.x = bb.x - static_cast<int> (pcl_round ((bb_extended.sx - bb.sx) / 2.f));
            bb_extended.y = bb.y - static_cast<int> (pcl_round ((bb_extended.sy - bb.sy) / 2.f));
            bb_extended.z = bb.z - static_cast<int> (pcl_round ((bb_extended.sz - bb.sz) / 2.f));

            bb_extended.x = std::max (bb_extended.x, 1);
            bb_extended.y = std::max (bb_extended.y, 1);
            bb_extended.z = std::max (bb_extended.z, 1);

            bb_extended.sx = std::min (GRIDSIZE_X - 1, bb_extended.x + bb_extended.sx) - bb_extended.x;
            bb_extended.sy = std::min (GRIDSIZE_Y - 1, bb_extended.y + bb_extended.sy) - bb_extended.y;
            bb_extended.sz = std::min (GRIDSIZE_Z - 1, bb_extended.z + bb_extended.sz) - bb_extended.z;
            bb_extended.angle = bb.angle;
            name << "extended";
            visBBox (vis, bb_extended, name);
          }
        }

        vis.spin ();

      }

    template<typename PointInT>
      void
      Objectness3D<PointInT>::visBBox (pcl::visualization::PCLVisualizer & vis, BBox & bb, std::stringstream & name, int viewport)
      {
        float fac = 0.f;
        Eigen::Vector4f minxyz, maxxyz;
        minxyz[0] = min_x + (bb.x) * resolution - resolution;
        minxyz[1] = min_y + (bb.y) * resolution - resolution;
        minxyz[2] = min_z + (bb.z) * resolution - resolution;
        minxyz[3] = 1.f;

        maxxyz[0] = min_x + (bb.sx + bb.x) * resolution + resolution;
        maxxyz[1] = min_y + (bb.sy + bb.y) * resolution + resolution;
        maxxyz[2] = min_z + (bb.sz + bb.z) * resolution + resolution;
        maxxyz[3] = 1.f;

        if (bb.angle != 0)
        {
          float rot_angle = pcl::deg2rad (static_cast<float> (bb.angle * angle_incr_ * -1.f));
          Eigen::Affine3f rot_trans = Eigen::Affine3f (Eigen::AngleAxisf (static_cast<float> (rot_angle), Eigen::Vector3f::UnitZ ()));

          minxyz = rot_trans * minxyz;
          maxxyz = rot_trans * maxxyz;
        }

        float rot_angle = pcl::deg2rad (static_cast<float> (bb.angle * angle_incr_ * -1.f));
        Eigen::Vector3f translation ((minxyz[0] + maxxyz[0]) / 2.f, (minxyz[1] + maxxyz[1]) / 2.f, (minxyz[2] + maxxyz[2]) / 2.f);
        Eigen::Quaternionf quat;
        quat = Eigen::AngleAxisf (static_cast<float> (rot_angle), Eigen::Vector3f::UnitZ ());
        vis.addCube (translation, quat, bb.sx * resolution * 1.f, bb.sy * resolution * 1.f, bb.sz * resolution * 1.f, name.str (), viewport);
        vis.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, name.str());
        vis.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, name.str());
        name << "sphere";
        pcl::PointXYZ p1;
        p1.getVector3fMap () = translation;
        vis.addSphere<pcl::PointXYZ> (p1, 0.02, 0, 255, 0, name.str (), viewport);

        /*{
          p1.getVector3fMap () = Eigen::Vector3f(minxyz[0],minxyz[1],minxyz[2]);
          name << "_sphere_start";
          vis.addSphere<pcl::PointXYZ> (p1, 0.01, 255, 0, 0, name.str ());
        }*/
      }

    template<typename PointInT>
      void
      Objectness3D<PointInT>::visBBox (pcl::visualization::PCLVisualizer & vis, BBox & bb, std::stringstream & name, Eigen::Matrix4f & matrix_trans, int viewport)
      {

        float fac = 0.f;
        Eigen::Vector4f minxyz, maxxyz;
        minxyz[0] = min_x + (bb.x) * resolution - resolution;
        minxyz[1] = min_y + (bb.y) * resolution - resolution;
        minxyz[2] = min_z + (bb.z) * resolution - resolution;
        minxyz[3] = 1.f;

        maxxyz[0] = min_x + (bb.sx + bb.x) * resolution + resolution;
        maxxyz[1] = min_y + (bb.sy + bb.y) * resolution + resolution;
        maxxyz[2] = min_z + (bb.sz + bb.z) * resolution + resolution;
        maxxyz[3] = 1.f;

        if (bb.angle != 0)
        {
          float rot_angle = pcl::deg2rad (static_cast<float> (bb.angle * angle_incr_ * -1.f));
          Eigen::Affine3f rot_trans = Eigen::Affine3f (Eigen::AngleAxisf (static_cast<float> (rot_angle), Eigen::Vector3f::UnitZ ()));

          minxyz = rot_trans * minxyz;
          maxxyz = rot_trans * maxxyz;
        }

        std::cout << "Transforming corner points..." << std::endl;
        minxyz = matrix_trans * minxyz;
        maxxyz = matrix_trans * maxxyz;

        float rot_angle = pcl::deg2rad (static_cast<float> (bb.angle * angle_incr_ * -1.f));
        Eigen::Vector3f translation ((minxyz[0] + maxxyz[0]) / 2.f, (minxyz[1] + maxxyz[1]) / 2.f, (minxyz[2] + maxxyz[2]) / 2.f);
        Eigen::Quaternionf quat;
        quat = Eigen::AngleAxisf (static_cast<float> (rot_angle), Eigen::Vector3f::UnitZ ());
        Eigen::Matrix3f rot_quat = quat.toRotationMatrix ();
        std::cout << rot_quat << std::endl;

        Eigen::Matrix3f mat_3f = matrix_trans.block<3, 3> (0, 0);
        std::cout << mat_3f << std::endl;
        Eigen::Matrix3f rot_mat = mat_3f * rot_quat;
        //rot_mat = rot_mat.inverse().eval();
        std::cout << rot_mat << std::endl;
        Eigen::Quaternionf quat_mat (rot_mat);
        std::cout << quat_mat.x () << " " << quat_mat.y () << " " << quat_mat.z () << " " << quat_mat.w () << std::endl;
        //quat_mat.normalize();
        //std::cout << quat_mat.x() << " " << quat_mat.y() << " " << quat_mat.z() << " " << quat_mat.w() << std::endl;
        /*quat.normalize();
         quat_mat.normalize();
         quat = quat_mat * quat;
         quat.normalize();*/
        //std::cout << quat_mat << std::endl;
        //std::cout << quat << std::endl;

        vis.addCube (translation, quat_mat, bb.sx * resolution * 1.f, bb.sy * resolution * 1.f, bb.sz * resolution * 1.f, name.str ());
        name << "sphere";
        pcl::PointXYZ p1;
        p1.getVector3fMap () = translation;
        vis.addSphere<pcl::PointXYZ> (p1, 0.02, 0, 255, 0, name.str ());

      }

    inline void
    visualizeIVData (float * data, int GRIDSIZE_X, int GRIDSIZE_Y, int GRID_SIZE_Z)
    {
      pcl::visualization::PCLVisualizer vis ("[float] visualizeIVData");

      pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
      for (int x = 0; x < GRIDSIZE_X; x++)
      {
        for (int y = 0; y < GRIDSIZE_Y; y++)
        {
          for (int z = 0; z < GRID_SIZE_Z; z++)
          {
            int idx = z * GRIDSIZE_X * GRIDSIZE_Y + y * GRIDSIZE_X + x;
            if (data[idx])
            {
              pcl::PointXYZI p;
              p.getVector3fMap () = Eigen::Vector3f (x, y, z);
              p.intensity = data[idx];
              cloud->push_back (p);
            }
          }
        }
      }

      pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> random_handler (cloud, "intensity");
      vis.addPointCloud<pcl::PointXYZI> (cloud, random_handler, "original points");
      vis.addCoordinateSystem (100);
      vis.spin ();
    }

    template<typename PointInT>
      void
      Objectness3D<PointInT>::transformToBeCenteredOnPlane (Eigen::Vector4f & plane_, Eigen::Matrix4f & inv)
      {
        Eigen::Vector3f zp = plane_.head<3> ();
        zp.normalize ();
        Eigen::Vector3f yp = (Eigen::Vector3f::UnitY ()).cross (zp);
        yp.normalize ();
        Eigen::Vector3f xp = zp.cross (yp);
        xp.normalize ();

        Eigen::Vector3f p_on_plane = Eigen::Vector3f (0.f, 0.f, 0.f);
        p_on_plane[2] = -plane_[3] / plane_[2];

        Eigen::Matrix4f rot_matrix;
        rot_matrix.setIdentity ();
        rot_matrix.block<3, 1> (0, 0) = xp;
        rot_matrix.block<3, 1> (0, 1) = yp;
        rot_matrix.block<3, 1> (0, 2) = zp;
        rot_matrix.block<3, 1> (0, 3) = p_on_plane.head<3> ();

        inv = rot_matrix.inverse ();
      }

    template<typename PointInT>
    void
    Objectness3D<PointInT>::nonMaximaSupressionExplainedPoints (const std::vector<BBox> & rectangles,
                                                                int n_rects, std::vector<BBox> & maximas, PointInTPtr & cloud_downsampled_, bool fine) {
      typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, BBox> Graph;
      Graph conflict_graph_;
      std::map<int, BBox> graph_id_model_map_;

      n_rects = std::min (n_rects, static_cast<int> (rectangles.size ()));
      std::vector<std::vector<int> > explained_points;
      explained_points.resize(n_rects);

      //std::cout << "Points cloud downsampled:" << cloud_downsampled_->points.size() << " " << cloud_for_iv->points.size() << std::endl;

      for (size_t i = 0; i < n_rects; i++)
      {
        BBox ri = rectangles[i];
        ri.id = i;
        const typename Graph::vertex_descriptor v = boost::add_vertex (ri, conflict_graph_);
        graph_id_model_map_[int (v)] = static_cast<BBox> (ri);
        getInsideBox (ri, cloud_downsampled_, explained_points[i]);
        /*if (fine)
          std::sort (explained_points[i].begin (), explained_points[i].end ());*/
      }

      for (size_t i = 0; i < n_rects; i++) {
        for (size_t j = i; j < n_rects; j++)
        {
          if (i != j) {
            float min_p = static_cast<float>(std::min(explained_points[i].size(), explained_points[j].size()));
            float max_p = static_cast<float>(std::max(explained_points[i].size(), explained_points[j].size()));

            if(std::abs(rectangles[i].angle - rectangles[j].angle) > 10)
              continue;

            if((min_p / max_p) > 0.9f) {
              if (fine)
              {
                //compute a refined intersection of the explained points from the two hypothesis
                std::map<int, int> count_p;
                std::map<int, int>::iterator it;
                for (size_t k = 0; k < explained_points[i].size (); k++)
                {
                  count_p[explained_points[i][k]] = 1;
                }

                for (size_t k = 0; k < explained_points[j].size (); k++)
                {
                  it = count_p.find (explained_points[j][k]);
                  if (it != count_p.end ())
                  {
                    (*it).second++;
                  }
                }

                float intersection = 0;
                for (it = count_p.begin (); it != count_p.end (); it++)
                {
                  if ((*it).second > 1)
                  {
                    intersection+=1.f;
                  }
                }

                float thres = 0.95f;
                if ((intersection / max_p) > thres && (intersection / min_p) > thres)
                {
                  boost::add_edge (i, j, conflict_graph_);
                }

              }
              else
              {
                boost::add_edge (i, j, conflict_graph_);
              }
            }
          }
        }
      }

      std::vector<bool> mask_ (n_rects, true);

      // iterate over all vertices of the graph and check if they have a better neighbour, then remove that vertex
      typedef typename boost::graph_traits<Graph>::vertex_iterator VertexIterator;
      VertexIterator vi, vi_end, next;
      boost::tie (vi, vi_end) = boost::vertices (conflict_graph_);

      size_t i = 0;
      for (next = vi; next != vi_end; next++)
      {
        const typename Graph::vertex_descriptor v = boost::vertex (*next, conflict_graph_);
        typename boost::graph_traits<Graph>::adjacency_iterator ai;
        typename boost::graph_traits<Graph>::adjacency_iterator ai_end;

        BBox current = static_cast<BBox> (graph_id_model_map_[int (v)]);

        bool a_better_one = false;
        for (boost::tie (ai, ai_end) = boost::adjacent_vertices (v, conflict_graph_); (ai != ai_end) && !a_better_one; ++ai)
        {
          BBox neighbour = static_cast<BBox> (graph_id_model_map_[int (*ai)]);
          if (neighbour.score >= current.score && mask_[neighbour.id])
          {
            a_better_one = true;
          }
        }

        if (a_better_one)
        {
          //std::cout << current.id << std::endl;
          mask_[current.id] = false;
        }
        i++;
      }

      for (size_t i = 0; i < n_rects; i++)
      {
        if (mask_[i])
        {
          maximas.push_back (rectangles[i]);
        }
      }
    }

    template<typename PointInT>
      void
      Objectness3D<PointInT>::nonMaximaSupression (const std::vector<BBox> & rectangles, int n_rects, std::vector<BBox> & maximas)
      {
        typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, BBox> Graph;
        Graph conflict_graph_;
        std::map<int, BBox> graph_id_model_map_;

        n_rects = std::min (n_rects, static_cast<int> (rectangles.size ()));
        for (size_t i = 0; i < n_rects; i++)
        {
          BBox ri = rectangles[i];
          ri.id = i;
          const typename Graph::vertex_descriptor v = boost::add_vertex (ri, conflict_graph_);
          graph_id_model_map_[int (v)] = static_cast<BBox> (ri);
        }

        float min_dist_conflict = 0.05f;
        min_dist_conflict *= min_dist_conflict;
        float min_diff_edges = 0.5f;

        for (size_t i = 0; i < n_rects; i++)
        {
          Eigen::Vector3f center_ri = BBoxCenter (rectangles[i]);
          //Eigen::Vector3f vol_ri = Eigen::Vector3f (rectangles[i].sx, rectangles[i].sy, rectangles[i].sz);
          float sxi = static_cast<float> (rectangles[i].sx);
          float syi = static_cast<float> (rectangles[i].sy);
          float szi = static_cast<float> (rectangles[i].sz);
          for (size_t j = i; j < n_rects; j++)
          {
            if (i != j)
            {
              Eigen::Vector3f center_rj = BBoxCenter (rectangles[j]);
              float sxj = static_cast<float> (rectangles[j].sx);
              float syj = static_cast<float> (rectangles[j].sy);
              float szj = static_cast<float> (rectangles[j].sz);
              //Eigen::Vector3f vol_rj = Eigen::Vector3f (rectangles[j].sx, rectangles[j].sy, rectangles[j].sz);
              if ((center_ri - center_rj).squaredNorm () < min_dist_conflict
              //&& (vol_ri - vol_rj).squaredNorm () < min_dist_conflict_volume
                  /*&& (rectangles[i].angle == rectangles[j].angle*/
                  && ((std::min (sxi, sxj) / std::max (sxi, sxj)) > min_diff_edges) && ((std::min (syi, syj) / std::max (syi, syj)) > min_diff_edges)
                  && ((std::min (szi, szj) / std::max (szi, szj)) > min_diff_edges) && (std::abs (rectangles[i].angle - rectangles[j].angle) <= 15.f))
              {
                //create a conflict
                boost::add_edge (i, j, conflict_graph_);
              }
            }
          }
        }

        std::vector<bool> mask_ (n_rects, true);

        // iterate over all vertices of the graph and check if they have a better neighbour, then remove that vertex
        typedef typename boost::graph_traits<Graph>::vertex_iterator VertexIterator;
        VertexIterator vi, vi_end, next;
        boost::tie (vi, vi_end) = boost::vertices (conflict_graph_);

        size_t i = 0;
        for (next = vi; next != vi_end; next++)
        {
          const typename Graph::vertex_descriptor v = boost::vertex (*next, conflict_graph_);
          typename boost::graph_traits<Graph>::adjacency_iterator ai;
          typename boost::graph_traits<Graph>::adjacency_iterator ai_end;

          BBox current = static_cast<BBox> (graph_id_model_map_[int (v)]);

          bool a_better_one = false;
          for (boost::tie (ai, ai_end) = boost::adjacent_vertices (v, conflict_graph_); (ai != ai_end) && !a_better_one; ++ai)
          {
            BBox neighbour = static_cast<BBox> (graph_id_model_map_[int (*ai)]);
            if (neighbour.score >= current.score && mask_[neighbour.id])
            {
              a_better_one = true;
            }
          }

          if (a_better_one)
          {
            //std::cout << current.id << std::endl;
            mask_[current.id] = false;
          }
          i++;
        }

        for (size_t i = 0; i < n_rects; i++)
        {
          if (mask_[i])
          {
            maximas.push_back (rectangles[i]);
          }
        }

        std::cout << "Number of rectangles after NonMaxima" << maximas.size () << std::endl;
      }

    template<typename PointInT>
      Objectness3D<PointInT>::Objectness3D (std::string & sf, int nsw, int nr, int minsw, int maxsw, int angle_incr, float max_thres)
      {
        table_plane_set_ = false;
        min_z = -0.04f;
        max_z = 0.42f;
        min_x = -0.4f;
        max_x = 0.4f;
        min_y = -0.4f;
        max_y = 0.4f;
        resolution = 0.01f;
        start_z_ = static_cast<int>(abs(pcl_round(min_z / resolution)));

        float fac_res_to_cm = 1.f / resolution;

        std::cout << "START Z:" << start_z_ << std::endl;

        std::vector<std::string> strs;
        boost::split (strs, sf, boost::is_any_of (","));

        shrink_factor_x = atof (strs[0].c_str ());
        shrink_factor_y = atof (strs[1].c_str ());
        shrink_factor_z = atof (strs[2].c_str ());

        num_sampled_wins_ = nsw;
        num_wins_ = nr;
        min_size_w_ = (minsw / 100.f) / resolution;
        max_size_w_ = (maxsw / 100.f) / resolution;
        angle_incr_ = angle_incr;
        max_score_threshold_ = max_thres;
        visualize_ = true;
        do_z_ = false;
        do_optimize_ = true;
        expand_factor_ = 1.5f;

        do_cuda_ = true;

        vpx_ = 0.f;
        vpy_ = 0.f;
        vpz_ = 0.f;

        ycolor_size_ = 2;
        ucolor_size_ = 4;
        vcolor_size_ = 4;

        best_wins_ = -1;
        opt_type_ = 1;
      }
  }
}
template class faat_pcl::segmentation::Objectness3D<pcl::PointXYZRGB>;
