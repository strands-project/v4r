/*
 * objectness_3D.H
 *
 *  Created on: Jul 27, 2012
 *      Author: aitor
 */

#ifndef OBJECTNESS_3D_H_
#define OBJECTNESS_3D_H_

#include "faat_pcl/utils/integral_volume.h"
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include "pcl/common/angles.h"
#include "faat_pcl/segmentation/objectness_3d/objectness_common.h"
#include <pcl/search/search.h>
#include "pcl/filters/crop_box.h"
#include <pcl/filters/voxel_grid.h>

namespace faat_pcl
{
  namespace segmentation
  {

    template<typename PointT, typename NormalT>
      inline void
      extractEuclideanClustersSmooth (const typename pcl::PointCloud<PointT> &cloud, const typename pcl::PointCloud<NormalT> &normals,
                                      float tolerance, const typename pcl::search::Search<PointT>::Ptr &tree,
                                      std::vector<pcl::PointIndices> &clusters, double eps_angle, float curvature_threshold,
                                      unsigned int min_pts_per_cluster, unsigned int max_pts_per_cluster = (std::numeric_limits<int>::max) ())
      {

        if (tree->getInputCloud ()->points.size () != cloud.points.size ())
        {
          PCL_ERROR("[pcl::extractEuclideanClusters] Tree built for a different point cloud dataset\n");
          return;
        }
        if (cloud.points.size () != normals.points.size ())
        {
          PCL_ERROR("[pcl::extractEuclideanClusters] Number of points in the input point cloud different than normals!\n");
          return;
        }

        // Create a bool vector of processed point indices, and initialize it to false
        std::vector<bool> processed (cloud.points.size (), false);

        std::vector<int> nn_indices;
        std::vector<float> nn_distances;
        // Process all points in the indices vector
        int size = static_cast<int> (cloud.points.size ());
        for (int i = 0; i < size; ++i)
        {
          if (processed[i])
            continue;

          std::vector<unsigned int> seed_queue;
          int sq_idx = 0;
          seed_queue.push_back (i);

          processed[i] = true;

          while (sq_idx < static_cast<int> (seed_queue.size ()))
          {

            if (normals.points[seed_queue[sq_idx]].curvature > curvature_threshold)
            {
              sq_idx++;
              continue;
            }

            // Search for sq_idx
            if (!tree->radiusSearch (seed_queue[sq_idx], tolerance, nn_indices, nn_distances))
            {
              sq_idx++;
              continue;
            }

            for (size_t j = 1; j < nn_indices.size (); ++j) // nn_indices[0] should be sq_idx
            {
              if (processed[nn_indices[j]]) // Has this point been processed before ?
                continue;

              if (normals.points[nn_indices[j]].curvature > curvature_threshold)
              {
                continue;
              }

              //processed[nn_indices[j]] = true;
              // [-1;1]

              double dot_p = normals.points[seed_queue[sq_idx]].normal[0] * normals.points[nn_indices[j]].normal[0]
                  + normals.points[seed_queue[sq_idx]].normal[1] * normals.points[nn_indices[j]].normal[1]
                  + normals.points[seed_queue[sq_idx]].normal[2] * normals.points[nn_indices[j]].normal[2];

              if (fabs (acos (dot_p)) < eps_angle)
              {
                processed[nn_indices[j]] = true;
                seed_queue.push_back (nn_indices[j]);
              }
            }

            sq_idx++;
          }

          // If this queue is satisfactory, add to the clusters
          if (seed_queue.size () >= min_pts_per_cluster && seed_queue.size () <= max_pts_per_cluster)
          {
            pcl::PointIndices r;
            r.indices.resize (seed_queue.size ());
            for (size_t j = 0; j < seed_queue.size (); ++j)
              r.indices[j] = seed_queue[j];

            std::sort (r.indices.begin (), r.indices.end ());
            r.indices.erase (std::unique (r.indices.begin (), r.indices.end ()), r.indices.end ());
            clusters.push_back (r); // We could avoid a copy by working directly in the vector
          }
        }
      }

    template<typename PointInT>
      class Objectness3D
      {
        typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;

      private:
        PointInTPtr input_cloud;
        std::vector<int> edges_;
        pcl::PointCloud<pcl::PointXYZL>::Ptr label_cloud_;

        Eigen::Vector4f table_plane_;
        bool table_plane_set_;

        float min_z;
        float max_z;
        int start_z_;
        float min_x;
        float max_x;

        float min_y;
        float max_y;
        float resolution;

        float shrink_factor_x;
        float shrink_factor_y;
        float shrink_factor_z;
        int num_sampled_wins_;
        int num_wins_; //number of windows that we want as result...
        int min_size_w_;
        int max_size_w_;
        int angle_incr_; //in degrees
        float max_score_threshold_;
        bool do_z_;
        bool do_optimize_;
        float expand_factor_;
        int opt_type_;

        std::vector<boost::shared_ptr<IntegralVolume> > projected_occupancy_;
        std::vector<boost::shared_ptr<IntegralVolume> > projected_occluded_;
        std::vector<boost::shared_ptr<IntegralVolume> > rivs;
        std::vector<boost::shared_ptr<IntegralVolume> > rivs_occupancy;
        std::vector<boost::shared_ptr<IntegralVolume> > rivs_occupancy_complete_;
        std::vector<boost::shared_ptr<IntegralVolume> > rivs_occluded;
        std::vector<boost::shared_ptr<IntegralVolume> > rivs_full;
        //std::vector<float *> edges_heat_maps;
        std::vector<std::vector<boost::shared_ptr<IntegralVolume> > > rivhistograms;

        int ycolor_size_;
        int ucolor_size_;
        int vcolor_size_;
        std::vector<std::vector<boost::shared_ptr<IntegralVolume> > > riv_color_histograms_; //yuv color histogram (2x4x4)
        std::vector<std::vector<boost::shared_ptr<IntegralVolume> > > riv_squared_color_histograms_;
        std::vector< boost::shared_ptr<IntegralVolume> > riv_points_color_histogram_;

        unsigned int max_label_;
        bool visualize_;
        int best_wins_;

        std::vector<BBox> final_boxes_;
        PointInTPtr cloud_for_iv_;

        pcl::PointCloud<pcl::PointXYZL>::Ptr smooth_labels_cloud_;
        unsigned int table_plane_label_;
        std::vector<std::vector<int> > npoints_label_;

        Eigen::Matrix4f transform_to_plane;

        bool do_cuda_;

        float vpx_, vpy_, vpz_;

        void
        transformToBeCenteredOnPlane (Eigen::Vector4f & plane_, Eigen::Matrix4f & inv);

        void
        nonMaximaSupression (const std::vector<BBox> & rectangles, int n_rects, std::vector<BBox> & maximas);

        inline void getInsideBox(BBox & bb, PointInTPtr & cloud, std::vector<int> & inside_box) {
          int v = bb.angle;
          Eigen::Affine3f incr_rot_trans;
          incr_rot_trans.setIdentity ();

          Eigen::Vector4f minxyz, maxxyz;
          minxyz[0] = min_x + (bb.x) * resolution - resolution / 2.f;
          minxyz[1] = min_y + (bb.y) * resolution - resolution / 2.f;
          minxyz[2] = min_z + (bb.z) * resolution - resolution / 2.f;
          minxyz[3] = 1.f;

          maxxyz[0] = min_x + (bb.sx + bb.x) * resolution + resolution / 2.f;
          maxxyz[1] = min_y + (bb.sy + bb.y) * resolution + resolution / 2.f;
          maxxyz[2] = min_z + (bb.sz + bb.z) * resolution + resolution / 2.f;
          maxxyz[3] = 1.f;

          /*Eigen::Vector4f minxyz, maxxyz;
          minxyz[0] = min_x + bb.x * resolution;
          minxyz[1] = min_y + bb.y * resolution;
          minxyz[2] = min_z + bb.z * resolution;
          minxyz[3] = 1.f;

          maxxyz[0] = min_x + (bb.sx + bb.x) * resolution;
          maxxyz[1] = min_y + (bb.sy + bb.y) * resolution;
          maxxyz[2] = min_z + (bb.sz + bb.z) * resolution;
          maxxyz[3] = 1.f;*/

          if (v != 0)
          {
            float rot_rads = pcl::deg2rad (static_cast<float> (angle_incr_ * v));
            incr_rot_trans = Eigen::Affine3f (Eigen::AngleAxisf (static_cast<float> (rot_rads), Eigen::Vector3f::UnitZ ()));
          }

          {
            pcl::CropBox<PointInT> cb;
            cb.setInputCloud (cloud);
            cb.setMin (minxyz);
            cb.setMax (maxxyz);
            cb.setTransform (incr_rot_trans);
            cb.filter (inside_box);
          }
        }

        void printSomeObjectnessValues(BBox & box);

        void
        nonMaximaSupressionExplainedPoints (const std::vector<BBox> & rectangles, int n_rects, std::vector<BBox> & maximas, PointInTPtr & cloud_for_iv, bool fine=false);

        void
        generateBoundingBoxes (std::vector<BBox> & bounding_boxes);

        void
        generateBoundingBoxesExhaustiveSearch (std::vector<BBox> & bounding_boxes, int zs = -1, int zlimit = -1);

        void
        evaluateBoundingBoxes (std::vector<BBox> & bounding_boxes, bool print_objectness = false);

        void
        visBBox (pcl::visualization::PCLVisualizer & vis, BBox & bb, std::stringstream & name, int viewport=0);

        void
        visBBox (pcl::visualization::PCLVisualizer & vis, BBox & bb, std::stringstream & name, Eigen::Matrix4f & matrix_trans, int viewport=0);

        void
        visualizeBoundingBoxes (PointInTPtr & cloud_for_iv, std::vector<BBox> & bounding_boxes, Eigen::Matrix4f & transform_to_plane,
                                bool vis_shrinked = false, bool vis_extended = false);

        void
        optimizeBoundingBoxes (PointInTPtr & cloud_for_iv, std::vector<BBox> & bounding_boxes, Eigen::Matrix4f & transform_to_plane);

        int
        getValueByFaces (BBox * bb, boost::shared_ptr<IntegralVolume> & iv)
        {
          int face_edges[6];
          iv->getRectangleFromCorner (bb->x, bb->y, bb->z, 1, bb->sy, bb->sz, face_edges[0]);
          iv->getRectangleFromCorner (bb->x, bb->y, bb->z, bb->sx, 1, bb->sz, face_edges[1]);
          iv->getRectangleFromCorner (bb->x, bb->y, bb->z, bb->sx, bb->sy, 1, face_edges[2]);
          iv->getRectangleFromCorner (bb->x, bb->y, bb->z + bb->sz, bb->sx, bb->sy, 1, face_edges[3]);
          iv->getRectangleFromCorner (bb->x, bb->y + bb->sy, bb->z, bb->sx, 1, bb->sz, face_edges[4]);
          iv->getRectangleFromCorner (bb->x + bb->sx, bb->y, bb->z, 1, bb->sy, bb->sz, face_edges[5]);

          int sum_faces = 0;
          for (size_t kk = 0; kk < 6; kk++)
            sum_faces += face_edges[kk];

          return sum_faces;
        }

        int
        getValueByFaces (BBox * bb, boost::shared_ptr<IntegralVolume> & iv, bool * visible_faces)
        {
          int face_edges[6];
          int sum_faces = 0;

          iv->getRectangleFromCorner (bb->x, bb->y, bb->z, 1, bb->sy, bb->sz, face_edges[0]);
          iv->getRectangleFromCorner (bb->x, bb->y, bb->z, bb->sx, 1, bb->sz, face_edges[1]);
          iv->getRectangleFromCorner (bb->x, bb->y, bb->z, bb->sx, bb->sy, 1, face_edges[2]);
          iv->getRectangleFromCorner (bb->x, bb->y, bb->z + bb->sz, bb->sx, bb->sy, 1, face_edges[3]);
          iv->getRectangleFromCorner (bb->x, bb->y + bb->sy, bb->z, bb->sx, 1, bb->sz, face_edges[4]);
          iv->getRectangleFromCorner (bb->x + bb->sx, bb->y, bb->z, 1, bb->sy, bb->sz, face_edges[5]);

          for (size_t kk = 0; kk < 6; kk++)
          {
            if (visible_faces[kk])
            {
              sum_faces += face_edges[kk];
            }
          }

          return sum_faces;
        }

        int
        getValueByEdges (BBox * bb, boost::shared_ptr<IntegralVolume> & iv)
        {
          int rectangle_edges[12];
          iv->getRectangleFromCorner (bb->x, bb->y, bb->z, 1, bb->sy, 1, rectangle_edges[0]); //bottom side
          iv->getRectangleFromCorner (bb->x, bb->y, bb->z, bb->sx, 1, 1, rectangle_edges[1]); //bottom side
          iv->getRectangleFromCorner (bb->x + bb->sx, bb->y, bb->z, 1, bb->sy, 1, rectangle_edges[2]); //bottom side
          iv->getRectangleFromCorner (bb->x, bb->y + bb->sy, bb->z, bb->sx, 1, 1, rectangle_edges[3]); //bottom side
          iv->getRectangleFromCorner (bb->x, bb->y, bb->z + bb->sz, 1, bb->sy, 1, rectangle_edges[4]); //top side
          iv->getRectangleFromCorner (bb->x, bb->y, bb->z + bb->sz, bb->sx, 1, 1, rectangle_edges[5]); //top side
          iv->getRectangleFromCorner (bb->x + bb->sx, bb->y, bb->z + bb->sz, 1, bb->sy, 1, rectangle_edges[6]); //top side
          iv->getRectangleFromCorner (bb->x, bb->y + bb->sy, bb->z + bb->sz, bb->sx, 1, 1, rectangle_edges[7]); //top side
          iv->getRectangleFromCorner (bb->x, bb->y, bb->z, 1, 1, bb->sz, rectangle_edges[8]); //other edges
          iv->getRectangleFromCorner (bb->x + bb->sx, bb->y, bb->z, 1, 1, bb->sz, rectangle_edges[9]); //other edges
          iv->getRectangleFromCorner (bb->x, bb->y + bb->sy, bb->z, 1, 1, bb->sz, rectangle_edges[10]); //other edges
          iv->getRectangleFromCorner (bb->x + bb->sx, bb->y + bb->sy, bb->z, 1, 1, bb->sz, rectangle_edges[11]); //other edges

          int sum_edges = 0;
          for (size_t kk = 0; kk < 12; kk++)
            sum_edges += rectangle_edges[kk];

          return sum_edges;
        }

        Eigen::Vector3f
        BBoxCenter (const BBox & bb)
        {

          Eigen::Vector3f center;

          Eigen::Vector4f minxyz, maxxyz;
          minxyz[0] = min_x + bb.x * resolution;
          minxyz[1] = min_y + bb.y * resolution;
          minxyz[2] = min_z + bb.z * resolution;
          minxyz[3] = 0.f;

          maxxyz[0] = min_x + (bb.sx + bb.x) * resolution;
          maxxyz[1] = min_y + (bb.sy + bb.y) * resolution;
          maxxyz[2] = min_z + (bb.sz + bb.z) * resolution;
          maxxyz[3] = 0.f;

          if (bb.angle != 0)
          {
            float rot_angle = pcl::deg2rad (static_cast<float> (bb.angle * angle_incr_ * -1.f));
            Eigen::Affine3f rot_trans = Eigen::Affine3f (Eigen::AngleAxisf (static_cast<float> (rot_angle), Eigen::Vector3f::UnitZ ()));

            minxyz = rot_trans * minxyz;
            maxxyz = rot_trans * maxxyz;
          }

          center = Eigen::Vector3f ((minxyz[0] + maxxyz[0]) / 2.f, (minxyz[1] + maxxyz[1]) / 2.f, (minxyz[2] + maxxyz[2]) / 2.f);
          return center;
        }

        void
        createRotatedIV (PointInTPtr & cloud_for_iv, PointInTPtr & occluded_cloud_transformed_back
        /*std::vector<boost::shared_ptr<IntegralVolume> > & rivs, std::vector<boost::shared_ptr<IntegralVolume> > & rivs_occupancy,
         std::vector<boost::shared_ptr<IntegralVolume> > & rivs_occluded, std::vector<float *> & edges_heat_maps*/);

        void
        shrink_bbox (BBox * bb, BBox & bb_shrinked)
        {
          bb_shrinked.sx = std::max(std::min (std::max (static_cast<int> (std::floor (bb->sx * shrink_factor_x)), 1), bb->sx - 2),1);
          bb_shrinked.sy = std::max(std::min (std::max (static_cast<int> (std::floor (bb->sy * shrink_factor_y)), 1), bb->sy - 2),1);
          bb_shrinked.sz = std::max(std::min (std::max (static_cast<int> (std::floor (bb->sz * shrink_factor_z)), 1), bb->sz - 2),1);

          bb_shrinked.x = bb->x + std::max (static_cast<int> (std::floor ((bb->sx - bb_shrinked.sx) / 2.f)), 1);
          bb_shrinked.y = bb->y + std::max (static_cast<int> (std::floor ((bb->sy - bb_shrinked.sy) / 2.f)), 1);
          bb_shrinked.z = bb->z + std::max (static_cast<int> (std::floor ((bb->sz - bb_shrinked.sz) / 2.f)), 1);
        }

        void
        shrink_bbox (BBox & bb, BBox & bb_shrinked)
        {
          shrink_bbox (&bb, bb_shrinked);
        }

      public:

        Objectness3D (std::string & sf, int nsw, int nr, int minsw, int maxsw, int angle_incr = 90, float max_thres = 0.75f);

        void
        setVisualize (bool b)
        {
          visualize_ = b;
        }

        void
        setBestWins(int n) {
          best_wins_ = n;
        }

        void setOptType(int t) {
          opt_type_ = t;
        }

        void
       setViewpoint(float vx, float vy, float vz) {
         vpx_ = vx;
         vpy_ = vy;
         vpz_ = vz;
       }

        void
        setSmoothLabelsCloud (pcl::PointCloud<pcl::PointXYZL>::Ptr & label_cloud)
        {
          smooth_labels_cloud_ = label_cloud;
        }

        void
        setDoOptimize (bool b)
        {
          do_optimize_ = b;
        }

        void
        setEdgeLabelsCloud (pcl::PointCloud<pcl::PointXYZL>::Ptr & label_cloud)
        {
          label_cloud_ = label_cloud;
        }

        void
        setEdges (std::vector<int> & edges)
        {
          edges_ = edges;
        }

        void
        setExpandFactor (float f)
        {
          expand_factor_ = f;
        }

        void
        doZ (bool b)
        {
          do_z_ = b;
        }

        void
        setInputCloud (PointInTPtr & cloud)
        {
          input_cloud = cloud;
        }

        void
        setTablePlane (Eigen::Vector4f & plane)
        {
          table_plane_ = plane;
          table_plane_set_ = true;
        }

        void setDoCuda(bool doit) {
          do_cuda_ = doit;
        }

        void
        computeObjectness (bool compute_voi = false);

        void
        getObjectIndices (std::vector<pcl::PointIndices> & indices, PointInTPtr & cloud);

        void
        visualizeBoundingBoxes (pcl::visualization::PCLVisualizer & vis)
        {
          for (size_t i = 0; i < final_boxes_.size (); i++)
          {
            std::stringstream name;
            name << "box_" << i;
            BBox bb = final_boxes_[i];
            Eigen::Matrix4f inv = transform_to_plane.inverse();
            visBBox (vis, bb, name, inv);
          }
        }
      };
  }
}
#endif /* OBJECTNESS_3D_H_ */
