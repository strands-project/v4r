/******************************************************************************
 * Copyright (c) 2012 Aitor Aldoma
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/

#ifndef V4R_ZBUFFERING_H_
#define V4R_ZBUFFERING_H_

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/common/io.h>
#include <v4r/core/macros.h>


namespace v4r
{
    /**
     * \brief Class to reason about occlusions
     * \author Thomas Faeulhammer, Aitor Aldoma
     */
    template<typename PointT>
      class V4R_EXPORTS ZBuffering
      {
      public:
          class Parameter
          {
          public:
              float f_;
              int width_, height_;
              int u_margin_, v_margin_;
              bool compute_focal_length_;
              bool do_smoothing_;
              float inlier_threshold_;
              int smoothing_radius_;
              bool force_unorganized_; ///@brief re-projects points by given camera intrinsics even if point cloud is already organized
              Parameter(
                      float f=525.f,
                      int width = 640,
                      int height = 480,
                      int u_margin = 0,
                      int v_margin = 0,
                      bool compute_focal_length = false,
                      bool do_smoothing = true,
                      float inlier_threshold = 0.01f,
                      int smoothing_radius = 1,
                      bool force_unorganized = false) :
                  f_(f), width_ (width), height_(height), u_margin_(u_margin), v_margin_(v_margin),
                  compute_focal_length_(compute_focal_length), do_smoothing_(do_smoothing), inlier_threshold_(inlier_threshold),
                  smoothing_radius_(smoothing_radius), force_unorganized_ (force_unorganized)
              {}
          }param_;

      private:
        std::vector<float> depth_;
        std::vector<int> kept_indices_;
        boost::shared_ptr<std::vector<int> > indices_map_;  /// @brief saves for each pixel which indices of the input cloud it represents. Non-occupied pixels are labelled with index -1.

      public:

        ZBuffering (const Parameter &p=Parameter()) : param_(p) { }

        void computeDepthMap (const pcl::PointCloud<PointT> &scene);

        void computeDepthMap (const pcl::PointCloud<PointT> &scene, Eigen::MatrixXf &depth_image, std::vector<int> &visible_indices);

        void renderPointCloud(const typename pcl::PointCloud<PointT> &cloud, typename pcl::PointCloud<PointT> & rendered_view);

        void filter (const typename pcl::PointCloud<PointT> & model, typename pcl::PointCloud<PointT> & filtered);

        void filter (const typename pcl::PointCloud<PointT> & model, std::vector<int> & indices);

        static void erode(const Eigen::MatrixXf &input, Eigen::MatrixXf &output, int erosion_size = 3, int erosion_elem=0);

        boost::shared_ptr<std::vector<int> > getIndicesMap() const
        {
            return indices_map_;
        }

        void getKeptIndices(std::vector<int> &indices) const
        {
            indices = kept_indices_;
        }

        std::vector<float> getDepthMap() const
        {
            return depth_;
        }
      };

      template<typename SceneT, typename ModelT>
      std::vector<bool>
      computeOccludedPoints (const pcl::PointCloud<SceneT> & organized_cloud,
                             const pcl::PointCloud<ModelT> & to_be_filtered,
                             float f = 525.f,
                             float threshold = 0.01f,
                             bool is_occluded_out_fov = true)
      {
        const float cx = (static_cast<float> (organized_cloud.width) / 2.f - 0.5f);
        const float cy = (static_cast<float> (organized_cloud.height) / 2.f - 0.5f);
        std::vector<bool> is_occluded (to_be_filtered.points.size(), false);

        for (size_t i = 0; i < to_be_filtered.points.size (); i++)
        {
          if ( !pcl::isFinite(to_be_filtered.points[i]) )
               continue;

          const float x = to_be_filtered.points[i].x;
          const float y = to_be_filtered.points[i].y;
          const float z = to_be_filtered.points[i].z;
          const int u = static_cast<int> (f * x / z + cx);
          const int v = static_cast<int> (f * y / z + cy);

          // points out of the field of view in the first frame
          if ((u >= static_cast<int> (organized_cloud.width)) || (v >= static_cast<int> (organized_cloud.height)) || (u < 0) || (v < 0))
          {
              is_occluded[i] = is_occluded_out_fov;
              continue;
          }

          // Check for invalid depth
          if ( !pcl::isFinite (organized_cloud.at (u, v)) )
          {
              is_occluded[i] = true;
              continue;
          }


          //Check if point depth (distance to camera) is greater than the (u,v)
          if ( ( z - organized_cloud.at(u, v).z ) > threshold)
              is_occluded[i] = true;
        }
        return is_occluded;
      }



      template<typename SceneT, typename ModelT>
      std::vector<bool>
      computeOccludedPoints (const pcl::PointCloud<SceneT> & organized_cloud,
                             const pcl::PointCloud<ModelT> & to_be_filtered,
                             const Eigen::Matrix4f &transform_2to1,
                             float f = 525.f,
                             float threshold = 0.01f,
                             bool is_occluded_out_fov = true)
      {
          typename pcl::PointCloud<ModelT> cloud_trans;
          pcl::transformPointCloud(to_be_filtered, cloud_trans, transform_2to1);
          return computeOccludedPoints(organized_cloud, cloud_trans, f, threshold, is_occluded_out_fov);
      }


    template<typename ModelT, typename SceneT>
    typename pcl::PointCloud<ModelT>::Ptr
    filter (const typename pcl::PointCloud<SceneT> & organized_cloud,
            const typename pcl::PointCloud<ModelT> & to_be_filtered,
            float f = 525.f,    // Kinect_v1 focal length
            float threshold = 0.01f)
    {
      const float cx = (static_cast<float> (organized_cloud.width) / 2.f - 0.5f);
      const float cy = (static_cast<float> (organized_cloud.height) / 2.f - 0.5f);
      typename pcl::PointCloud<ModelT>::Ptr filtered (new pcl::PointCloud<ModelT>);

      std::vector<int> indices_to_keep;
      indices_to_keep.reserve(to_be_filtered.points.size ());

      for (size_t i = 0; i < to_be_filtered.points.size (); i++)
      {
        const float x = to_be_filtered.points[i].x;
        const float y = to_be_filtered.points[i].y;
        const float z = to_be_filtered.points[i].z;
        const int u = static_cast<int> (f * x / z + cx);
        const int v = static_cast<int> (f * y / z + cy);

        //Not out of bounds
        if ((u >= static_cast<int> (organized_cloud.width)) || (v >= static_cast<int> (organized_cloud.height)) || (u < 0) || (v < 0))
          continue;

        //Check for invalid depth
        if (!pcl::isFinite (organized_cloud.at (u, v)))
          continue;

        float z_oc = organized_cloud.at (u, v).z;

        //Check if point depth (distance to camera) is greater than the (u,v)
        if ((z - z_oc) > threshold)
          continue;

        indices_to_keep.push_back(static_cast<int> (i));
      }

      std::vector<int>(indices_to_keep.begin(), indices_to_keep.end()).swap(indices_to_keep);

      pcl::copyPointCloud (*to_be_filtered, indices_to_keep, *filtered);
      return filtered;
    }

      /**
     * @brief filters points which are not visible with respect to an organized reference cloud
     * @param organized_cloud: reference cloud used to compare
     * @param to_be_filtered: cloud to be filtered with respect to the reference cloud
     * @param f focal length used for back-projection of points
     * @param threshold all points further away from the reference point than this threshold will be filtered
     * @param indices_to_keep indices of the points which are passed
     * @return filtered cloud
     */
    template<typename ModelT, typename SceneT>
      typename pcl::PointCloud<ModelT>::Ptr
    filter (const typename pcl::PointCloud<SceneT> & organized_cloud,
            const typename pcl::PointCloud<ModelT> & to_be_filtered,
            float f,
            float threshold,
            std::vector<int> & indices_to_keep)
    {
      float cx = (static_cast<float> (organized_cloud.width) / 2.f - 0.5f);
      float cy = (static_cast<float> (organized_cloud.height) / 2.f - 0.5f);
      typename pcl::PointCloud<ModelT>::Ptr filtered (new pcl::PointCloud<ModelT>);

      //std::vector<int> indices_to_keep;
      indices_to_keep.resize (to_be_filtered.points.size ());

      pcl::PointCloud<float> filtered_points_depth;
      pcl::PointCloud<int> closest_idx_points;
      filtered_points_depth.points.resize (organized_cloud.points.size ());
      closest_idx_points.points.resize (organized_cloud.points.size ());

      filtered_points_depth.width = closest_idx_points.width = organized_cloud.width;
      filtered_points_depth.height = closest_idx_points.height = organized_cloud.height;
      for (size_t i = 0; i < filtered_points_depth.points.size (); i++)
      {
        filtered_points_depth.points[i] = std::numeric_limits<float>::quiet_NaN ();
        closest_idx_points.points[i] = -1;
      }

      int keep = 0;
      for (size_t i = 0; i < to_be_filtered.points.size (); i++)
      {
        float x = to_be_filtered.points[i].x;
        float y = to_be_filtered.points[i].y;
        float z = to_be_filtered.points[i].z;
        int u = static_cast<int> (f * x / z + cx);
        int v = static_cast<int> (f * y / z + cy);

        //Not out of bounds
        if ((u >= static_cast<int> (organized_cloud.width)) || (v >= static_cast<int> (organized_cloud.height)) || (u < 0) || (v < 0))
          continue;

        //Check for invalid depth
        if (!pcl::isFinite (organized_cloud.at (u, v)))
          continue;

        float z_oc = organized_cloud.at (u, v).z;

        //Check if point depth (distance to camera) is greater than the (u,v)
        if ((z - z_oc) > threshold)
          continue;

        if (pcl_isnan(filtered_points_depth.at (u, v)) || (z < filtered_points_depth.at (u, v)))
        {
          closest_idx_points.at (u, v) = static_cast<int> (i);
          filtered_points_depth.at (u, v) = z;
        }

        //indices_to_keep[keep] = static_cast<int> (i);
        //keep++;
      }

      for (size_t i = 0; i < closest_idx_points.points.size (); i++)
      {
        if(closest_idx_points[i] != -1)
        {
          indices_to_keep[keep] = closest_idx_points[i];
          keep++;
        }
      }

      indices_to_keep.resize (keep);
      pcl::copyPointCloud (to_be_filtered, indices_to_keep, *filtered);
      return filtered;
    }

    template<typename ModelT, typename SceneT>
      typename pcl::PointCloud<ModelT>::Ptr
    filter (const typename pcl::PointCloud<SceneT> & organized_cloud,
            const typename pcl::PointCloud<ModelT> & to_be_filtered,
            float f,
            float threshold,
            bool check_invalid_depth = true)
    {
      float cx = (static_cast<float> (organized_cloud.width) / 2.f - 0.5f);
      float cy = (static_cast<float> (organized_cloud.height) / 2.f - 0.5f);
      typename pcl::PointCloud<ModelT>::Ptr filtered (new pcl::PointCloud<ModelT>);

      std::vector<int> indices_to_keep;
      indices_to_keep.resize (to_be_filtered.points.size ());

      int keep = 0;
      for (size_t i = 0; i < to_be_filtered.points.size (); i++)
      {
        float x = to_be_filtered.points[i].x;
        float y = to_be_filtered.points[i].y;
        float z = to_be_filtered.points[i].z;
        int u = static_cast<int> (f * x / z + cx);
        int v = static_cast<int> (f * y / z + cy);

        //Not out of bounds
        if ((u >= static_cast<int> (organized_cloud.width)) || (v >= static_cast<int> (organized_cloud.height)) || (u < 0) || (v < 0))
          continue;

        //Check for invalid depth
        if (check_invalid_depth && !pcl::isFinite (organized_cloud.at (u, v)))
            continue;

        float z_oc = organized_cloud.at (u, v).z;

        //Check if point depth (distance to camera) is greater than the (u,v)
        if ((z - z_oc) > threshold)
          continue;

        indices_to_keep[keep] = static_cast<int> (i);
        keep++;
      }

      indices_to_keep.resize (keep);
      pcl::copyPointCloud (to_be_filtered, indices_to_keep, *filtered);
      return filtered;
    }

    template<typename ModelT, typename SceneT>
      typename pcl::PointCloud<ModelT>::Ptr
    getOccludedCloud (const pcl::PointCloud<SceneT> & organized_cloud,
                      const pcl::PointCloud<ModelT> & to_be_filtered,
                      float f,
                      float threshold,
                      bool check_invalid_depth = true)
    {
      float cx = static_cast<float> (organized_cloud.width) / 2.f - 0.5f;
      float cy = static_cast<float> (organized_cloud.height) / 2.f - 0.5f;
      typename pcl::PointCloud<ModelT>::Ptr filtered (new pcl::PointCloud<ModelT>);

      std::vector<int> indices_to_keep;
      indices_to_keep.resize (to_be_filtered.points.size ());

      int keep = 0;
      for (size_t i = 0; i < to_be_filtered.points.size (); i++)
      {
        float x = to_be_filtered.points[i].x;
        float y = to_be_filtered.points[i].y;
        float z = to_be_filtered.points[i].z;
        int u = static_cast<int> (f * x / z + cx);
        int v = static_cast<int> (f * y / z + cy);

        //Out of bounds
        if ((u >= static_cast<int> (organized_cloud.width)) || (v >= static_cast<int> (organized_cloud.height)) || (u < 0) || (v < 0))
            continue;

        //Check for invalid depth
        if (check_invalid_depth && !pcl::isFinite (organized_cloud.at(u, v)))
            continue;

        float z_oc = organized_cloud.at (u, v).z;

        //Check if point depth (distance to camera) is greater than the (u,v)
        if ((z - z_oc) > threshold)
        {
          indices_to_keep[keep] = static_cast<int> (i);
          keep++;
        }
      }

      indices_to_keep.resize (keep);
      pcl::copyPointCloud (to_be_filtered, indices_to_keep, *filtered);
      return filtered;
    }
}

#endif
