/*
 * objectness_3D_cuda.cu
 *
 *  Created on: Nov 8, 2012
 *      Author: aitor
 */

#include "faat_pcl/segmentation/objectness_3d/cuda/cuda_objectness_3D.h"
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/extrema.h>

#include "valid_indices.h"

namespace faat_pcl
{
  namespace cuda
  {
    namespace segmentation
    {

      __host__ __device__
      int
      getColorHistogram(int bb_x, int bb_y, int bb_z, int bb_sx, int bb_sy, int bb_sz,
                          thrust::device_ptr<int> riv_points_color_histogram_,
                          thrust::device_ptr<int> riv_color_histograms_,
                          int m_width, int m_height, int m_depth, float * color_hist, int size_color_hist) {

        int npoints_color = getRectangleFromCorner( riv_points_color_histogram_, bb_x, bb_y, bb_z,
                                                    bb_sx, bb_sy, bb_sz, m_width, m_height);
        //int colors[3];
        //int colors_squared_sum[3];
        for(int j=0; j < size_color_hist; j++) {

          int idx_color = j * m_width * m_height * m_depth;
          color_hist[j] = getRectangleFromCorner (riv_color_histograms_ + idx_color, bb_x, bb_y, bb_z,
                                              bb_sx, bb_sy, bb_sz,
                                              m_width, m_height);

          /*color_hist[j] /= (float)(npoints_color);*/
          color_hist[j] /= 255.f;

          /*colors_squared_sum[j] = getRectangleFromCorner (riv_squared_color_histograms_ + (idx_vol * 3) + idx_color, getX(idx), getY(idx), getZ(idx), getSX(idx), getSY(idx), getSZ(idx),
                                                       m_width, m_height);
          colors_squared_sum[j] /= (float)(npoints_color);*/
        }

        /*int colors_variance[3];
        for(int j=0; j < 3; j++) {
          colors_variance[j] = colors_squared_sum[j] - colors[j]*colors[j];
        }*/

        return npoints_color;
      }

      __host__ __device__
      int
      getValueByFaces (int bb_x, int bb_y, int bb_z, int bb_sx, int bb_sy, int bb_sz, thrust::device_ptr<int> vol_, int m_width, int m_height, bool * visible_faces)
      {
        int sum = 0;
        if (visible_faces[0])
          sum += getRectangleFromCorner (vol_, bb_x, bb_y, bb_z, 1, bb_sy, bb_sz, m_width, m_height);
        if (visible_faces[1])
          sum += getRectangleFromCorner (vol_, bb_x, bb_y, bb_z, bb_sx, 1, bb_sz, m_width, m_height);
        if (visible_faces[2])
          sum += getRectangleFromCorner (vol_, bb_x, bb_y, bb_z, bb_sx, bb_sy, 1, m_width, m_height);
        if (visible_faces[3])
          sum += getRectangleFromCorner (vol_, bb_x, bb_y, bb_z + bb_sz, bb_sx, bb_sy, 1, m_width, m_height);
        if (visible_faces[4])
          sum += getRectangleFromCorner (vol_, bb_x, bb_y + bb_sy, bb_z, bb_sx, 1, bb_sz, m_width, m_height);
        if (visible_faces[5])
          sum += getRectangleFromCorner (vol_, bb_x + bb_sx, bb_y, bb_z, 1, bb_sy, bb_sz, m_width, m_height);

        return sum;
        /*return getRectangleFromCorner (vol_, bb_x, bb_y, bb_z, 1, bb_sy, bb_sz, m_width, m_height) +
            getRectangleFromCorner (vol_, bb_x, bb_y, bb_z, bb_sx, 1, bb_sz, m_width, m_height) +
            getRectangleFromCorner (vol_, bb_x, bb_y, bb_z, bb_sx, bb_sy, 1, m_width, m_height) +
            getRectangleFromCorner (vol_, bb_x, bb_y, bb_z + bb_sz, bb_sx, bb_sy, 1, m_width, m_height) +
            getRectangleFromCorner (vol_, bb_x, bb_y + bb_sy, bb_z, bb_sx, 1, bb_sz, m_width, m_height) +
            getRectangleFromCorner (vol_, bb_x + bb_sx, bb_y, bb_z, 1, bb_sy, bb_sz, m_width, m_height);*/
      }

      __host__ __device__
      int
      getValueByFacesInternal (int bb_x, int bb_y, int bb_z, int bb_sx, int bb_sy, int bb_sz, thrust::device_ptr<int> vol_, int m_width, int m_height, bool * visible_faces)
      {
        int sum = 0;
        if (visible_faces[0])
          sum += getRectangleFromCorner (vol_, bb_x + 1, bb_y, bb_z, 1, bb_sy, bb_sz, m_width, m_height);
        if (visible_faces[1])
          sum += getRectangleFromCorner (vol_, bb_x, bb_y + 1, bb_z, bb_sx, 1, bb_sz, m_width, m_height);
        if (visible_faces[2])
          sum += getRectangleFromCorner (vol_, bb_x, bb_y, bb_z, bb_sx, bb_sy, 1, m_width, m_height);
        if (visible_faces[3])
          sum += getRectangleFromCorner (vol_, bb_x, bb_y, bb_z + bb_sz - 1, bb_sx, bb_sy, 1, m_width, m_height);
        if (visible_faces[4])
          sum += getRectangleFromCorner (vol_, bb_x, bb_y + bb_sy - 1, bb_z, bb_sx, 1, bb_sz, m_width, m_height);
        if (visible_faces[5])
          sum += getRectangleFromCorner (vol_, bb_x + bb_sx - 1, bb_y, bb_z, 1, bb_sy, bb_sz, m_width, m_height);

        return sum;
      }

      __host__ __device__
      void
      getValueByFaces (int bb_x, int bb_y, int bb_z, int bb_sx, int bb_sy, int bb_sz,
                         thrust::device_ptr<int> vol_, int m_width, int m_height, int * values, bool * visible_faces)
      {
        if (visible_faces[0])
          values[0] = getRectangleFromCorner (vol_, bb_x, bb_y, bb_z, 1, bb_sy, bb_sz, m_width, m_height);
        if (visible_faces[1])
          values[1] = getRectangleFromCorner (vol_, bb_x, bb_y, bb_z, bb_sx, 1, bb_sz, m_width, m_height);
        if (visible_faces[2])
          values[2] = getRectangleFromCorner (vol_, bb_x, bb_y, bb_z, bb_sx, bb_sy, 1, m_width, m_height);
        if (visible_faces[3])
          values[3] = getRectangleFromCorner (vol_, bb_x, bb_y, bb_z + bb_sz, bb_sx, bb_sy, 1, m_width, m_height);
        if (visible_faces[4])
          values[4] = getRectangleFromCorner (vol_, bb_x, bb_y + bb_sy, bb_z, bb_sx, 1, bb_sz, m_width, m_height);
        if (visible_faces[5])
          values[5] = getRectangleFromCorner (vol_, bb_x + bb_sx, bb_y, bb_z, 1, bb_sy, bb_sz, m_width, m_height);
      }

      /*__device__
      void
      getValueByFaces (int bb_x, int bb_y, int bb_z, int bb_sx, int bb_sy, int bb_sz,
                         thrust::device_ptr<int> vol_, int m_width, int m_height, int * values, bool * visible_faces)
      {
        if (visible_faces[0])
          values[0] = getRectangleFromCorner (vol_, bb_x, bb_y, bb_z, 1, bb_sy, bb_sz, m_width, m_height);
        if (visible_faces[1])
          values[1] = getRectangleFromCorner (vol_, bb_x, bb_y, bb_z, bb_sx, 1, bb_sz, m_width, m_height);
        if (visible_faces[2])
          values[2] = getRectangleFromCorner (vol_, bb_x, bb_y, bb_z, bb_sx, bb_sy, 1, m_width, m_height);
        if (visible_faces[3])
          values[3] = getRectangleFromCorner (vol_, bb_x, bb_y, bb_z + bb_sz - 1, bb_sx, bb_sy, 1, m_width, m_height);
        if (visible_faces[4])
          values[4] = getRectangleFromCorner (vol_, bb_x, bb_y + bb_sy - 1, bb_z, bb_sx, 1, bb_sz, m_width, m_height);
        if (visible_faces[5])
          values[5] = getRectangleFromCorner (vol_, bb_x + bb_sx - 1, bb_y, bb_z, 1, bb_sy, bb_sz, m_width, m_height);
      }*/

      struct Mat3f {
        float mat[3][3];
      };

      struct Vector3f {
        float x,y,z;
        __host__ __device__
        Vector3f(float _x, float _y, float _z) {
          x = _x;
          y = _y;
          z = _z;
        }

        __host__ __device__
        Vector3f() {
          x = 0.f;
          y = 0.f;
          z = 0.f;
        }

        __host__ __device__
        float
        selfDot() {
          return x * x + y * y + z * z;
        }
        __host__ __device__
        void normalize() {
          float invLen = 1.0f / sqrtf(selfDot());
          x *= invLen;
          y *= invLen;
          z *= invLen;
        }
      };

      inline __host__ __device__ Vector3f operator-(Vector3f a, Vector3f b)
      {
          return Vector3f(a.x - b.x, a.y - b.y, a.z - b.z);
      }

      inline __host__ __device__ Vector3f operator+(Vector3f a, Vector3f b)
      {
          return Vector3f(a.x + b.x, a.y + b.y, a.z + b.z);
      }

      inline __host__ __device__ Vector3f operator/(Vector3f a, float b)
      {
          return Vector3f(a.x / b, a.y / b, a.z / b);
      }
      // dot product
      inline __host__ __device__ float dot(Vector3f a, Vector3f b)
      {
          return a.x * b.x + a.y * b.y + a.z * b.z;
      }

      inline __host__ __device__ Vector3f normalize (Vector3f a)
      {
        float invLen = 1.0f / sqrtf(dot(a, a));
        return Vector3f(a.x * invLen, a.y * invLen, a.z * invLen);
      }

      // cross product
      inline __host__  __device__ Vector3f cross(Vector3f a, Vector3f b)
      {
          return Vector3f(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
      }

      // rotate vector
      inline __host__ __device__ Vector3f rotate(Mat3f rot_, Vector3f pt)
      {
        Vector3f min_tp;
        min_tp.x = min_tp.y = min_tp.z = 0;
        min_tp.x += rot_.mat[0][0] * pt.x + rot_.mat[0][1] * pt.y + rot_.mat[0][2] * pt.z;
        min_tp.y += rot_.mat[1][0] * pt.x + rot_.mat[1][1] * pt.y + rot_.mat[1][2] * pt.z;
        min_tp.z += rot_.mat[2][0] * pt.x + rot_.mat[2][1] * pt.y + rot_.mat[2][2] * pt.z;
        return min_tp;
      }


      inline __host__ __device__ Vector3f computeAndOrientNormal(Vector3f center, Vector3f center_face, Vector3f a, Vector3f b)
      {
        Vector3f face_centroid_to_center = center_face - center;
        face_centroid_to_center.normalize();
        Vector3f an = normalize(a);
        Vector3f bn = normalize(b);
        Vector3f normal = cross(an, bn);
        normal.normalize();

        if(dot(face_centroid_to_center, normal) < 0) {
          normal.x *= -1.f;
          normal.y *= -1.f;
          normal.z *= -1.f;
        }

        return normal;
      }

      struct generateAndEvaluateBox
      {
        thrust::device_ptr<int> rivs;
        thrust::device_ptr<int> rivs_occluded;
        thrust::device_ptr<int> rivs_occupancy;
        thrust::device_ptr<int> rivs_occupancy_complete;
        thrust::device_ptr<int> rivs_histograms;
        thrust::device_ptr<int> npoints_label_;
        thrust::device_ptr<int> riv_points_color_histogram_;
        thrust::device_ptr<int> riv_color_histograms_;
        thrust::device_ptr<int> rivs_full_;
        int m_width;
        int m_height;
        int m_depth;
        float expand_factor;
        int max_label;
        int min_size_w_;
        int max_size_w_;
        int step_x_;
        int step_y_;
        int step_z_;
        int step_sx_;
        int step_sy_;
        int step_sz_;

        int size_angle_;
        int size_x_;
        int size_y_;
        int size_sx_;
        int size_sy_;

        float shrink_factor_x;
        float shrink_factor_y;
        float shrink_factor_z;

        float vpx_, vpy_, vpz_;

        float min_z_;
        float max_z_;
        float min_x_;
        float max_x_;
        float min_y_;
        float max_y_;

        float resolution_;
        int angle_incr_;
        int table_plane_label_;

        int size_yuv_hist_;

        generateAndEvaluateBox (thrust::device_ptr<int> _rivs, thrust::device_ptr<int> _rivs_occluded, thrust::device_ptr<int> _rivs_occupancy,
                                thrust::device_ptr<int> _rivs_occupancy_complete, thrust::device_ptr<int> _rivs_histograms,
                                thrust::device_ptr<int> _npoints_label, thrust::device_ptr<int> riv_color_histograms,
                                thrust::device_ptr<int> riv_points_color_histogram, thrust::device_ptr<int> rivs_full, int _w, int _h, int _d, float _expand_factor, int _max_label,
                                int _min_size_w_, int _max_size_w_, int _step_x_, int _step_y_, int _step_z_, int _step_sx_, int _step_sy_,
                                int _step_sz_, float vpx, float vpy, float vpz,
                                float minx, float miny, float minz, float maxx, float maxy, float maxz,
                                float resolution, int angle_incr, int table_plane_label, int size_yuv_hist) :
          rivs (_rivs), rivs_occluded (_rivs_occluded), rivs_occupancy (_rivs_occupancy), rivs_occupancy_complete (_rivs_occupancy_complete),
              rivs_histograms (_rivs_histograms), npoints_label_ (_npoints_label),
              riv_color_histograms_(riv_color_histograms), riv_points_color_histogram_(riv_points_color_histogram), rivs_full_(rivs_full),
              m_width (_w), m_height (_h), m_depth (_d),
              expand_factor (_expand_factor), max_label (_max_label), min_size_w_ (_min_size_w_), max_size_w_ (_max_size_w_), step_x_ (_step_x_),
              step_y_ (_step_y_), step_z_ (_step_z_), step_sx_ (_step_sx_), step_sy_ (_step_sy_), step_sz_ (_step_sz_),
              vpx_(vpx), vpy_(vpy), vpz_(vpz), min_x_(minx), min_y_(miny), min_z_(minz), max_x_(maxx), max_y_(maxy), max_z_(maxz),
              resolution_(resolution), angle_incr_(angle_incr), table_plane_label_(table_plane_label), size_yuv_hist_(size_yuv_hist)
        {
          size_angle_ = ((m_width - min_size_w_) / step_x_) * ((m_height - min_size_w_) / step_y_) * ((max_size_w_ - min_size_w_) / step_sx_)
              * ((max_size_w_ - min_size_w_) / step_sy_) * ((std::min (max_size_w_ - min_size_w_, m_depth - min_size_w_)) / step_sz_);

          size_x_ = ((m_height - min_size_w_) / step_y_) * ((max_size_w_ - min_size_w_) / step_sx_) * ((max_size_w_ - min_size_w_) / step_sy_)
              * ((std::min (max_size_w_ - min_size_w_, m_depth - min_size_w_)) / step_sz_);

          size_y_ = ((max_size_w_ - min_size_w_) / step_sx_) * ((max_size_w_ - min_size_w_) / step_sy_) * ((std::min (max_size_w_ - min_size_w_,
                                                                                                                      m_depth - min_size_w_))
              / step_sz_);

          size_sx_ = ((max_size_w_ - min_size_w_) / step_sy_) * ((std::min (max_size_w_ - min_size_w_, m_depth - min_size_w_)) / step_sz_);

          size_sy_ = ((std::min (max_size_w_ - min_size_w_, m_depth - min_size_w_)) / step_sz_);

          shrink_factor_x = shrink_factor_y = shrink_factor_z = 0.5f;
        }

        __host__ __device__
        void
        shrink_bbox (int bb_x, int bb_y, int bb_z, int bb_sx, int bb_sy, int bb_sz, BBox * bb_shrinked)
        {
          bb_shrinked->sx = min (max (static_cast<int> (floor (bb_sx * shrink_factor_x)), 1), bb_sx - 2);
          bb_shrinked->sy = min (max (static_cast<int> (floor (bb_sy * shrink_factor_y)), 1), bb_sy - 2);
          bb_shrinked->sz = min (max (static_cast<int> (floor (bb_sz * shrink_factor_z)), 1), bb_sz - 2);

          bb_shrinked->x = bb_x + max (static_cast<int> (floor ((bb_sx - bb_shrinked->sx) / 2.f)), 1);
          bb_shrinked->y = bb_y + max (static_cast<int> (floor ((bb_sy - bb_shrinked->sy) / 2.f)), 1);
          bb_shrinked->z = bb_z + max (static_cast<int> (floor ((bb_sz - bb_shrinked->sz) / 2.f)), 1);
        }

        __host__ __device__
        int getAngle(const int idx) {
          return idx / size_angle_;
        }

        __host__ __device__
        int getX(const int idx) {
          return ((idx % size_angle_) / size_x_) * step_x_ + 1;
        }

        __host__ __device__
        int getY(const int idx) {
          return ((idx % size_angle_ % size_x_) / size_y_) * step_y_ + 1;
        }

        __host__ __device__
        int getZ(const int idx) {
          return 3;
        }

        __host__ __device__
        int getSX(const int idx) {
          return ((idx % size_angle_ % size_x_ % size_y_) / size_sx_) * step_sx_ + min_size_w_;
        }

        __host__ __device__
        int getSY(const int idx) {
          return ((idx % size_angle_ % size_x_ % size_y_ % size_sx_) / size_sy_) * step_sy_+ min_size_w_;
        }

        __host__ __device__
        int getSZ(const int idx) {
          return ((idx % size_angle_ % size_x_ % size_y_ % size_sx_ % size_sy_)) * step_sz_ + min_size_w_;
        }

        __host__ __device__
        float
        operator() (const int idx)
        {
          //based on the idx, generate the bbox values...
          int idx_vol = (getAngle(idx) * m_width * m_height * m_depth);
          int fac_to_1 = round(100.f * resolution_);

          int outer_edges = getRectangleFromCorner (rivs + idx_vol, getX(idx), getY(idx), getZ(idx), getSX(idx), getSY(idx), getSZ(idx), m_width, m_height);
          if (outer_edges <= 20)
            return -10.f;

          int occupancy_val = getRectangleFromCorner (rivs_occupancy + idx_vol, getX(idx), getY(idx), getZ(idx), getSX(idx), getSY(idx), getSZ(idx), m_width, m_height);
          //float vol_outer = static_cast<float> (getSX(idx) * getSY(idx) * getSZ(idx)) * fac_to_1;
          float vol_outer = (float)(getRectangleFromCorner (rivs_full_ + idx_vol, getX(idx), getY(idx), getZ(idx), getSX(idx), getSY(idx), getSZ(idx), m_width, m_height));
          int occluded_val = getRectangleFromCorner (rivs_occluded + idx_vol, getX(idx), getY(idx), getZ(idx), getSX(idx), getSY(idx), getSZ(idx), m_width, m_height);
          float free_space = vol_outer - occluded_val - occupancy_val;

          if(free_space > (occupancy_val * 5.f)) {
            return -10.f;
          }

          float clutter_score = 0.f;
          int num_points_inside = 0;
          float over_zero = 0;
          for (int j = 1; j < (max_label + 1); j++)
          {
           int idx_label = j * m_width * m_height * m_depth;
           int val = getRectangleFromCorner (rivs_histograms + (idx_vol * (max_label + 1)) + idx_label, getX(idx), getY(idx), getZ(idx), getSX(idx), getSY(idx), getSZ(idx),
                                             m_width, m_height);

           if(val==0)
             continue;

           num_points_inside += val;
           /*float sc = (val / (float)npoints_label_[(max_label + 1) * getAngle(idx) + j]) * (val / (float)(occupancy_val));
           if((val / (float)npoints_label_[(max_label + 1) * getAngle(idx) + j]) < 0.95f)
             sc *= -1.f;

           clutter_score += sc;*/
           clutter_score += (val / (float)npoints_label_[(max_label + 1) * getAngle(idx) + j]) * val;

           //the importance of the pixels in the superpixel to the whole scene
           //the importance of the pixel in the superpixel relative to the whole object
           over_zero++;
          }

          if (num_points_inside <= 0)
           return -10.f;
          else {
           clutter_score /= (float)(num_points_inside);
           if(clutter_score < 0.5f) {
             return -10.f;
           }
          }

          int vol_faces[6];
          vol_faces[0] = vol_faces[5] = getSY(idx) * getSZ(idx) * fac_to_1;
          vol_faces[1] = vol_faces[4] = getSX(idx) * getSZ(idx) * fac_to_1;
          vol_faces[2] = vol_faces[3] = getSY(idx) * getSX(idx) * fac_to_1;

          bool visible_faces[6];
          for(int j=0; j < 6; j++)
            visible_faces[j] = true;

          //compute oriented normals for faces and check the dot product with viewpoint
          /*Vector3f min_p, max_p;
          min_p.x = min_x_ + getX(idx) * resolution_;
          min_p.y = min_y_ + getY(idx) * resolution_;
          min_p.z = min_z_ + getZ(idx) * resolution_;

          max_p.x = min_x_ + (getSX(idx) + getX(idx)) * resolution_;
          max_p.y = min_y_ + (getSY(idx) + getY(idx)) * resolution_;
          max_p.z = min_z_ + (getSZ(idx) + getZ(idx)) * resolution_;

          Vector3f center((min_p.x + max_p.x) / 2.f, (min_p.y + max_p.y) / 2.f, (min_p.z + max_p.z) / 2.f);

          //compute bounding boxes vertices
          Vector3f vertices[8];
          vertices[0] = min_p;
          vertices[1] = Vector3f(min_p.x, max_p.y, min_p.z);
          vertices[2] = Vector3f(min_p.x, min_p.y, max_p.z);
          vertices[3] = Vector3f(min_p.x, max_p.y, max_p.z);
          vertices[4] = Vector3f(max_p.x, min_p.y, min_p.z);
          vertices[5] = Vector3f(max_p.x, max_p.y, min_p.z);
          vertices[6] = Vector3f(max_p.x, min_p.y, max_p.z);
          vertices[7] = max_p;

          //compute normals and orient them properly
          Vector3f normals[6];
          Vector3f faces_centroid[6];
          faces_centroid[0] = (vertices[0] + vertices[3]) / 2.f;
          faces_centroid[1] = (vertices[0] + vertices[6]) / 2.f;
          faces_centroid[2] = (vertices[0] + vertices[5]) / 2.f;
          faces_centroid[3] = (vertices[2] + vertices[7]) / 2.f;
          faces_centroid[4] = (vertices[7] + vertices[1]) / 2.f;
          faces_centroid[5] = (vertices[7] + vertices[4]) / 2.f;

          //rotate max and min points if necessary
          if (getAngle (idx) != 0)
          {
            Mat3f rot_;
            for (size_t i = 0; i < 3; i++)
            {
              for (size_t j = 0; j < 3; j++)
              {
                if (i == j)
                  rot_.mat[i][j] = 1.f;
                else
                  rot_.mat[i][j] = 0.f;
              }
            }

            double rot_rads = (static_cast<double> (angle_incr_ * getAngle (idx)) * -1.f) * 0.0174532925;
            rot_.mat[0][0] = cos (rot_rads);
            rot_.mat[1][1] = rot_.mat[0][0];
            rot_.mat[1][0] = sin (rot_rads);
            rot_.mat[0][1] = rot_.mat[1][0] * -1.f;

            min_p = rotate(rot_, min_p);
            max_p = rotate(rot_, max_p);

            center = rotate(rot_, center);

            for(int j=0; j < 6; j++) {
              faces_centroid[j] = rotate(rot_, faces_centroid[j]);
            }
          }

          for(int j=0; j < 6; j++) {
            normals[j] = faces_centroid[j] - center;
            normals[j].normalize();
          }

          Vector3f vp(vpx_, vpy_, vpz_);
          int sum_v = 0;
          //int faces_area = ((getSX(idx) * getSY(idx)) + (getSX(idx) * getSZ(idx)) + (getSY(idx) * getSZ(idx))) * 2.f;

          int faces_area = 0;
          for(int j=0; j < 6; j++) {
            Vector3f c_vp = (vp - faces_centroid[j]);
            c_vp.normalize();
            visible_faces[j] = (dot (normals[j], c_vp)) > 0.02f;
            if(visible_faces[j]) {
              sum_v++;
              faces_area += vol_faces[j];
            }
          }*/

          //int faces_area_iv = getValueByFaces(getX(idx), getY(idx), getZ(idx), getSX(idx), getSY(idx), getSZ(idx), rivs_full_ + idx_vol, m_width, m_height, visible_faces);

          /*if(faces_area_iv != faces_area) {
            printf("different areas: %d %d \n", faces_area_iv, faces_area);
          } else {
            printf("areas are equal: %d %d %d \n", faces_area_iv, faces_area, occluded_faces);
          }*/

          /*if(sum_v < 3) {
            if(!(visible_faces[0] || visible_faces[5])) {
              faces_area += vol_faces[0]; // + vol_faces[5];
              visible_faces[0] = visible_faces[5] = true;
            }

            if(!(visible_faces[1] || visible_faces[4])) {
              faces_area += vol_faces[1]; // + vol_faces[4];
              visible_faces[1] = visible_faces[4] = true;
            }
          }

          int occluded_faces = getValueByFaces (getX(idx), getY(idx), getZ(idx), getSX(idx), getSY(idx), getSZ(idx), rivs_occluded + idx_vol, m_width, m_height, visible_faces);
          //occluded_faces = (int)(occluded_faces * 0.75);*/
          visible_faces[5] = false;
          int occupancy_val_faces = getValueByFaces (getX(idx), getY(idx), getZ(idx), getSX(idx), getSY(idx), getSZ(idx), rivs_occupancy + idx_vol, m_width, m_height, visible_faces);
          //int faces_area = faces_area_iv;
          //int occluded_faces = getValueByFaces (getX(idx), getY(idx), getZ(idx), getSX(idx), getSY(idx), getSZ(idx), rivs_occluded + idx_vol, m_width, m_height, visible_faces);

          //expand
          BBox bb_extended;
          int size_ring = 2;
          int size_ring_z = 4;
          bb_extended.sx = getSX(idx) + size_ring * 2;
          bb_extended.sy = getSY(idx) + size_ring * 2;
          bb_extended.sz = getSZ(idx) + size_ring_z * 2;

          bb_extended.x = getX(idx) - int (round ((bb_extended.sx - getSX(idx)) / 2.f));
          bb_extended.y = getY(idx) - int (round ((bb_extended.sy - getSY(idx)) / 2.f));
          bb_extended.z = getZ(idx) - int (round ((bb_extended.sz - getSZ(idx)) / 2.f));

          bb_extended.x = max (bb_extended.x, 1);
          bb_extended.y = max (bb_extended.y, 1);
          bb_extended.z = max (bb_extended.z, 1);

          bb_extended.sx = min (m_width - 1, bb_extended.x + bb_extended.sx) - bb_extended.x;
          bb_extended.sy = min (m_height - 1, bb_extended.y + bb_extended.sy) - bb_extended.y;
          bb_extended.sz = min (m_depth - 1, bb_extended.z + bb_extended.sz) - bb_extended.z;

          int expanded_edges = getRectangleFromCorner (rivs + idx_vol, bb_extended.x, bb_extended.y, bb_extended.z, bb_extended.sx, bb_extended.sy,
                                                       bb_extended.sz, m_width, m_height);

          /*int expanded_occupancy = getRectangleFromCorner (rivs_occupancy + idx_vol, bb_extended.x, bb_extended.y, bb_extended.z, bb_extended.sx, bb_extended.sy,
                                                           bb_extended.sz, m_width, m_height);*/

          BBox bb_shrinked;
          bb_shrinked.sx = max(min (max (static_cast<int> (floor (getSX(idx) * shrink_factor_x)), 1), getSX(idx) - 2),1);
          bb_shrinked.sy = max(min (max (static_cast<int> (floor (getSY(idx) * shrink_factor_y)), 1), getSY(idx) - 2),1);
          bb_shrinked.sz = max(min (max (static_cast<int> (floor (getSZ(idx) * shrink_factor_z)), 1), getSZ(idx) - 2),1);

          bb_shrinked.x = getX(idx) + max (static_cast<int> (floor ((getSX(idx) - bb_shrinked.sx) / 2.f)), 1);
          bb_shrinked.y = getY(idx) + max (static_cast<int> (floor ((getSY(idx) - bb_shrinked.sy) / 2.f)), 1);
          bb_shrinked.z = getZ(idx) + max (static_cast<int> (floor ((getSZ(idx) - bb_shrinked.sz) / 2.f)), 1);

          float vol_shrinked = static_cast<float> (bb_shrinked.sx * bb_shrinked.sy * bb_shrinked.sz);
          int occupancy_val_inner = getRectangleFromCorner (rivs_occupancy + idx_vol, bb_shrinked.x, bb_shrinked.y,
                                                             bb_shrinked.z, bb_shrinked.sx, bb_shrinked.sy, bb_shrinked.sz, m_width, m_height);

          int edges_inner = getRectangleFromCorner (rivs + idx_vol, bb_shrinked.x, bb_shrinked.y,
                                                                    bb_shrinked.z, bb_shrinked.sx, bb_shrinked.sy, bb_shrinked.sz, m_width, m_height);

          /*int occluded_val_inner = getRectangleFromCorner (rivs_occluded + idx_vol, bb_shrinked.x, bb_shrinked.y,
                                                           bb_shrinked.z, bb_shrinked.sx, bb_shrinked.sy, bb_shrinked.sz, m_width, m_height);

          int edges_inner = getRectangleFromCorner (rivs + idx_vol, bb_shrinked.x, bb_shrinked.y,
                                                    bb_shrinked.z, bb_shrinked.sx, bb_shrinked.sy, bb_shrinked.sz, m_width, m_height);*/

          /*vol_faces[0] = vol_faces[5] = getSY(idx) * getSZ(idx);
          vol_faces[1] = vol_faces[4] = getSX(idx) * getSZ(idx);
          vol_faces[2] = vol_faces[3] = getSY(idx) * getSX(idx);*/

          int faces_area =  vol_faces[0] + vol_faces[1] + vol_faces[2];
          float edges_score = (float)(outer_edges) / (float)( (faces_area /*+ occupancy_val_faces - occluded_faces*/)); // - occluded_faces - occupancy_val_faces));
          //float edges_score = (float)(outer_edges) / (float)( (free_space));
          float exp_edges_score = ((outer_edges) / (float)expanded_edges);
          edges_score = min(edges_score, 1.f);
          float score;
          float clutter_occ_score = clutter_score;

          //Color...
          float color_distance = 0;

          {
            float * colors_box = new float[size_yuv_hist_];
            float * colors_box_expanded = new float[size_yuv_hist_];
            float * colors_ring = new float[size_yuv_hist_];
            for(int j=0; j < size_yuv_hist_; j++) {
              colors_box[j] = 0.f;
              colors_box_expanded[j] = 0.f;
              colors_ring[j] = 0.f;
            }

            int n_points = getColorHistogram(getX(idx), getY(idx), getZ(idx), getSX(idx), getSY(idx), getSZ(idx),
                              riv_points_color_histogram_ + idx_vol,
                              riv_color_histograms_ + idx_vol * size_yuv_hist_,
                              m_width, m_height, m_depth, colors_box, size_yuv_hist_);

            BBox bb_extended;
            bb_extended.sx = int (round (getSX(idx) * expand_factor));
            bb_extended.sy = int (round (getSY(idx) * expand_factor));
            bb_extended.sz = int (round (getSZ(idx) * expand_factor));

            bb_extended.x = getX(idx) - int (round ((bb_extended.sx - getSX(idx)) / 2.f));
            bb_extended.y = getY(idx) - int (round ((bb_extended.sy - getSY(idx)) / 2.f));
            bb_extended.z = getZ(idx) - int (round ((bb_extended.sz - getSZ(idx)) / 2.f));

            bb_extended.x = max (bb_extended.x, 1);
            bb_extended.y = max (bb_extended.y, 1);
            bb_extended.z = max (bb_extended.z, 1);

            bb_extended.sx = min (m_width - 1, bb_extended.x + bb_extended.sx) - bb_extended.x;
            bb_extended.sy = min (m_height - 1, bb_extended.y + bb_extended.sy) - bb_extended.y;
            bb_extended.sz = min (m_depth - 1, bb_extended.z + bb_extended.sz) - bb_extended.z;

            int n_points_exp = getColorHistogram(bb_extended.x, bb_extended.y, bb_extended.z, bb_extended.sx, bb_extended.sy, bb_extended.sz,
                              riv_points_color_histogram_ + idx_vol,
                              riv_color_histograms_ + idx_vol  * size_yuv_hist_,
                              m_width, m_height, m_depth, colors_box_expanded, size_yuv_hist_);

            int npoints_ring = n_points_exp - n_points;
            if(min(npoints_ring, n_points) > 0)
            {
              for(int j=0; j < size_yuv_hist_; j++) {
                colors_ring[j] = colors_box_expanded[j] - colors_box[j];
              }

              int sum_box, sum_ring;
              sum_box = sum_ring = 0;
              for(int j=0; j < size_yuv_hist_; j++) {
                sum_box += colors_box[j];
                sum_ring += colors_ring[j];
              }

              for(int j=0; j < size_yuv_hist_; j++) {
                /*colors_ring[j] /= (float)(npoints_ring);
                colors_box[j] /= (float)(n_points);*/
                colors_ring[j] /= (float)(sum_ring);
                colors_box[j] /= (float)(sum_box);
              }

              color_distance = 0.f;
              for(int j=0; j < size_yuv_hist_; j++) {
                color_distance += (colors_box[j] - colors_ring[j])*(colors_box[j] - colors_ring[j]);
              }

              color_distance = (float)(min(npoints_ring, n_points)) / (float)(max(npoints_ring, n_points)) * sqrt(color_distance);
            }

            delete[] colors_box;
            delete[] colors_ring;
            delete[] colors_box_expanded;
          } //the higher the color distance the better

          if(isnan(color_distance))
            color_distance = 0;

          /*float num_cues = 3.f;
          score = (exp_edges_score
                   + edges_score // (1.f - (occupancy_val_faces / (float)(faces_area - occluded_faces)))
                   + clutter_occ_score //clutter_occ_score
                   //- (occupancy_val_inner / vol_outer)
                   //+ color_distance
                   ) / num_cues;*/

          float num_cues = 2.f;
          score = (exp_edges_score * clutter_occ_score
                   + edges_score
                   + (1.f - (0.15f * free_space / faces_area))
                   //+ (1.f - (free_space / (vol_outer - occluded_val)))
                   //+ 0.25f * color_distance
                   ) / num_cues;

          score = (exp_edges_score * edges_score
                  //+ 0.2f * color_distance
                  //- 0.5f * occupancy_val_inner / (float)(vol_shrinked)
                  + 0.2f * (1.f - (free_space / vol_outer))
                  /*+ 0.4f * edges_score*/) * clutter_occ_score;

          if(exp_edges_score < 0.95f) {
            score += 0.2f * color_distance * score;
          }

          //some tests...
          //float faces_areas_inner = bb_shrinked.sy * bb_shrinked.sz +  bb_shrinked.sy * bb_shrinked.sx +  bb_shrinked.sx * bb_shrinked.sz;
          //edges_score = (float)(outer_edges - edges_inner) / (float)( (faces_area - faces_areas_inner /*+ occupancy_val_faces - occluded_faces*/)); // - occluded_faces - occupancy_val_faces));
          //edges_score = edges_inner / (float)(faces_areas_inner) / (outer_edges / (float)(faces_area));
          //score = edges_score * exp_edges_score * clutter_occ_score * (1.f - (0.15f * free_space / faces_area));

          /*if(score > 0.975) {
            printf("color distance: %f %d\n", color_distance, size_yuv_hist_);
          }*/

          /*if(score > 0.58) {
            printf("%d %d %f %f %f %f\n", occupancy_val_faces, faces_area, score, exp_edges_score, clutter_occ_score, 1.f - (occupancy_val_faces / (float)(faces_area)));
          }*/

          /*if((1.f - (occupancy_val_faces / (float)(faces_area - occluded_faces))) > 1) {
            printf("%d %d %d \n", occupancy_val_faces, faces_area, occluded_faces);
          }*/

          if (score <= 0 || isnan(score)) {
            if(isnan(score)) {
              printf("%f %f %f %f\n", score, exp_edges_score, clutter_occ_score, color_distance);
            }
            return 0.f;
          }

          return score;
        }
      };

      struct scoreGT
      {

        float threshold_;
        float max_score_;
        scoreGT (float _thres, float _max_score) :
          threshold_ (_thres), max_score_ (_max_score)
        {
        }

        __host__ __device__
        bool
        operator() (const thrust::tuple<int, float> & b1) const
        {
          if ((thrust::get<1> (b1) /*/ max_score_*/) > threshold_)
            return true;

          return false;
        }
      };

      //objectness 3d cuda
      Objectness3DCuda::Objectness3DCuda (int angle, float expand_factor, int max_label,
                                               int GRID_SIZEX, int GRID_SIZEY, int GRID_SIZEZ,
                                               float res, int min_sw, int max_sw,
                                               float minx, float maxx, float miny, float maxy, float minz, float maxz)
      {

        min_z = minz;
        max_z = maxz;

        min_y = miny;
        max_y = maxy;

        min_x = minx;
        max_x = maxx;

        angle_incr_ = angle;
        GRIDSIZE_X_ = GRID_SIZEX;
        GRIDSIZE_Y_ = GRID_SIZEY;
        GRIDSIZE_Z_ = GRID_SIZEZ;
        expand_factor_ = expand_factor;
        max_label_ = max_label;
        step_x_ = step_y_ = step_z_ = 2;
        step_sx_ = step_sy_ = 2;
        step_sz_ = 1;
        min_size_w_ = min_sw;
        max_size_w_ = max_sw;
        //start_z_ = 3;
        resolution = res;
        start_z_ = static_cast<int>(round(abs(min_z / resolution)));

        std::cout << "Some values..." << start_z_ << " " << resolution << " " << min_size_w_ << " " << max_size_w_ << std::endl;

        size_angle_ = ((GRID_SIZEX - min_size_w_) / step_x_) * ((GRID_SIZEY - min_size_w_) / step_y_) * ((max_size_w_ - min_size_w_) / step_sx_)
            * ((max_size_w_ - min_size_w_) / step_sy_) * ((std::min (max_size_w_ - min_size_w_, GRID_SIZEZ - min_size_w_)) / step_sz_);

        size_x_ = ((GRID_SIZEY - min_size_w_) / step_y_) * ((max_size_w_ - min_size_w_) / step_sx_) * ((max_size_w_ - min_size_w_) / step_sy_)
            * ((std::min (max_size_w_ - min_size_w_, GRID_SIZEZ - min_size_w_)) / step_sz_);

        size_y_ = ((max_size_w_ - min_size_w_) / step_sx_) * ((max_size_w_ - min_size_w_) / step_sy_) * ((std::min (max_size_w_ - min_size_w_,
                                                                                                                    GRID_SIZEZ - min_size_w_))
            / step_sz_);

        size_sx_ = ((max_size_w_ - min_size_w_) / step_sy_) * ((std::min (max_size_w_ - min_size_w_, GRID_SIZEZ - min_size_w_)) / step_sz_);

        size_sy_ = ((std::min (max_size_w_ - min_size_w_, GRID_SIZEZ - min_size_w_)) / step_sz_);

        table_plane_label_ = -1;
      }

      void
      Objectness3DCuda::assign_bb_values (BBox & bb, int idx, float score)
      {
        bb.angle = idx / size_angle_;
        int idx_local = idx % size_angle_;
        bb.x = (idx_local / size_x_) * step_x_ + 1;
        idx_local = idx_local % size_x_;
        bb.y = (idx_local / size_y_) * step_y_ + 1;
        idx_local = idx_local % size_y_;
        bb.z = start_z_;
        bb.sx = (idx_local / size_sx_) * step_sx_ + min_size_w_;
        idx_local = idx_local % size_sx_;
        bb.sy = (idx_local / size_sy_) * step_sy_  + min_size_w_;
        idx_local = idx_local % size_sy_;
        bb.sz = idx_local * step_sz_ + min_size_w_;
        bb.score = score;
      }

      void
      Objectness3DCuda::setNPointsLabel (std::vector<std::vector<int> > & npoints_label)
      {
        int num_ivs = (90 / angle_incr_); //number of integral volumes to compute
        cudaMalloc ((void **)&raw_ptr_rivs_npoints_label_, num_ivs * npoints_label[0].size () * sizeof(int));
        int * np_label = new int[num_ivs * npoints_label[0].size ()];

        for (size_t i = 0; i < num_ivs; i++)
        {
          for (size_t j = 0; j < npoints_label[i].size (); j++)
          {
            np_label[i * npoints_label[0].size () + j] = npoints_label[i][j];
          }
        }

        cudaMemcpy (raw_ptr_rivs_npoints_label_, np_label, num_ivs * npoints_label[0].size () * sizeof(int), cudaMemcpyHostToDevice);
        thrust::device_ptr<int> dev_ptr (raw_ptr_rivs_npoints_label_);
        npoints_label_ = dev_ptr;

        delete[] np_label;
      }

      void
      Objectness3DCuda::addOcclusionVolumes (std::vector<boost::shared_ptr<IntegralVolume> > & rivs)
      {
        int num_ivs = (90 / angle_incr_); //number of integral volumes to compute
        int vol_size = GRIDSIZE_X_ * GRIDSIZE_Y_ * GRIDSIZE_Z_;
        int vol_size_bytes = vol_size * sizeof(int);
        cudaMalloc ((void **)&raw_ptr_rivs_occluded, num_ivs * vol_size_bytes);
        for (size_t i = 0; i < num_ivs; i++)
        {
          cudaMemcpy (raw_ptr_rivs_occluded + i * vol_size, rivs[i]->getPointer (), vol_size_bytes, cudaMemcpyHostToDevice);
        }

        thrust::device_ptr<int> dev_ptr (raw_ptr_rivs_occluded);
        rivs_occluded = dev_ptr;
      }

      void
      Objectness3DCuda::addFullVolumes (std::vector<boost::shared_ptr<IntegralVolume> > & rivs)
      {
        int num_ivs = (90 / angle_incr_); //number of integral volumes to compute
        int vol_size = GRIDSIZE_X_ * GRIDSIZE_Y_ * GRIDSIZE_Z_;
        int vol_size_bytes = vol_size * sizeof(int);
        cudaMalloc ((void **)&raw_ptr_rivs_full, num_ivs * vol_size_bytes);
        for (size_t i = 0; i < num_ivs; i++)
        {
          cudaMemcpy (raw_ptr_rivs_full + i * vol_size, rivs[i]->getPointer (), vol_size_bytes, cudaMemcpyHostToDevice);
        }

        thrust::device_ptr<int> dev_ptr (raw_ptr_rivs_full);
        rivs_full = dev_ptr;
      }

      void
      Objectness3DCuda::addOccupancyVolumes (std::vector<boost::shared_ptr<IntegralVolume> > & rivs)
      {
        int num_ivs = (90 / angle_incr_); //number of integral volumes to compute
        int vol_size = GRIDSIZE_X_ * GRIDSIZE_Y_ * GRIDSIZE_Z_;
        int vol_size_bytes = vol_size * sizeof(int);

        cudaMalloc ((void **)&raw_ptr_rivs_occupancy, num_ivs * vol_size_bytes);
        for (size_t i = 0; i < num_ivs; i++)
        {
          cudaMemcpy (raw_ptr_rivs_occupancy + i * vol_size, rivs[i]->getPointer (), vol_size_bytes, cudaMemcpyHostToDevice);
        }

        thrust::device_ptr<int> dev_ptr (raw_ptr_rivs_occupancy);
        rivs_occupancy = dev_ptr;
      }

      void
      Objectness3DCuda::addOccupancyCompleteVolumes (std::vector<boost::shared_ptr<IntegralVolume> > & rivs)
      {
        int num_ivs = (90 / angle_incr_); //number of integral volumes to compute
        int * raw_ptr;
        int vol_size = GRIDSIZE_X_ * GRIDSIZE_Y_ * GRIDSIZE_Z_;
        int vol_size_bytes = vol_size * sizeof(int);

        cudaMalloc ((void **)&raw_ptr, num_ivs * vol_size_bytes);
        for (size_t i = 0; i < num_ivs; i++)
        {
          cudaMemcpy (raw_ptr + i * vol_size, rivs[i]->getPointer (), vol_size_bytes, cudaMemcpyHostToDevice);
        }

        thrust::device_ptr<int> dev_ptr (raw_ptr);
        rivs_occupancy_complete = dev_ptr;
      }

      void
      Objectness3DCuda::addEdgesIV (std::vector<boost::shared_ptr<IntegralVolume> > & rivs)
      {
        int num_ivs = (90 / angle_incr_); //number of integral volumes to compute
        int vol_size = GRIDSIZE_X_ * GRIDSIZE_Y_ * GRIDSIZE_Z_;
        int vol_size_bytes = vol_size * sizeof(int);

        cudaMalloc ((void **)&raw_ptr_rivs, num_ivs * vol_size_bytes);
        for (size_t i = 0; i < num_ivs; i++)
        {
          cudaMemcpy (raw_ptr_rivs + i * vol_size, rivs[i]->getPointer (), vol_size_bytes, cudaMemcpyHostToDevice);
        }

        thrust::device_ptr<int> dev_ptr (raw_ptr_rivs);
        rivs_ = dev_ptr;
      }

      void
      Objectness3DCuda::addColorHistogramVolumes (
                                           std::vector<std::vector<boost::shared_ptr<IntegralVolume> > > & rivs_color,
                                           std::vector< boost::shared_ptr<IntegralVolume> > & rivs_color_points, int size) {

        yuv_hist_size_ = size;
        int num_ivs = (90 / angle_incr_); //number of integral volumes to compute
        int vol_size_bytes = GRIDSIZE_X_ * GRIDSIZE_Y_ * GRIDSIZE_Z_ * sizeof(int);
        int single_vol_stride = GRIDSIZE_X_ * GRIDSIZE_Y_ * GRIDSIZE_Z_;
        int vol_stride = single_vol_stride * yuv_hist_size_;
        int total = vol_stride * num_ivs;

        cudaMalloc ((void **)&raw_ptr_rivs_rivs_color_histograms, total * sizeof(int));
        cudaMalloc ((void **)&raw_ptr_rivs_rivs_npoints_color, num_ivs * vol_size_bytes);

        for (int i = 0; i < num_ivs; i++)
        {
          cudaMemcpy (raw_ptr_rivs_rivs_npoints_color + i * single_vol_stride, rivs_color_points[i]->getPointer (), vol_size_bytes, cudaMemcpyHostToDevice);
          int vol_idx = vol_stride * i;
          for (int j = 0; j < yuv_hist_size_; j++)
          {
            int stride = vol_idx + single_vol_stride * j;
            cudaMemcpy ((int *)raw_ptr_rivs_rivs_color_histograms + stride, rivs_color[i][j]->getPointer (), vol_size_bytes, cudaMemcpyHostToDevice);
          }
        }

        {
          thrust::device_ptr<int> dev_ptr (raw_ptr_rivs_rivs_color_histograms);
          rivs_color_histograms = dev_ptr;
        }

        {
          thrust::device_ptr<int> dev_ptr (raw_ptr_rivs_rivs_npoints_color);
          rivs_npoints_color_histograms = dev_ptr;
        }
      }

      void
      Objectness3DCuda::addHistogramVolumes (std::vector<std::vector<boost::shared_ptr<IntegralVolume> > > & rivs)
      {
        int num_ivs = (90 / angle_incr_); //number of integral volumes to compute
        int vol_size_bytes = GRIDSIZE_X_ * GRIDSIZE_Y_ * GRIDSIZE_Z_ * sizeof(int);
        int single_vol_stride = GRIDSIZE_X_ * GRIDSIZE_Y_ * GRIDSIZE_Z_;
        int vol_stride = single_vol_stride * (max_label_ + 1);
        int total = vol_stride * num_ivs;
        cudaMalloc ((void **)&raw_ptr_rivs_rivs_histograms, total * sizeof(int));
        for (int i = 0; i < num_ivs; i++)
        {
          int vol_idx = vol_stride * i;
          for (int j = 0; j < (max_label_ + 1); j++)
          {
            int stride = vol_idx + single_vol_stride * j;
            cudaMemcpy ((int *)raw_ptr_rivs_rivs_histograms + stride, rivs[i][j]->getPointer (), vol_size_bytes, cudaMemcpyHostToDevice);
          }
        }

        thrust::device_ptr<int> dev_ptr (raw_ptr_rivs_rivs_histograms);
        rivs_histograms = dev_ptr;
      }

      void
      Objectness3DCuda::generateAndComputeBoundingBoxesScore (std::vector<BBox> & boxes, float threshold)
      {

        long unsigned int num_sampled_wins_ = static_cast<int> (90 / angle_incr_)
            * ((GRIDSIZE_X_ - min_size_w_) / step_x_)
            * ((GRIDSIZE_Y_ - min_size_w_) / step_y_)
            * ((max_size_w_ - min_size_w_) / step_sx_)
            * ((max_size_w_ - min_size_w_) / step_sy_)
            * ((std::min (max_size_w_ - min_size_w_, GRIDSIZE_Z_ - min_size_w_)) / step_sz_);

        //create a vector of integer indices and pass it to generateAndEvaluateBox kernel
        std::cout << "Number of sampled windows:" << num_sampled_wins_ << std::endl;
        boxes.clear();
        int indices_generated = 2000000;
        thrust::device_vector<int> indices(indices_generated);
        thrust::device_vector<float> scores (indices.size ());
        thrust::device_vector < thrust::tuple<int, float> > over_threshold(1000000);

        for(size_t ii=0; ii < static_cast<int> (std::ceil(num_sampled_wins_ / static_cast<float>(indices_generated))); ii++) {

          int start_idx = static_cast<int>(ii) * indices_generated;
          int num = std::min(indices_generated, static_cast<int>(num_sampled_wins_ - static_cast<int>(ii) * indices_generated));

          thrust::counting_iterator<int> idx_iterator (start_idx);
          thrust::device_vector<int>::iterator it = thrust::copy_if (
                                                                     idx_iterator,
                                                                     idx_iterator + num,
                                                                     indices.begin (),
                                                                     validIdx (rivs_occupancy, GRIDSIZE_X_, GRIDSIZE_Y_, GRIDSIZE_Z_, min_size_w_, max_size_w_,
                                                                               step_x_, step_y_, step_z_, step_sx_, step_sy_, step_sz_));

          int valid_ind = it - indices.begin ();
          //std::cout << "valid boxes:" << valid_ind << " from:" << num << std::endl;
          if(valid_ind == 0)
            continue;

          thrust::transform (indices.begin (),
                             indices.begin () + valid_ind,
                             scores.begin (),
                             generateAndEvaluateBox (rivs_, rivs_occluded, rivs_occupancy, rivs_occupancy_complete, rivs_histograms, npoints_label_,
                                                     rivs_color_histograms, rivs_npoints_color_histograms, rivs_full,
                                                     GRIDSIZE_X_, GRIDSIZE_Y_, GRIDSIZE_Z_, expand_factor_, max_label_, min_size_w_, max_size_w_,
                                                     step_x_, step_y_, step_z_, step_sx_, step_sy_, step_sz_, vpx_, vpy_, vpz_,
                                                     min_x, min_y, min_z, max_x, max_y, max_z, resolution, angle_incr_, table_plane_label_, yuv_hist_size_));

          //std::cout << "valid boxes:" << valid_ind << " from:" << num << std::endl;

          /*int over_thres = thrust::count_if (thrust::make_zip_iterator (thrust::make_tuple (indices.begin (), scores.begin ())),
                                             thrust::make_zip_iterator (thrust::make_tuple (indices.begin () + valid_ind, scores.begin () + valid_ind)),
                           scoreGT (threshold, *(thrust::max_element (scores.begin (), scores.begin () + valid_ind))));*/

          float max_score = *(thrust::max_element (scores.begin (), scores.begin () + valid_ind));

          if(max_score <= 0.f)
            continue;

          //filter those over threshold
          thrust::device_vector< thrust::tuple<int, float> >::iterator it_ot =
                        thrust::copy_if (thrust::make_zip_iterator (thrust::make_tuple (indices.begin (), scores.begin ())),
                                         thrust::make_zip_iterator (thrust::make_tuple (indices.begin () + valid_ind, scores.begin () + valid_ind)),
                                         over_threshold.begin (),
                                         scoreGT (threshold, max_score));

          int n_over = it_ot - over_threshold.begin ();
          //std::cout << "bounding boxes over threshold:" << n_over << std::endl;
          if(n_over == 0)
            continue;

          std::cout << "bounding boxes over threshold:" << n_over << " " << *(thrust::max_element (scores.begin (), scores.begin () + valid_ind)) << std::endl;

          //once the bounding boxes have been sorted, generate actual bboxes and fill the host bounding boxes vector
          thrust::host_vector < thrust::tuple<int, float> > host_tuples;
          host_tuples = over_threshold;
          host_tuples.resize(n_over);

          int prev_size = static_cast<int>(boxes.size());
          boxes.resize (prev_size + n_over);
          int valid = prev_size;
          for (size_t i = 0; i < host_tuples.size (); i++)
          {
            thrust::tuple<int, float> tp = host_tuples[i];
            int idx = thrust::get<0> (tp);
            if (idx != 0)
            {
              assign_bb_values (boxes[valid], idx, thrust::get<1> (tp));
              valid++;
            }
          }

          boxes.resize(valid);

          //std::cout << "finished adding..." << std::endl;
        }
      }
    }
  }
}
