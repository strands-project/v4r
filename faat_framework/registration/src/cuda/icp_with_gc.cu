/*
 * icp_with_gc.cu
 *
 *  Created on: May 6, 2013
 *      Author: aitor
 */

#include "faat_pcl/registration/cuda/icp_with_gc.h"
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/extrema.h>

namespace faat_pcl
{
  namespace cuda
  {
    namespace registration
    {

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
        void
        set(float _x, float _y, float _z)
        {
          x = _x;
          y = _y;
          z = _z;
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

      inline __host__ __device__ float sqrDist(Vector3f a, Vector3f b)
      {
        return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);
      }

      template<typename PointType, typename VoxelType>
      ICPWithGC<PointType, VoxelType>::ICPWithGC()
      {

      }

      template<typename PointType>
      inline __host__ __device__ PointType transform(Mat4f rot_, PointType pt)
      {
        PointType min_tp;
        min_tp.x = rot_.mat[0][0] * pt.x + rot_.mat[0][1] * pt.y + rot_.mat[0][2] * pt.z + rot_.mat[0][3];
        min_tp.y = rot_.mat[1][0] * pt.x + rot_.mat[1][1] * pt.y + rot_.mat[1][2] * pt.z + rot_.mat[1][3];
        min_tp.z = rot_.mat[2][0] * pt.x + rot_.mat[2][1] * pt.y + rot_.mat[2][2] * pt.z + rot_.mat[2][3];
        min_tp.rgb_set = pt.rgb_set;
        if(pt.rgb_set)
        {
          min_tp.rgb = pt.rgb;
        }
        return min_tp;
      }

      template<typename PointType>
      inline __host__ __device__ float sqrDist(PointType a, PointType b)
      {
        return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);
      }

      template<typename PointType>
      inline __host__ __device__ float sqrDist(Vector3f a, PointType b)
      {
        return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);
      }

      template<typename PointType, typename VoxelData>
      struct transformAndComputeOverlapPlusRegError
      {
        thrust::device_ptr<PointType> input_cloud_dev_ptr_;
        thrust::device_ptr<PointType> target_cloud_dev_ptr_;
        thrust::device_ptr<ICPNodeGPU> nodes_dev_ptr_;
        //target distance transform
        int input_num_points_;
        int num_nodes_;
        float inliers_threshold_;
        float inliers_threshold_sq_;
        //dist transform stuff
        thrust::device_ptr<PropDistanceFieldVoxel> vx_data_;
        float origin_[3];
        int num_cells_[3];
        int stride1_;
        int stride2_;
        float resolution_;
        float sigma_;
        transformAndComputeOverlapPlusRegError(typename thrust::device_ptr<PointType> _input_cloud,
                                                    typename thrust::device_ptr<PointType> _target_cloud,
                                                    thrust::device_ptr<ICPNodeGPU> _nodes,
                                                    int num_points, int num_nodes, float inliers_threshold,
                                                    thrust::device_ptr<PropDistanceFieldVoxel> _vx_data,
                                                    float * origin, int * num_cells,
                                                    int stride1, int stride2, float res)
        {
          input_cloud_dev_ptr_ = _input_cloud;
          target_cloud_dev_ptr_ = _target_cloud;
          nodes_dev_ptr_ = _nodes;
          input_num_points_ = num_points;
          num_nodes_ = num_nodes;
          inliers_threshold_ = inliers_threshold;
          vx_data_ = _vx_data;
          for(size_t i=0; i < 3; i++)
          {
            origin_[i] = *(origin++);
            num_cells_[i] = *(num_cells++);
          }

          stride1_ = stride1;
          stride2_ = stride2;
          resolution_ = res;
          inliers_threshold_sq_ = inliers_threshold_ * inliers_threshold_;
          sigma_ = 15.f;
          sigma_ *= sigma_;
        }

        __host__ __device__
        inline bool isCellValid(int x, int y, int z) const
        {
          return (
              x>=0 && x<num_cells_[0] &&
              y>=0 && y<num_cells_[1] &&
              z>=0 && z<num_cells_[2]);
        }

        __host__ __device__
        inline int ref(int x, int y, int z) const
        {
          return x*stride1_ + y*stride2_ + z;
        }

        __host__ __device__
        inline void getCell(int x, int y, int z, PropDistanceFieldVoxel & cell) const
        {
          const int r = ref(x,y,z);
          cell = vx_data_[r];
        }

        __host__ __device__
        inline int getCellFromLocation(int dim, float loc) const
        {
          float res = (loc-origin_[dim])/resolution_;
          if (res > 0)
            return floor(res + 0.5);
          else
            return ceil(res - 0.5);
        }

        __host__ __device__
        inline float getLocationFromCell(int dim, int cell) const
        {
          return origin_[dim] + resolution_*(float(cell));
        }

        __host__ __device__
        inline bool worldToGrid(float world_x, float world_y, float world_z, int& x, int& y, int& z) const
        {
          x = getCellFromLocation(0, world_x);
          y = getCellFromLocation(1, world_y);
          z = getCellFromLocation(2, world_z);
          return isCellValid(x,y,z);
        }

        __host__ __device__
        point_statistics
        operator() (const int idx)
        {
          //transform p using mat
          PointType p_trans = transform<PointType>(((ICPNodeGPU)(nodes_dev_ptr_[idx / input_num_points_])).transform_, input_cloud_dev_ptr_[idx % input_num_points_]);
          float distance;
          bool rgb = p_trans.rgb_set;

          point_statistics ps;
          ps.overlap_ = 0;
          ps.reg_error_ = 0.f;
          ps.color_reg_ = 0.f;
          int x,y,z;
          bool valid = worldToGrid(p_trans.x, p_trans.y, p_trans.z, x, y, z);
          if (valid)
          {
            Vector3f p_voxel;
            float voxel_rgb;
            if(((PropDistanceFieldVoxel)vx_data_[ref(x,y,z)]).occupied_)
            {
              //p_voxel.set(getLocationFromCell(0, x), getLocationFromCell(1, y), getLocationFromCell(2, z));
              PointType p_t = target_cloud_dev_ptr_[((PropDistanceFieldVoxel)vx_data_[ref(x,y,z)]).idx_to_input_cloud_];
              p_voxel.set(p_t.x, p_t.y, p_t.z);
              if(rgb)
              {
                voxel_rgb = p_t.rgb;
              }
            }
            else
            {
              const int r = ref(x,y,z);
              int * cp;
              cp = ((PropDistanceFieldVoxel)vx_data_[r]).closest_point_;

              /*p_voxel.set(getLocationFromCell(0, *cp),
                          getLocationFromCell(1, *(cp+1)),
                          getLocationFromCell(2, *(cp+2)));*/

              PointType p_t = target_cloud_dev_ptr_[((PropDistanceFieldVoxel)vx_data_[ref(*cp,*(cp+1),*(cp+2))]).idx_to_input_cloud_];
              p_voxel.set(p_t.x, p_t.y, p_t.z);
              if(rgb)
              {
                voxel_rgb = p_t.rgb;
              }
            }

            distance = sqrDist(p_voxel, p_trans);
            if((distance < inliers_threshold_sq_))
            {
              ps.reg_error_ = -(distance / (inliers_threshold_sq_)) + 1;
              ps.overlap_ = 1;

              if(rgb)
              {
                unsigned int rgb = *reinterpret_cast<int*> (&p_trans.rgb);
                unsigned char rm = (rgb >> 16) & 0x0000ff;
                unsigned char gm = (rgb >> 8) & 0x0000ff;
                unsigned char bm = (rgb) & 0x0000ff;

                //float ym = 0.257f * rm + 0.504f * gm + 0.098f * bm + 16; //between 16 and 235
                float um = -(0.148f * rm) - (0.291f * gm) + (0.439f * bm) + 128;
                float vm = (0.439f * rm) - (0.368f * gm) - (0.071f * bm) + 128;

                rgb = *reinterpret_cast<int*> (&voxel_rgb);
                unsigned char rs = (rgb >> 16) & 0x0000ff;
                unsigned char gs = (rgb >> 8) & 0x0000ff;
                unsigned char bs = (rgb) & 0x0000ff;

                //float ys = 0.257f * rs + 0.504f * gs + 0.098f * bs + 16;
                float us = -(0.148f * rs) - (0.291f * gs) + (0.439f * bs) + 128;
                float vs = (0.439f * rs) - (0.368f * gs) - (0.071f * bs) + 128;
                ps.color_reg_ = exp ((-0.5f * (um - us) * (um - us)) / (sigma_)) * exp ((-0.5f * (vm - vs) * (vm - vs)) / (sigma_));
                //printf("%f %f %f\n", ym, um, vm);
                //printf("%f %f %f\n", ys, us, vs);
              }
              else
              {
                ps.color_reg_ = 1.f;
              }
            }
          }

          return ps;
        }
      };

      template<typename PointType, typename VoxelData>
      struct isCellValidKernel
      {

        thrust::device_ptr<PointType> input_cloud_dev_ptr_;
        thrust::device_ptr<ICPNodeGPU> nodes_dev_ptr_;
        int input_num_points_;
        thrust::device_ptr<PropDistanceFieldVoxel> vx_data_;
        float origin_[3];
        int num_cells_[3];
        int stride1_;
        int stride2_;
        float resolution_;
        int num_nodes_;

        isCellValidKernel (typename thrust::device_ptr<PointType> _input_cloud,
                      thrust::device_ptr<ICPNodeGPU> _nodes, int num_points, int num_nodes,
                      thrust::device_ptr<PropDistanceFieldVoxel> _vx_data, float * origin,
                      int * num_cells, int stride1, int stride2, float res)
        {
          input_cloud_dev_ptr_ = _input_cloud;
          nodes_dev_ptr_ = _nodes;
          input_num_points_ = num_points;
          num_nodes_ = num_nodes;
          vx_data_ = _vx_data;
          for (size_t i = 0; i < 3; i++)
          {
            origin_[i] = *(origin++);
            num_cells_[i] = *(num_cells++);
          }

          stride1_ = stride1;
          stride2_ = stride2;
          resolution_ = res;
        }

        __host__ __device__
        inline bool isCellValid(int x, int y, int z) const
        {
          return (
              x>=0 && x<num_cells_[0] &&
              y>=0 && y<num_cells_[1] &&
              z>=0 && z<num_cells_[2]);
        }

        __host__ __device__
        inline int ref(int x, int y, int z) const
        {
          return x*stride1_ + y*stride2_ + z;
        }

        __host__ __device__
        inline void getCell(int x, int y, int z, PropDistanceFieldVoxel & cell) const
        {
          const int r = ref(x,y,z);
          cell = vx_data_[r];
        }

        __host__ __device__
        inline int getCellFromLocation(int dim, float loc) const
        {
          float res = (loc-origin_[dim])/resolution_;
          if (res > 0)
            return floor(res + 0.5);
          else
            return ceil(res - 0.5);
        }

        __host__ __device__
        inline float getLocationFromCell(int dim, int cell) const
        {
          return origin_[dim] + resolution_*(float(cell));
        }

        __host__ __device__
        inline bool worldToGrid(float world_x, float world_y, float world_z, int& x, int& y, int& z) const
        {
          x = getCellFromLocation(0, world_x);
          y = getCellFromLocation(1, world_y);
          z = getCellFromLocation(2, world_z);
          return isCellValid(x,y,z);
        }

        __host__ __device__
         int
         operator() (const int idx)
         {
           //transform p using mat
           PointType p_trans = transform<PointType>(((ICPNodeGPU)(nodes_dev_ptr_[idx / input_num_points_])).transform_, input_cloud_dev_ptr_[idx % input_num_points_]);
           int x,y,z;
           bool valid = worldToGrid(p_trans.x, p_trans.y, p_trans.z, x, y, z);
           return (int)(valid);
         }
      };

      struct sumPointStatistics
      {
        __host__ __device__
          point_statistics
          operator() (const point_statistics & p1, const point_statistics & p2)
          {
            point_statistics sum;
            sum.overlap_ = p1.overlap_ + p2.overlap_;
            sum.reg_error_ = p1.reg_error_ + p2.reg_error_;
            sum.color_reg_ = p1.color_reg_ + p2.color_reg_;
            return sum;
          }
      };

      struct generateKeys
      {

        int points_per_hyp_;
        generateKeys(int _points_per_hyp)
        {
          points_per_hyp_ = _points_per_hyp;
        }

        __host__ __device__
        int
        operator() (const int idx)
        {
          return idx / points_per_hyp_;
        }
      };

      template<typename PointType, typename VoxelType>
      void ICPWithGC<PointType, VoxelType>::computeOverlapAndRegistrationError()
      {
        //std::cout << "computeOverlapAndRegistrationError called..." << std::endl;
        //std::cout << "num_nodes * points:" << num_nodes_ * input_->size() << std::endl;

        //this function takes the icp nodes (including a transformation) and takes each p in input_
        //transforms it num_nodes_ times with the corresponding node transformation, get the NN in target and computes an overlap
        //vector as well as a reg_error num_nodes times for each point that can be afterwards summed up using the reduce_by_key method
        //the basic unit being processed in parallel is num_nodes times the points in input_

        int indices_generated = 10000000;
        int total = num_nodes_ * input_->size();
        int num_nodes_per_iteration = static_cast<int> (std::floor(indices_generated / static_cast<float>(input_->size())));
        thrust::equal_to<int> binary_pred;

        for(size_t ii=0; ii < static_cast<int> (std::ceil(num_nodes_ / static_cast<float>(num_nodes_per_iteration))); ii++)
        {
          int indices_iter = std::min(num_nodes_per_iteration * input_->size(), total - ii * num_nodes_per_iteration * input_->size());
          int num_nodes_iter = std::min(num_nodes_per_iteration, static_cast<int>(num_nodes_ - ii * num_nodes_per_iteration));
          //std::cout << ii << " " << indices_iter << " " << num_nodes_iter << std::endl;
          thrust::counting_iterator<int> first(ii * (input_->size() * num_nodes_per_iteration));
          thrust::counting_iterator<int> last = first + indices_iter;
          /*thrust::device_vector<int> invalid(indices_iter);

          thrust::transform (first,
                             last,
                             invalid.begin(),
                             isCellValidKernel<PointType, VoxelType>(input_dev_ptr_, hyp_dev_ptr_,
                                                                       input_->size(), num_nodes_,
                                                                       vx_grid_->data_,
                                                                       vx_grid_->origin_,
                                                                       vx_grid_->num_cells_,
                                                                       vx_grid_->stride1_, vx_grid_->stride2_, vx_grid_->resolution_));

          typedef thrust::device_vector<int>::iterator IndexIterator;
          thrust::device_vector<int> valid_indices(indices_iter);

          // use make_counting_iterator to define the sequence [0, 8)
          IndexIterator indices_end = thrust::copy_if(first,
                                                      last,
                                                      invalid.begin(),
                                                      valid_indices.begin(),
                                                      thrust::identity<int>());
          //std::cout << "Number of invalid cells..." << invalid << " over:" << indices_iter << std::endl;

          indices_iter = (indices_end - valid_indices.begin());
          thrust::device_vector<point_statistics> points_stats(indices_iter);
          thrust::transform (valid_indices.begin(),
                             valid_indices.end(),
                             points_stats.begin (),
                             transformAndComputeOverlapPlusRegError<PointType, VoxelType>(input_dev_ptr_, hyp_dev_ptr_,
                                                                                             input_->size(), num_nodes_, inliers_threshold_,
                                                                                             vx_grid_->data_,
                                                                                             vx_grid_->origin_,
                                                                                             vx_grid_->num_cells_,
                                                                                             vx_grid_->stride1_, vx_grid_->stride2_, vx_grid_->resolution_));*/

          thrust::device_vector<point_statistics> points_stats(indices_iter);
          thrust::device_vector<int> hyp_keys(indices_iter);
          thrust::transform(first,last,hyp_keys.begin(),generateKeys(input_->size()));
          thrust::device_vector<int> output_keys(indices_iter);

          thrust::transform (first,
                             last,
                             points_stats.begin (),
                             transformAndComputeOverlapPlusRegError<PointType, VoxelType>(input_dev_ptr_, target_dev_ptr_, hyp_dev_ptr_,
                                                                                             input_->size(), num_nodes_, inliers_threshold_,
                                                                                             vx_grid_->data_,
                                                                                             vx_grid_->origin_,
                                                                                             vx_grid_->num_cells_,
                                                                                             vx_grid_->stride1_, vx_grid_->stride2_, vx_grid_->resolution_));

          thrust::device_vector<point_statistics> output_values(indices_iter);
          thrust::reduce_by_key(hyp_keys.begin(),
                                hyp_keys.end(),
                                points_stats.begin(),
                                output_keys.begin(),
                                output_values.begin(),
                                binary_pred,
                                sumPointStatistics()
                                );

          thrust::host_vector<point_statistics> output_values_host;
          output_values_host = output_values;

          for(size_t i=0; i < num_nodes_iter; i++)
          {
            hyps_[ii * num_nodes_per_iteration + i].overlap_ = ((point_statistics)(output_values_host[i])).overlap_;
            hyps_[ii * num_nodes_per_iteration + i].reg_error_ = ((point_statistics)(output_values_host[i])).reg_error_;
            hyps_[ii * num_nodes_per_iteration + i].color_reg_ = ((point_statistics)(output_values_host[i])).color_reg_;
          }
        }
      }
    }
  }
}

template class faat_pcl::cuda::registration::ICPWithGC<faat_pcl::xyz_rgb, faat_pcl::PropDistanceFieldVoxel>;
