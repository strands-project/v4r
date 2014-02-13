#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

namespace faat_pcl
{

  struct xyz_p
  {
    float x;
    float y;
    float z;
  };

  struct xyz_rgb
  {
    float x;
    float y;
    float z;
    bool rgb_set;
    float rgb;
  };

  struct Mat4f {
    float mat[4][4];
  };

  struct ICPNodeGPU
  {
    Mat4f transform_;
    float reg_error_;
    int overlap_;
    float color_reg_;
  };

  struct point_statistics
  {
    float reg_error_;
    float color_reg_;
    int overlap_;
  };

  struct PropDistanceFieldVoxel
  {
    int distance_square_; /**< Squared distance from the closest obstacle */
    int location_[3]; /**< Grid location of this voxel */
    int closest_point_[3]; /**< Closes obstacle from this voxel */
    bool occupied_;
    int idx_to_input_cloud_;
  };

  namespace cuda
  {
    namespace registration
    {

      template<typename Type>
      class VoxelGrid
      {
        private:
        public:

        float size_[3];
        float origin_[3];
        thrust::device_ptr<Type> data_;
        Type * raw_data_;
        int num_cells_[3];
        int num_cells_total_;
        int stride1_;
        int stride2_;
        float resolution_;

          VoxelGrid()
          {

          }

          ~VoxelGrid()
          {
            cudaFree(raw_data_);
          }

          void setOrigin(float x, float y, float z)
          {
            origin_[0] = x;
            origin_[1] = y;
            origin_[2] = z;
          }

          void setSize(float x, float y, float z)
          {
            size_[0] = x;
            size_[1] = y;
            size_[2] = z;
          }

          void setNumCells(int x, int y, int z)
          {
            num_cells_total_ = x * y * z;
            num_cells_[0] = x;
            num_cells_[1] = y;
            num_cells_[2] = z;

            stride1_ = num_cells_[1]*num_cells_[2];
            stride2_ = num_cells_[2];
          }

          void
          setResolution(float r)
          {
            resolution_ = r;
          }

          void setCells(Type * data)
          {
            cudaMalloc ((void **)&raw_data_, num_cells_total_ * sizeof(Type));
            cudaMemcpy (raw_data_, data, num_cells_total_ * sizeof(Type), cudaMemcpyHostToDevice);
            thrust::device_ptr<Type> dev_ptr (raw_data_);
            data_ = dev_ptr;
          }
      };

      template<typename PointType, typename VoxelType>
      class ICPWithGC
      {

        typedef typename std::vector<PointType> cloudType;
        float inliers_threshold_;

        cloudType * input_;
        cloudType * target_;

        thrust::device_ptr<PointType> input_dev_ptr_;
        PointType * raw_input_ptr_;

        thrust::device_ptr<ICPNodeGPU> hyp_dev_ptr_;
        ICPNodeGPU * raw_hyp_ptr_;
        ICPNodeGPU * hyps_;

        thrust::device_ptr<PointType> target_dev_ptr_;
        PointType * raw_target_ptr_;
        int num_nodes_;

        VoxelGrid<VoxelType> * vx_grid_;

      public:
        ICPWithGC();

        ~ICPWithGC()
        {
          cudaFree(raw_input_ptr_);
          cudaFree(raw_target_ptr_);
          cudaFree(raw_hyp_ptr_);
        }

        void setInliersThreshold(float t)
        {
          inliers_threshold_ = t;
        }

        void setInputCloud(cloudType * t)
        {
          input_ = t;

          cudaMalloc ((void **)&raw_input_ptr_, input_->size () * sizeof(PointType));
          PointType * input_pts = new PointType[input_->size ()];

          for (size_t j = 0; j < input_->size (); j++)
          {
            input_pts[j] = input_->at(j);
          }

          cudaMemcpy (raw_input_ptr_, input_pts, input_->size () * sizeof(PointType), cudaMemcpyHostToDevice);
          thrust::device_ptr<PointType> dev_ptr (raw_input_ptr_);
          input_dev_ptr_ = dev_ptr;
          delete[] input_pts;
        }

        void setTargetCloud(cloudType * t)
        {
          target_ = t;
          cudaMalloc ((void **)&raw_target_ptr_, target_->size () * sizeof(PointType));
          PointType * input_pts = new PointType[target_->size ()];

          for (size_t j = 0; j < target_->size (); j++)
          {
            input_pts[j] = target_->at(j);
          }

          cudaMemcpy (raw_target_ptr_, input_pts, target_->size () * sizeof(PointType), cudaMemcpyHostToDevice);
          thrust::device_ptr<PointType> dev_ptr (raw_target_ptr_);
          target_dev_ptr_ = dev_ptr;
          delete[] input_pts;
        }

        void setHypothesesToEvaluate(ICPNodeGPU * hyp, int elem)
        {

          if(raw_hyp_ptr_)
            cudaFree(raw_hyp_ptr_);

          cudaMalloc ((void **)&raw_hyp_ptr_, elem * sizeof(ICPNodeGPU));
          ICPNodeGPU * input_pts = new ICPNodeGPU[elem];

          for (int j = 0; j < elem; j++)
          {
            input_pts[j] = hyp[j];
          }

          cudaMemcpy (raw_hyp_ptr_, input_pts, elem * sizeof(ICPNodeGPU), cudaMemcpyHostToDevice);
          thrust::device_ptr<ICPNodeGPU> dev_ptr (raw_hyp_ptr_);
          hyp_dev_ptr_ = dev_ptr;
          delete[] input_pts;

          num_nodes_ = elem;
          hyps_ = hyp;
        }

        ICPNodeGPU * getHyps()
        {
          return hyps_;
        }

        void computeOverlapAndRegistrationError();

        void setVoxelGrid(faat_pcl::cuda::registration::VoxelGrid<VoxelType> * vx)
        {
          vx_grid_ = vx;
        }
      };
    }
  }
}
