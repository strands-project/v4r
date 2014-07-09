#include <faat_pcl/recognition/hv/cuda/ghv.h>
#include <stdio.h>

#include "cuda_runtime_api.h"
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <thrust/count.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <set>

namespace faat_pcl
{

    class CudaTiming
    {

        private:
            cudaEvent_t start, stop;
            float time;
            std::string text_;

        public:
            CudaTiming(std::string text)
            {
                text_ = text;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start, 0);
            }

            ~CudaTiming()
            {
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&time, start, stop);
                std::cout << text_ << " took:" << time << " ms" << std::endl;
            }
    };
}


static inline int divUp(int total, int grain) { return (total + grain - 1) / grain; }

__global__ void useClass(faat_pcl::recognition_cuda::XYZPointCloud *cudaClass)
{
    printf("lolo %d\n", cudaClass->width_);
};

void PrintFreeMemory()
{
    size_t mem_tot_0 = 0;
    size_t mem_free_0 = 0;
    cudaMemGetInfo(&mem_free_0, & mem_tot_0);
    int mem_free_mb = (float)mem_free_0 / 1024.f / 1024.f;
    float mem_tot_mb = (float)mem_tot_0 / 1024.f / 1024.f;
    std::cout<< "Free memory: "<< mem_free_mb << " " << mem_tot_mb << std::endl;
}

void faat_pcl::recognition_cuda::GHV::setSceneCloud(XYZPointCloud *cloud)
{
    scene_.upload(cloud->points, cloud->width_ * sizeof(xyz_p), cloud->height_, cloud->width_);
}

void faat_pcl::recognition_cuda::GHV::setSceneRGBValues(float3 * rgb_scene_values, int size)
{
    scene_rgb_values_.upload(rgb_scene_values, size);
}

void faat_pcl::recognition_cuda::GHV::setScenePointCloud(XYZPointCloud *cloud)
{
    scene_downsampled_.upload(cloud->points, cloud->width_ * cloud->height_);
    scene_downsampled_host_ = cloud;
}

void faat_pcl::recognition_cuda::GHV::setSceneNormals(XYZPointCloud * normals)
{
    scene_normals_host_ = normals;
}

void faat_pcl::recognition_cuda::GHV::setModelColors(std::vector<float3 *> & model_colors,
                                                     int models_size,
                                                     std::vector<int> & sizes_per_model)
{
    model_colours_device_array_ = new pcl::gpu::DeviceArray<float3>[models_size];
    cudaMalloc( (void**)&model_rgb_values_, sizeof(float3 *) * models_size);
    float3 ** mcolors_local = new float3 * [models_size];

    for(int i=0; i < models_size; i++)
    {
        model_colours_device_array_[i].upload(model_colors[i], sizes_per_model[i]);
        mcolors_local[i] = model_colours_device_array_[i].ptr();
    }

    cudaMemcpy(model_rgb_values_, mcolors_local, sizeof(float3 *) * models_size, cudaMemcpyHostToDevice);

    delete[] mcolors_local;
    color_exists_ = true;
}

void faat_pcl::recognition_cuda::GHV::setModelClouds(std::vector<XYZPointCloud *> & model_clouds,
                                                     std::vector<XYZPointCloud *> & model_normals)
{
    model_clouds_device_array_ = new pcl::gpu::DeviceArray<xyz_p>[model_clouds.size()];
    cudaMalloc( (void**)&model_clouds_, sizeof(xyz_p *) * model_clouds.size());
    xyz_p ** mclouds_local = new xyz_p * [model_clouds.size()];

    model_normals_device_array_ = new pcl::gpu::DeviceArray<xyz_p>[model_clouds.size()];
    cudaMalloc( (void**)&model_normals_, sizeof(xyz_p *) * model_clouds.size());
    xyz_p ** mnormals_local = new xyz_p * [model_clouds.size()];

    for(size_t i=0; i < model_clouds.size(); i++)
    {
        model_clouds_device_array_[i].upload(model_clouds[i]->points, model_clouds[i]->width_ * model_clouds[i]->height_);
        mclouds_local[i] = model_clouds_device_array_[i].ptr();

        model_normals_device_array_[i].upload(model_normals[i]->points, model_clouds[i]->width_ * model_clouds[i]->height_);
        mnormals_local[i] = model_normals_device_array_[i].ptr();
    }

    cudaMemcpy(model_clouds_, mclouds_local, sizeof(xyz_p *) * model_clouds.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(model_normals_, mnormals_local, sizeof(xyz_p *) * model_clouds.size(), cudaMemcpyHostToDevice);

    n_models_ = model_clouds.size();
    model_clouds_host_ = model_clouds;

    delete[] mclouds_local;
    delete[] mnormals_local;
}


void faat_pcl::recognition_cuda::GHV::setHypotheses(std::vector<HypothesisGPU> & hypotheses)
{
    hypotheses_ = hypotheses;

    HypothesisGPU * hypotheses_gpu = new HypothesisGPU[hypotheses_.size()];
    total_points_hypotheses_ = 0;

    for(size_t i=0; i < hypotheses_.size(); i++)
    {
        hypotheses_[i].size_ = model_clouds_host_[hypotheses_[i].model_idx_]->width_;
        hypotheses_gpu[i] = hypotheses_[i];
        hypotheses_gpu[i].size_ = model_clouds_host_[hypotheses_[i].model_idx_]->width_;
        total_points_hypotheses_ += model_clouds_host_[hypotheses_[i].model_idx_]->width_;
    }

    hypotheses_dev_.upload(hypotheses_gpu, hypotheses_.size());
}

void faat_pcl::recognition_cuda::GHV::freeMemory()
{
    cudaFree(model_clouds_);
    cudaFree(model_normals_);
    cudaFree(model_rgb_values_);

    for(size_t i=0; i < model_clouds_device_array_->size(); i++)
    {
        model_clouds_device_array_[i].release();
        model_normals_device_array_[i].release();
        model_colours_device_array_[i].release();
    }

    delete[] model_clouds_device_array_;
    delete[] model_normals_device_array_;
    delete[] model_colours_device_array_;

    scene_.release();
    scene_downsampled_.release();
    scene_rgb_values_.release();

    visible_extended_points.release();
    extended_points.release();
    outlier_extended_points_.release();
    hypotheses_dev_.release();

    PrintFreeMemory();
}

namespace faat_pcl
{

    namespace recognition_cuda
    {

        template<typename PointType>
        inline __host__ __device__ PointType transform(Mat4f rot_, PointType pt)
        {
            PointType min_tp;
            min_tp.x = rot_.mat[0][0] * pt.x + rot_.mat[0][1] * pt.y + rot_.mat[0][2] * pt.z + rot_.mat[0][3];
            min_tp.y = rot_.mat[1][0] * pt.x + rot_.mat[1][1] * pt.y + rot_.mat[1][2] * pt.z + rot_.mat[1][3];
            min_tp.z = rot_.mat[2][0] * pt.x + rot_.mat[2][1] * pt.y + rot_.mat[2][2] * pt.z + rot_.mat[2][3];
            return min_tp;
        }

        template<typename PointType>
        inline __host__ __device__ PointType transformNormal(Mat4f rot_, PointType pt)
        {
            PointType min_tp;
            min_tp.x = rot_.mat[0][0] * pt.x + rot_.mat[0][1] * pt.y + rot_.mat[0][2] * pt.z;
            min_tp.y = rot_.mat[1][0] * pt.x + rot_.mat[1][1] * pt.y + rot_.mat[1][2] * pt.z;
            min_tp.z = rot_.mat[2][0] * pt.x + rot_.mat[2][1] * pt.y + rot_.mat[2][2] * pt.z;
            return min_tp;
        }

        template<typename PointType>
        inline __host__ __device__ float dot(PointType pt1, PointType pt2)
        {
            return pt1.x * pt2.x + pt1.y * pt2.y + pt1.z * pt2.z;
        }

        inline __host__ __device__ float sqrDist(xyz_p & a, xyz_p & b)
        {
            return   (a.x - b.x) * (a.x - b.x) +
                     (a.y - b.y) * (a.y - b.y) +
                     (a.z - b.z) * (a.z - b.z);
        }

        inline __host__ __device__ float sqrDist_color(const float3 & a, const float3 & b)
        {
            return   (a.x - b.x) * (a.x - b.x) +
                     (a.y - b.y) * (a.y - b.y) +
                     (a.z - b.z) * (a.z - b.z);
        }

        inline __host__ __device__ float rad2deg(float rad)
        {
            return 57.2957795 * rad;
        }

        struct checkVisibility
        {
            const xyz_p * scene;
            int scene_width, scene_height;
            xyz_p ** model_clouds;
            xyz_p ** model_normals;
            float3 ** colors;
            const HypothesisGPU * hypotheses;
            int n_hypotheses;
            bool color_exist;
            mutable bool * visible;
            mutable pointExtended * visible_points;
            mutable float * angles_;

            int n_models;
            int max_size_model;

            float f_;
            int cx_, cy_;
            float threshold_;

            xyz_p unit_z;

            checkVisibility()
            {
                f_ = 525.f;
                cx_ = 320;
                cy_ = 240;
                threshold_ = 0.0075f;
                unit_z.x = unit_z.y = 0;
                unit_z.z = -1;
            }

            __device__ __forceinline__ void operator()(int idx) const
            {

                visible[idx] = false;

                int hyp_id = idx / max_size_model;
                int point_id = idx % max_size_model;
                int model_id = hypotheses[hyp_id].model_idx_;

                pointExtended pe;
                pe.set = false;
                pe.hypotheses_id = hyp_id;
                pe.point_id = point_id;
                pe.model_id = model_id;

                visible_points[idx] = pe;

                if(hyp_id >= n_hypotheses)
                    return;


                if(point_id >= hypotheses[hyp_id].size_)
                    return;

                xyz_p p_trans = transform<xyz_p>(hypotheses[hyp_id].transform_, model_clouds[model_id][point_id]);
                xyz_p normal_trans = transformNormal<xyz_p>(hypotheses[hyp_id].transform_, model_normals[model_id][point_id]);

                float dot_p = dot(normal_trans, unit_z);
                float angle = rad2deg(acos(dot_p));
                angles_[idx] = angle;

                //normal dot product with view ray should be positive
                if(dot_p < 0.1f)
                    return;

                int u = (int)(f_ * p_trans.x / p_trans.z + cx_);
                int v = (int)(f_ * p_trans.y / p_trans.z + cy_);

                //out of bounds
                if ((u >= scene_width) || (v >= scene_height) || (u < 0) || (v < 0))
                   return;

                xyz_p p_scene = scene[v * scene_width + u];

                //invalid depth
                if(!isfinite(p_scene.z))
                    return;

                //behind point cloud
                if ((p_trans.z - p_scene.z) > threshold_)
                    return;

                //otherwise
                visible[idx] = true;

                visible_points[idx].set = true;
                visible_points[idx].p = p_trans;

                if(color_exist)
                {
                    visible_points[idx].color = colors[model_id][point_id];
                    visible_points[idx].color_set_ = true;
                }
            }
        };

        __global__ void visibilityCheckKernel(const checkVisibility flip)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            flip(idx);
        }

        struct computeDistFromExplainedToUnexplainedInNeighborhood
        {
            const xyz_p * scene_; //scene
            const explainedSPExtended * explained_extended_;
            const int * nn_sizes_; //size(unique(explained_points)
            const int * nn_; //size(unique(explained_points) * max_elems_
            const bool * scene_point_explained_by_hypothesis_; //size(hypotheses.size() * scene_.size())
            const int * scene_to_unique_exp_;

            mutable float * distance_explained_to_unexplained_;
            int total_size_;
            int max_elems_;
            int scene_size_;

            computeDistFromExplainedToUnexplainedInNeighborhood()
            {

            }

            __device__ __forceinline__ void operator()(int idx) const
            {
                //idx goes from zero to size(explained_extended_) * max_elems_
                //printf("%d %d\n", idx, total_size_);
                if(idx >= total_size_)
                    return;

                int idx_indices = idx / max_elems_;
                int nn = idx % max_elems_;

                int s_idx = explained_extended_[idx_indices].scene_idx;

                //i need the index from s_idx to nn_sizes_ which maps to unique(explained_points)
                int idx_unique_exp = scene_to_unique_exp_[s_idx];
                if(nn >= nn_sizes_[idx_unique_exp])
                {
                    distance_explained_to_unexplained_[idx] = -1.f;
                    return;
                }

                int hyp_id = explained_extended_[idx_indices].hypotheses_id;
                int idx_scene_nn = nn_[idx_unique_exp * max_elems_ + nn];

                if(scene_point_explained_by_hypothesis_[hyp_id * scene_size_ + idx_scene_nn])
                {
                    distance_explained_to_unexplained_[idx] = 5.f;
                    return;
                }

                distance_explained_to_unexplained_[idx] = 0.5f;

                //printf("%d %d %d %d\n", idx_indices, nn, s_idx, hyp_id);

                /*int idx_indices = idx / max_elems_;
                int nn = idx % max_elems_;
                int idx_vep = indices_[idx_indices];*/

            }
        };

        __global__ void computeDistFromExplainedToUnexplainedInNeighborhoodKernel(const computeDistFromExplainedToUnexplainedInNeighborhood flip)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            flip(idx);
        }

        struct computeInlierWeights
        {
            const xyz_p * scene_; //scene
            const float3 * scene_colors_;
            const pointExtended * visible_extended_points; //visible model extended points
            const int * indices_; //pointer to visible_extended_points
            int max_elems_;
            mutable float * weights_; //size(indices_) * max_elems_
            const int * nn_sizes_; //size(visible_extended_points)
            const int * nn_; //size(visible_extended_points) * max_elems_
            float inliers_gaussian_;
            //mutable thrust::pair<int, int> * keys_; //only necessary if using chunks
            mutable int * keys_scene_;
            mutable int * keys_hyp_id_;
            int problem_size_;

            computeInlierWeights(float inlier_thres=0.01f)
            {
                inliers_gaussian_ = inlier_thres * inlier_thres;
            }

            __device__ __forceinline__ void operator()(int idx) const
            {
                //idx goes from zero to size(indices_) * max_elems_
                if(idx >= problem_size_)
                    return;

                int idx_indices = idx / max_elems_;
                int nn = idx % max_elems_;
                int idx_vep = indices_[idx_indices];
                pointExtended model_p = visible_extended_points[idx_vep];
                //keys_[idx].first = model_p.hypotheses_id;
                keys_hyp_id_[idx] = model_p.hypotheses_id;
                weights_[idx] = -1;
                //keys_[idx].second = -1;
                keys_scene_[idx] = -1;

                if(nn >= nn_sizes_[idx_vep])
                {
                    //nn does not need to be processed
                    return;
                }

                int idx_scene = nn_[idx_vep * max_elems_ + nn];

                xyz_p scene_p = scene_[idx_scene];

                float dist = sqrDist(model_p.p, scene_p);
                float d_weight = exp( -(dist / inliers_gaussian_) );

                float color_w = 1.f;

                float sigma_y, sigma;
                sigma_y = 0.5;
                sigma = 0.3 * 0.3;

                if(model_p.color_set_)
                {
                    float3 yuvm = model_p.color;
                    float3 yuvs = scene_colors_[idx_scene];
                    color_w = exp ((-0.5f * (yuvm.x - yuvs.x) * (yuvm.x - yuvs.x)) / (sigma_y));
                    color_w *= exp ((-0.5f * (yuvm.y - yuvs.y) * (yuvm.y - yuvs.y)) / (sigma));
                    color_w *= exp ((-0.5f * (yuvm.z - yuvs.z) * (yuvm.z - yuvs.z)) / (sigma));
                }

                float w = d_weight * color_w;
                weights_[idx] = w;

                //keys_[idx].second = idx_scene;
                keys_scene_[idx] = idx_scene;
                keys_hyp_id_[idx] = model_p.hypotheses_id;
            }
        };

        __global__ void computeInlierWeightsKernel(const computeInlierWeights flip)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            flip(idx);
        }

        struct computeOutlierWeights
        {
            const float * angles_;
            const int * indices_; //pointer to angles_
            mutable float * weights_;
            int problem_size_;
            float max_angle_;
            float w_;

            __device__ __forceinline__ void operator()(int idx) const
            {
                if(idx >= problem_size_)
                    return;

                //idx goes from zero to size(indices_) * max_elems_
                int idx_angles = indices_[idx];
                float angle = angles_[idx_angles];
                if(angle > max_angle_)
                {
                    weights_[idx] *= w_;
                }
            }
        };

        __global__ void computeOutlierWeightsKernel(const computeOutlierWeights flip)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            flip(idx);
        }

        __global__ void xyzp_to_float4(xyz_p * xyz_p_cloud,
                                       float4 * float4_cloud,
                                       int size_cloud)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if(idx >= size_cloud)
                return;

            float4 p;
            p.x = xyz_p_cloud[idx].x;
            p.y = xyz_p_cloud[idx].y;
            p.z = xyz_p_cloud[idx].z;
            p.w = 0;
            float4_cloud[idx] = p;
        }

        __global__ void xyzp_to_float4_with_indices(xyz_p * xyz_p_cloud,
                                       float4 * float4_cloud,
                                       int * indices,
                                       int idx_size)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;

            if(idx >= idx_size)
                return;

            float4 p;
            p.x = xyz_p_cloud[indices[idx]].x;
            p.y = xyz_p_cloud[indices[idx]].y;
            p.z = xyz_p_cloud[indices[idx]].z;
            p.w = 0;
            float4_cloud[idx] = p;
        }

        __global__ void pointExtended_to_float4(pointExtended * xyz_p_cloud,
                                                float4 * float4_cloud,
                                                int size_cloud)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if(idx >= size_cloud)
                return;

            float4 p;
            p.x = xyz_p_cloud[idx].p.x;
            p.y = xyz_p_cloud[idx].p.y;
            p.z = xyz_p_cloud[idx].p.z;
            p.w = 0;
            float4_cloud[idx] = p;
        }

        struct visible_point
        {
        __host__ __device__
        bool operator()(const pointExtended x)
        {
            return (x.set);
        }
        };

        struct is_zero
        {
        __host__ __device__
        bool operator()(const int x)
        {
          return x == 0;
        }
        };

        struct is_not_zero
        {
        __host__ __device__
        bool operator()(const int x)
        {
          return x != 0;
        }
        };


        //sorting based on hypotheses_id and scene_point
        struct cmpHypScenePoint
        {
        __host__ __device__
        bool operator()(const thrust::pair<int, int> & x, const thrust::pair<int, int> & y)
        {
            if(x.first == y.first)
            {
                return x.second < y.second;
            }

            return x.first < y.first;
        }
        };

        struct binaryPredHypScenePoints
        {
        __host__ __device__
        bool operator()(const thrust::pair<int, int> & x, const thrust::pair<int, int> & y)
        {
            return (x.first == y.first) && (x.second == y.second);
        }
        };

        struct binaryPredHypScenePointsTuple
        {
        __host__ __device__
        bool operator()(const thrust::tuple<int, int> & x, const thrust::tuple<int, int> & y)
        {
            return (thrust::get<0>(x) == thrust::get<0>(y)) && (thrust::get<1>(x) == thrust::get<1>(y));
        }
        };

        struct binaryOpHypScenePoints
        {
        __host__ __device__
        float operator()(const float & x, const float & y)
        {
            if(y > x)
                return y;

            return x;
        }
        };

        __global__ void extract_hyp_id(pointExtended * xyz_p_cloud,
                                       int * hyp_id,
                                       int size)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if(idx >= size)
                return;

            hyp_id[idx] = xyz_p_cloud[idx].hypotheses_id;
        }

        __global__ void extract_hyp_id_with_indices(pointExtended * xyz_p_cloud,
                                       int * hyp_id,
                                       int * indices,
                                       int size)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if(idx >= size)
                return;
            hyp_id[idx] = xyz_p_cloud[indices[idx]].hypotheses_id;
        }

        struct computeExplainedPointsOpt
        {
            const int * explained_by_RM_;
            mutable int * explained_by_RM_moves_;

            const explainedSPExtended * explained_extended_;
            const float * explainedSPWeights_;

            mutable float * increment_per_point_explained_;
            mutable float * increment_per_point_multiple_assignment_;

            int total_size_;
            const int * solution_;
            int scene_size_;

            computeExplainedPointsOpt()
            {

            }

            __device__ __forceinline__ void operator()(int idx) const
            {
                //idx goes from zero to size(explained_extended_) * max_elems_
                //printf("%d %d\n", idx, total_size_);
                if(idx >= total_size_)
                    return;

                int s_idx = explained_extended_[idx].scene_idx;
                int h_id = explained_extended_[idx].hypotheses_id;
                int s_idx_moves = h_id * scene_size_ + s_idx;

                //if hypothesis is deactivated, we are adding now (1), otherwise removing (-1)
                int add_int = (solution_[h_id] == 0) ? 1 : -1;
                float add_float = explainedSPWeights_[idx] * add_int;
                bool prev_dup = explained_by_RM_[s_idx] > 1;
                int new_exp = explained_by_RM_[s_idx] + add_int;
                explained_by_RM_moves_[s_idx_moves] = new_exp;

                increment_per_point_explained_[idx] += add_float;
                if((new_exp > 1) && prev_dup)
                {
                    //still a duplicate (increment or decrement one), we are adding
                    increment_per_point_multiple_assignment_[idx] += add_int;
                }
                else if( (new_exp == 1) && prev_dup)
                {
                    //was a duplicate before, now its not
                    increment_per_point_multiple_assignment_[idx] -= 2;
                }
                else if( (new_exp > 1) && !prev_dup)
                {
                    //it was not a duplicate but it is now
                    increment_per_point_multiple_assignment_[idx] += 2;
                }
            }
        };

        __global__ void computeExplainedPointsOptKernel(const computeExplainedPointsOpt flip)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            flip(idx);
        }

        struct computeUnexplainedPointsOpt
        {
            const double * unexplained_by_RM_;
            const int * explained_by_RM_moves_;
            mutable double * unexplained_by_RM_moves_;

            const explainedSPExtended * unexplained_extended_;
            const float * unexplainedSPWeights_;

            mutable double * increment_per_point_unexplained_;

            int total_size_;
            const int * solution_;
            int scene_size_;

            computeUnexplainedPointsOpt()
            {

            }

            __device__ __forceinline__ void operator()(int idx) const
            {
                //idx goes from zero to size(explained_extended_) * max_elems_
                //printf("%d %d\n", idx, total_size_);
                if(idx >= total_size_)
                    return;

                int s_idx = unexplained_extended_[idx].scene_idx;
                int h_id = unexplained_extended_[idx].hypotheses_id;
                int s_idx_moves = h_id * scene_size_ + s_idx;

                //if hypothesis is deactivated, we are adding now (1), otherwise removing (-1)
                int add_int = (solution_[h_id] == 0) ? 1 : -1;
                float add_float = unexplainedSPWeights_[idx] * add_int;

                bool prev_unexplained = (unexplained_by_RM_[s_idx]) && (explained_by_RM_moves_[s_idx_moves] == 0);
                unexplained_by_RM_moves_[s_idx_moves] += add_int * add_float;

                if(add_int < 0)
                {
                    if (prev_unexplained)
                    {
                        increment_per_point_unexplained_[idx] -= add_float;
                    }
                }
                else
                {
                    if ((explained_by_RM_moves_[s_idx_moves] == 0))
                    {
                        increment_per_point_unexplained_[idx] += add_float;
                    }
                }
            }
        };

        __global__ void computeUnexplainedPointsOptKernel(const computeUnexplainedPointsOpt flip)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            flip(idx);
        }

        struct computeUnexplainedPointsOptWithExplainedPoints
        {
            const double * unexplained_by_RM_;
            const int * explained_by_RM_moves_;
            mutable double * unexplained_by_RM_moves_;

            const explainedSPExtended * explained_extended_;
            mutable double * increment_per_point_unexplained_;

            int total_size_;
            const int * solution_;
            int scene_size_;

            computeUnexplainedPointsOptWithExplainedPoints()
            {

            }

            __device__ __forceinline__ void operator()(int idx) const
            {
                //idx goes from zero to size(explained_extended_) * max_elems_
                //printf("%d %d\n", idx, total_size_);
                if(idx >= total_size_)
                    return;

                int s_idx = explained_extended_[idx].scene_idx;
                int h_id = explained_extended_[idx].hypotheses_id;
                int s_idx_moves = h_id * scene_size_ + s_idx;

                //if hypothesis is deactivated, we are adding now (1), otherwise removing (-1)
                int add_int = (solution_[h_id] == 0) ? 1 : -1;

                if(add_int < 0)
                {
                    if ( (explained_by_RM_moves_[s_idx_moves] == 0) && (unexplained_by_RM_moves_[s_idx_moves] > 0))
                    {
                        increment_per_point_unexplained_[idx] += unexplained_by_RM_moves_[s_idx_moves];
                    }
                }
                else
                {
                    if ((explained_by_RM_moves_[s_idx_moves] == 1) && (unexplained_by_RM_moves_[s_idx_moves] > 0))
                    {
                        increment_per_point_unexplained_[idx] -= unexplained_by_RM_moves_[s_idx_moves];
                    }
                }
            }
        };

        __global__ void computeUnexplainedPointsOptWithExplainedPointsKernel(const computeUnexplainedPointsOptWithExplainedPoints flip)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            flip(idx);
        }
    }
}

thrust::device_vector<int> faat_pcl::recognition_cuda::GHV::getHypothesesIdsFromPointExtendedWithIndices(thrust::device_vector<int> & indices,
                                                                                                         pcl::gpu::DeviceArray<pointExtended> & points)
{
    thrust::device_vector<int> hypotheses_ids;
    hypotheses_ids.resize(indices.size());
    {

        int threads = 512;
        int blocks = divUp((int)indices.size(), threads);

        int * raw_ptr = thrust::raw_pointer_cast(hypotheses_ids.data());
        int * indices_raw_ptr = thrust::raw_pointer_cast(indices.data());
        extract_hyp_id_with_indices<<<blocks, threads>>>(points.ptr(), raw_ptr, indices_raw_ptr, (int)indices.size());
        cudaDeviceSynchronize();
    }

    return hypotheses_ids;
}

thrust::device_vector<int> faat_pcl::recognition_cuda::GHV::getHypothesesIdsFromPointExtended(thrust::device_vector<int> & indices,
                                                                                              pcl::gpu::DeviceArray<pointExtended> & points)
{
    thrust::device_vector<int> hypotheses_ids;
    hypotheses_ids.resize(indices.size());
    {

        int threads = 512;
        int blocks = divUp((int)indices.size(), threads);

        int * raw_ptr = thrust::raw_pointer_cast(hypotheses_ids.data());
        extract_hyp_id<<<blocks, threads>>>(points.ptr(), raw_ptr, (int)indices.size());
        cudaDeviceSynchronize();
    }

    return hypotheses_ids;
}

void faat_pcl::recognition_cuda::GHV::optimize()
{
    thrust::device_vector<int> solution_(hypotheses_.size(), 0); //all deactivated first

    //current solution explained and unexplained
    thrust::device_vector<int> explained_by_RM_(scene_downsampled_.size(), 0);
    thrust::device_vector<double> unexplained_by_RM_(scene_downsampled_.size(), 0);

    //explained stuff
    size_t num_exp = 0;
    for(size_t i=0; i < hypotheses_.size(); i+=1)
    {
        num_exp += explained_points_[i].size();
        if(explained_points_[i].size() == 0)
        {
            std::cout << "WARN: No points being explained by hypothesis:" << i << std::endl;
        }
    }

    thrust::host_vector<int> hypotheses_keys(num_exp);

    thrust::host_vector<explainedSPExtended> explainedSP(num_exp);
    thrust::host_vector<float> explainedSP_weights(num_exp);

    num_exp = 0;
    for(size_t i=0; i < hypotheses_.size(); i+=1)
    {
        for(size_t k=0; k < explained_points_[i].size(); k++)
        {
            explainedSP[num_exp].hypotheses_id = i;
            explainedSP[num_exp].scene_idx = explained_points_[i][k];

            hypotheses_keys[num_exp] = i;
            explainedSP_weights[num_exp] = explained_points_weights_[i][k];
            num_exp++;
        }
    }

    thrust::device_vector<int> hypotheses_keys_dev = hypotheses_keys;
    thrust::device_vector<explainedSPExtended> explainedSP_dev = explainedSP;
    thrust::device_vector<float> explainedSP_weights_dev = explainedSP_weights;

    explainedSPExtended * exp_dev_raw_ptr = thrust::raw_pointer_cast(explainedSP_dev.data());
    float * exp_weights_raw__ptr = thrust::raw_pointer_cast(explainedSP_weights_dev.data());

    //unexplained (clutter) stuff
    size_t num_unexp = 0;
    for(size_t i=0; i < hypotheses_.size(); i+=1)
    {
        num_unexp += unexplained_points_[i].size();
    }

    thrust::host_vector<int> hypotheses_keys_unexplained(num_unexp);
    thrust::host_vector<explainedSPExtended> unexplainedSP(num_unexp);
    thrust::host_vector<float> unexplainedSP_weights(num_unexp);

    num_unexp = 0;
    for(size_t i=0; i < hypotheses_.size(); i+=1)
    {
        for(size_t k=0; k < unexplained_points_[i].size(); k++)
        {
            unexplainedSP[num_unexp].hypotheses_id = i;
            unexplainedSP[num_unexp].scene_idx = unexplained_points_[i][k];

            hypotheses_keys_unexplained[num_unexp] = i;
            unexplainedSP_weights[num_unexp] = unexplained_points_weights_[i][k];
            num_unexp++;
        }
    }

    thrust::device_vector<int> hypotheses_keys_unexplained_dev = hypotheses_keys_unexplained;
    thrust::device_vector<explainedSPExtended> unexplainedSP_dev = unexplainedSP;
    thrust::device_vector<float> unexplainedSP_weights_dev = unexplainedSP_weights;

    std::cout << "Number of explained points:" << explainedSP.size() << std::endl;
    std::cout << "Number of unexplained points:" << unexplainedSP.size() << std::endl;

    float current_best_cost = 0.f;
    float good_info = 0.f;
    float bad_info = 0.f;
    float ma_info = 0.f;
    float active_penalty = 10.f;
    int active_hypotheses = 0;
    float unexplained_info = 0.f;

    thrust::host_vector<int> solution_host;

    thrust::device_vector<int> explained_by_RM_moves_(scene_downsampled_.size() * hypotheses_.size(), 0);
    thrust::device_vector<double> unexplained_by_RM_moves_(scene_downsampled_.size() * hypotheses_.size(), 0);

    int i=0;
    while(i < 50)
    {

        CudaTiming iter_time("iteration\n");
        //scene * number of moves (hypotheses_.size() if no replace moves)

        /************************************************************************/
        /************************ explained points and MA ***********************/
        /************************************************************************/

        thrust::device_vector<float> increment_per_point_explained(explainedSP_dev.size(), 0.f);
        thrust::device_vector<float> increment_per_point_ma(explainedSP_dev.size(), 0.f);

        float * output_exp_raw__ptr = thrust::raw_pointer_cast(increment_per_point_explained.data());
        float * output_ma_raw__ptr = thrust::raw_pointer_cast(increment_per_point_ma.data());

        int threads = 512;
        int blocks = divUp((int)explainedSP.size(), threads);

        computeExplainedPointsOpt cepo;
        //scene
        cepo.explained_by_RM_ = thrust::raw_pointer_cast(explained_by_RM_.data());
        cepo.explained_by_RM_moves_ = thrust::raw_pointer_cast(explained_by_RM_moves_.data());
        cepo.scene_size_ = scene_downsampled_.size();
        //hypotheses
        cepo.explained_extended_ = exp_dev_raw_ptr;
        cepo.explainedSPWeights_ = exp_weights_raw__ptr;
        //output
        cepo.increment_per_point_explained_ = output_exp_raw__ptr;
        cepo.increment_per_point_multiple_assignment_ = output_ma_raw__ptr;
        //total size and current solution
        cepo.total_size_ = explainedSP.size();
        cepo.solution_ = thrust::raw_pointer_cast(solution_.data());

        computeExplainedPointsOptKernel<<<blocks, threads>>>(cepo);
        cudaDeviceSynchronize();

        //reduce_by_key (summing up values_exp using hypotheses_ids as key)
        thrust::device_vector<float> increments_explained_per_hypothesis(explainedSP.size(), 0.f);
        thrust::device_vector<float> increments_ma_per_hypothesis(explainedSP.size(), 0.f);
        thrust::host_vector<float> increments_explained_per_hypothesis_host;
        thrust::host_vector<float> increments_ma_per_hypothesis_host;

        thrust::device_vector<int> output_keys(explainedSP.size());

        typedef thrust::device_vector<double>::iterator IdxIter3;
        typedef thrust::device_vector<float>::iterator IdxIter2;
        typedef thrust::device_vector< int >::iterator IdxIter1;

        thrust::pair< IdxIter1 , IdxIter2 > new_end1 = thrust::reduce_by_key(   hypotheses_keys_dev.begin(),
                                                                                  hypotheses_keys_dev.end(),
                                                                                  increment_per_point_explained.begin(),
                                                                                  output_keys.begin(),
                                                                                  increments_explained_per_hypothesis.begin());

        int size_not_empty = (new_end1.first - output_keys.begin());
        if(size_not_empty != hypotheses_.size())
        {
            increments_explained_per_hypothesis_host.resize(hypotheses_.size(), 0);
            for(size_t h=0; h < size_not_empty; h++)
            {
                increments_explained_per_hypothesis_host[output_keys[h]] = increments_explained_per_hypothesis[h];
            }
        }
        else
        {
            increments_explained_per_hypothesis.resize(hypotheses_.size());
            increments_explained_per_hypothesis_host = increments_explained_per_hypothesis;
        }

        thrust::pair< IdxIter1 , IdxIter2 > new_end2 = thrust::reduce_by_key(hypotheses_keys_dev.begin(),
                                                                             hypotheses_keys_dev.end(),
                                                                             increment_per_point_ma.begin(),
                                                                             output_keys.begin(),
                                                                             increments_ma_per_hypothesis.begin());

        int size_not_empty2 = (new_end2.first - output_keys.begin());
        if(size_not_empty2 != hypotheses_.size())
        {
            increments_ma_per_hypothesis_host.resize(hypotheses_.size(), 0);
            for(size_t h=0; h < size_not_empty2; h++)
            {
                increments_ma_per_hypothesis_host[output_keys[h]] = increments_ma_per_hypothesis[h];
            }
        }
        else
        {
            increments_ma_per_hypothesis.resize(hypotheses_.size());
            increments_ma_per_hypothesis_host = increments_ma_per_hypothesis;
        }

        /************************************************************************/
        /************************ clutter ***************************************/
        /************************************************************************/

        thrust::host_vector<double> increments_explained_clutter_per_hypothesis_host;
        thrust::host_vector<double> increments_unexplained_clutter_per_hypothesis_host;

        //with unexplained points
        thrust::device_vector<double> increments_unexplained_per_hypothesis(unexplainedSP.size(), 0.f);

        {
            thrust::device_vector<double> increment_per_unexplained_point(unexplainedSP.size(), 0.f);

            computeUnexplainedPointsOpt cupo;
            cupo.explained_by_RM_moves_ = thrust::raw_pointer_cast(explained_by_RM_moves_.data());
            cupo.unexplained_by_RM_ = thrust::raw_pointer_cast(unexplained_by_RM_.data());
            cupo.unexplained_by_RM_moves_ = thrust::raw_pointer_cast(unexplained_by_RM_moves_.data());
            cupo.unexplained_extended_ = thrust::raw_pointer_cast(unexplainedSP_dev.data());
            cupo.unexplainedSPWeights_ = thrust::raw_pointer_cast(unexplainedSP_weights_dev.data());
            cupo.total_size_ = unexplainedSP.size();
            cupo.solution_ = thrust::raw_pointer_cast(solution_.data());
            cupo.increment_per_point_unexplained_ = thrust::raw_pointer_cast(increment_per_unexplained_point.data());
            cupo.scene_size_ = scene_downsampled_.size();

            int threads = 512;
            int blocks = divUp((int)unexplainedSP.size(), threads);

            computeUnexplainedPointsOptKernel<<<blocks, threads>>>(cupo);
            cudaDeviceSynchronize();

            thrust::device_vector<int> output_keys_un(unexplainedSP.size());

            thrust::pair< IdxIter1 , IdxIter3 > new_end1 = thrust::reduce_by_key(hypotheses_keys_unexplained_dev.begin(),
                                                                                  hypotheses_keys_unexplained_dev.end(),
                                                                                  increment_per_unexplained_point.begin(),
                                                                                  output_keys_un.begin(),
                                                                                  increments_unexplained_per_hypothesis.begin());

            int size_not_empty = (new_end1.first - output_keys_un.begin());
            if(size_not_empty != hypotheses_.size())
            {
                increments_unexplained_clutter_per_hypothesis_host.resize(hypotheses_.size(), 0);
                for(size_t h=0; h < size_not_empty; h++)
                {
                    increments_unexplained_clutter_per_hypothesis_host[output_keys_un[h]] = increments_unexplained_per_hypothesis[h];
                }
            }
            else
            {
                increments_unexplained_per_hypothesis.resize(hypotheses_.size());
                increments_unexplained_clutter_per_hypothesis_host = increments_unexplained_per_hypothesis;
            }
        }

        //with explained points
        thrust::device_vector<double> increments_clutter_explained_per_hypothesis(explainedSP.size(), 0.f);

        {
            thrust::device_vector<double> increment_per_unexplained_point(explainedSP.size(), 0.f);

            computeUnexplainedPointsOptWithExplainedPoints cupo;

            cupo.explained_by_RM_moves_ = thrust::raw_pointer_cast(explained_by_RM_moves_.data());
            cupo.unexplained_by_RM_moves_ = thrust::raw_pointer_cast(unexplained_by_RM_moves_.data());
            cupo.explained_extended_ = exp_dev_raw_ptr;
            cupo.total_size_ = explainedSP.size();
            cupo.solution_ = thrust::raw_pointer_cast(solution_.data());
            cupo.increment_per_point_unexplained_ = thrust::raw_pointer_cast(increment_per_unexplained_point.data());
            cupo.scene_size_ = scene_downsampled_.size();

            {
                int threads = 512;
                int blocks = divUp((int)explainedSP.size(), threads);

                computeUnexplainedPointsOptWithExplainedPointsKernel<<<blocks, threads>>>(cupo);
                cudaDeviceSynchronize();
            }

            thrust::device_vector<int> output_keys_un(explainedSP.size());

            thrust::pair< IdxIter1 , IdxIter3 > new_end1 = thrust::reduce_by_key(hypotheses_keys_dev.begin(),
                                  hypotheses_keys_dev.end(),
                                  increment_per_unexplained_point.begin(),
                                  output_keys_un.begin(),
                                  increments_clutter_explained_per_hypothesis.begin());

            int size_not_empty = (new_end1.first - output_keys_un.begin());
            if(size_not_empty != hypotheses_.size())
            {
                increments_explained_clutter_per_hypothesis_host.resize(hypotheses_.size(), 0);
                for(size_t h=0; h < size_not_empty; h++)
                {
                    increments_explained_clutter_per_hypothesis_host[output_keys_un[h]] = increments_clutter_explained_per_hypothesis[h];
                }
            }
            else
            {
                increments_clutter_explained_per_hypothesis.resize(hypotheses_.size());
                increments_explained_clutter_per_hypothesis_host = increments_clutter_explained_per_hypothesis;
            }
        }

        /************************************************************************/
        /************************ cost function **********************************/
        /************************************************************************/

        CudaTiming time_rep("cost function and replicate");
        int idx_best = -1;

        solution_host = solution_;

        for(size_t h=0; h < increments_explained_per_hypothesis_host.size(); h++)
        {
            float exp = good_info + increments_explained_per_hypothesis_host[h];
            float dup = ma_info + increments_ma_per_hypothesis_host[h];
            float bad = bad_info + (outliers_per_hypothesis_[h] * ((solution_host[h] == 0) ? 1 : -1) );
            float act = (active_hypotheses + ((solution_host[h] == 0) ? 1 : -1)) * active_penalty;
            float un = unexplained_info + (increments_explained_clutter_per_hypothesis_host[h] + increments_unexplained_clutter_per_hypothesis_host[h]);
            float cost = (exp - bad - dup - act - un) * -1;
            if(cost < current_best_cost)
            {
                current_best_cost = cost;
                idx_best = h;
            }
        }

        if(idx_best == -1)
        {
            break;
        }

        good_info += increments_explained_per_hypothesis_host[idx_best];
        ma_info += increments_ma_per_hypothesis_host[idx_best];
        bad_info += (outliers_per_hypothesis_[idx_best] * ((solution_host[idx_best] == 0) ? 1 : -1) );
        active_hypotheses += ((solution_host[idx_best] == 0) ? 1 : -1);
        unexplained_info += (increments_explained_clutter_per_hypothesis_host[idx_best] + increments_unexplained_clutter_per_hypothesis_host[idx_best]);

        //std::cout << "current best cost:" << current_best_cost << " iteration:" << i << " " << idx_best << " activating:" << ((solution_host[idx_best] == 0) ? 1 : 0) << std::endl;
        //std::cout << good_info << " " << ma_info << " " << bad_info << " " << unexplained_info << " " << active_hypotheses << std::endl;

        /************************************************************************/
        /************************ update solution and status ********************/
        /************************************************************************/

        solution_[idx_best] = ((solution_host[i] == 0) ? 1 : 0);

        {
            CudaTiming t("copy explained_by_RM and unexplained_by_RM");
            //copy explained_by_RM_ according to the best move
            thrust::device_vector<int>::iterator first = explained_by_RM_moves_.begin() + idx_best * scene_downsampled_.size();
            thrust::device_vector<int>::iterator end = first + scene_downsampled_.size();

            thrust::copy(first, end, explained_by_RM_.begin());

            //replicate for all moves
            for(size_t h=0; h < hypotheses_.size(); h++)
            {
                thrust::device_vector<int>::iterator first = explained_by_RM_moves_.begin() + h * scene_downsampled_.size();
                thrust::copy(explained_by_RM_.begin(), explained_by_RM_.end(), first);
            }

            {
                //copy unexplained_by_RM_ according to the best move
                thrust::device_vector<double>::iterator first = unexplained_by_RM_moves_.begin() + idx_best * scene_downsampled_.size();
                thrust::device_vector<double>::iterator end = first + scene_downsampled_.size();

                thrust::copy(first, end, unexplained_by_RM_.begin());

                //replicate for all moves
                for(size_t h=0; h < hypotheses_.size(); h++)
                {
                    thrust::device_vector<double>::iterator first = unexplained_by_RM_moves_.begin() + h * scene_downsampled_.size();
                    thrust::copy(unexplained_by_RM_.begin(), unexplained_by_RM_.end(), first);
                }
            }
        }

        i++;
    }

    std::cout << "iterations:" << i << " " << current_best_cost << std::endl;

    solution_host = solution_;
    solution_global_.resize(solution_host.size());
    for(size_t h=0; h < solution_host.size(); h++)
    {
        solution_global_[h] = solution_host[h];
    }
}

//result of this kernel (visible points) is saved in an array of size (max_model_size * n_hypotheses)
void faat_pcl::recognition_cuda::GHV::computeVisibility()
{
    int max_model_size = 0;

    for(size_t i=0; i < n_models_; i++)
    {
        if(model_clouds_host_[i]->width_ > max_model_size)
            max_model_size = (int)model_clouds_host_[i]->width_;
    }

    pcl::gpu::DeviceArray<bool> visible;
    visible.create(max_model_size * hypotheses_.size());

    extended_points.create(max_model_size * hypotheses_.size());

    pcl::gpu::DeviceArray<float> angles_with_vp;
    angles_with_vp.create(max_model_size * hypotheses_.size());

    int threads = 512;
    int blocks = divUp(max_model_size * hypotheses_.size(), threads);

    checkVisibility vis_check;
    vis_check.visible = visible;
    vis_check.max_size_model = max_model_size;
    vis_check.n_hypotheses = hypotheses_.size();
    vis_check.n_models = n_models_;
    vis_check.model_clouds = model_clouds_;
    vis_check.model_normals = model_normals_;
    vis_check.scene = scene_.ptr();
    vis_check.hypotheses = hypotheses_dev_.ptr();
    vis_check.scene_width = 640;
    vis_check.scene_height = 480;
    vis_check.visible_points = extended_points;
    vis_check.angles_ = angles_with_vp;
    vis_check.colors = model_rgb_values_;
    vis_check.color_exist = color_exists_;

    visibilityCheckKernel<<<blocks, threads>>>(vis_check);
    cudaDeviceSynchronize();
    //std::cout << blocks << " " << threads << " " << max_model_size * hypotheses_.size() << " " << max_model_size << " " << hypotheses_.size() << std::endl;

    //create an indices vector pointing to visible points in visible_extended_points (valid_indices) so that we can process them further (trhust::copy_if)
        //inliers, outliers and explained points

    thrust::counting_iterator<int> first(0);
    thrust::counting_iterator<int> end( max_model_size * hypotheses_.size() );
    thrust::device_ptr<pointExtended> pe_dev_ptr = thrust::device_pointer_cast(extended_points.ptr());

    thrust::device_vector<int> valid_indices_;
    valid_indices_.resize(total_points_hypotheses_);

    typedef thrust::device_vector<int>::iterator IndexIterator;
    IndexIterator indices_end = thrust::copy_if(first, end, pe_dev_ptr, valid_indices_.begin(), visible_point());

    valid_indices_.resize(indices_end - valid_indices_.begin());

    //copy visibile indices
    visible_extended_points.create(valid_indices_.size());
    thrust::device_ptr<pointExtended> pe_visible_dev_ptr = thrust::device_pointer_cast(visible_extended_points.ptr());

    thrust::gather(valid_indices_.begin(),
                   valid_indices_.end(),
                   pe_dev_ptr,
                   pe_visible_dev_ptr);

    angles_with_vp_.create(valid_indices_.size());
    thrust::device_ptr<float> angles_with_vp_ptr = thrust::device_pointer_cast(angles_with_vp.ptr());
    thrust::device_ptr<float> angles_with_vp__ptr = thrust::device_pointer_cast(angles_with_vp_.ptr());
    thrust::gather(valid_indices_.begin(),
                   valid_indices_.end(),
                   angles_with_vp_ptr,
                   angles_with_vp__ptr);

    //visible_extended_points
    /*
     *
                   H1               H2      H3                  Hn
        -------------------------|------|-----------| ... |----------------|

        can we get the size of visible points per hypothesis? |H1| and so on
        i could create a vector of hypotheses_id from pointExtended
     *
     */

    //compute visible sizes for all hypotheses
    thrust::device_vector<int> hypotheses_ids = getHypothesesIdsFromPointExtended(valid_indices_, visible_extended_points);

    /*hypotheses_ids.resize(valid_indices_.size());
    {

        int threads = 512;
        int blocks = divUp((int)valid_indices_.size(), threads);

        int * raw_ptr = thrust::raw_pointer_cast(hypotheses_ids.data());
        extract_hyp_id<<<blocks, threads>>>(visible_extended_points.ptr(), raw_ptr);
        cudaDeviceSynchronize();
    }*/

    thrust::constant_iterator<int> values(1);
    visible_sizes_per_hypothesis_.resize(valid_indices_.size(), 0);
    thrust::device_vector<int> output_keys(valid_indices_.size());

    thrust::reduce_by_key(hypotheses_ids.begin(),
                          hypotheses_ids.end(),
                          values,
                          output_keys.begin(),
                          visible_sizes_per_hypothesis_.begin());

    visible_sizes_per_hypothesis_.resize(hypotheses_.size());

    //download visible and process
    bool * visible_host = new bool[max_model_size * hypotheses_.size()];
    visible.download(visible_host);

    visible_points_.resize(hypotheses_.size());
     for(size_t i=0; i < hypotheses_.size(); i++)
    {
        visible_points_[i].resize(hypotheses_[i].size_);
        int v = 0;
        for(int k=0; k < hypotheses_[i].size_; k++)
        {
            if(visible_host[i * max_model_size + k])
            {
                visible_points_[i][v] = k;
                v++;
            }
        }

        if(v == 0)
        {
            std::cout << "WARN: All points are occluded..." << i << std::endl;
        }
        visible_points_[i].resize(v);
    }

    delete[] visible_host;
    visible.release();
}

void faat_pcl::recognition_cuda::GHV::computeExplainedAndModelCues()
{

    explained_points_.clear();
    explained_points_weights_.clear();

    //build octree using the scene_cloud already on gpu
    pcl::gpu::DeviceArray<float4> scene_cloud_octree;

    {

        int size_cloud = scene_downsampled_.size();
        scene_cloud_octree.create(size_cloud);

        int threads = 512;
        int blocks = divUp(size_cloud, threads);

        xyzp_to_float4<<<blocks, threads>>>(scene_downsampled_.ptr(), scene_cloud_octree.ptr(), size_cloud);
    }

    octree_device_.setCloud(scene_cloud_octree);
    octree_device_.build();
    cudaDeviceSynchronize();

    pcl::gpu::DeviceArray<float4> visible_extended_points_float4;
    visible_extended_points_float4.create(visible_extended_points.size());

    {
        int threads = 512;
        int blocks = divUp(visible_extended_points.size(), threads);
        int size_cloud = visible_extended_points.size();

        pointExtended_to_float4<<<blocks, threads>>>(visible_extended_points.ptr(), visible_extended_points_float4.ptr(), size_cloud);
    }

    int max_results = 50;

    pcl::gpu::NeighborIndices results;
    results.create(static_cast<int> (visible_extended_points_float4.size()), max_results);
    results.sizes.create(visible_extended_points_float4.size());
    octree_device_.radiusSearch(visible_extended_points_float4, inlier_threshold, results);
    cudaDeviceSynchronize();

    //process the results in parallel on the GPU (results.sizes and results.data - vector of ints with neighbours)
    //in particular, results.data is a vector of max_results * #queries
    //for each hypothesis, we need a vector of explained points (unique indices) and the best weight

    thrust::device_ptr<int> rs_dev_ptr = thrust::device_pointer_cast(results.sizes.ptr());

    //for efficiency, we can process outliers (model outliers will simply be points where results.sizes[p_idx] = 0) and inliers separately
    //get ouliers indices
    //eventually, we need N(#hypotheses) vectors with the outlier indices (pointing to original models)
    //pointers to original models are in the pointExtended structure so its trivial

    //first process inliers to detect additional outliers because of color

    /*
     *
                   H1               H2      H3                  Hn
        -----X----XX-----X-----|------|---X---X---| ... |-------XX----XX----|
        X represent outliers
        gather outliers in a vector (similar to visible_extended_points) and reduce_by_key to count #outliers per hypothesis
     *
     */

    //outlier indices
    thrust::counting_iterator<int> first(0);
    thrust::counting_iterator<int> end( visible_extended_points.size() );

    thrust::device_vector<int> outlier_indices;
    outlier_indices.resize(visible_extended_points.size());

    typedef thrust::device_vector<int>::iterator IndexIterator;
    IndexIterator indices_end = thrust::copy_if(first, end, rs_dev_ptr, outlier_indices.begin(), is_zero());
    outlier_indices.resize(indices_end - outlier_indices.begin());

    //gather outliers
    outlier_extended_points_.create(outlier_indices.size());
    thrust::device_ptr<pointExtended> pe_visible_dev_ptr = thrust::device_pointer_cast(visible_extended_points.ptr());
    thrust::device_ptr<pointExtended> pe_outliers_dev_ptr = thrust::device_pointer_cast(outlier_extended_points_.ptr());
    thrust::gather(outlier_indices.begin(),
                   outlier_indices.end(),
                   pe_visible_dev_ptr,
                   pe_outliers_dev_ptr);

    //compute weights for outliers using normals
    //angles_with_vp_ has the angle information of visible points
    //use outlier_indices to compute weight and use it as input for the reduce_by_key instead of 1s
    thrust::device_vector<float> outliers_weight(outlier_indices.size(), outlier_regularizer_);
    float * ow_dev_ptr = thrust::raw_pointer_cast(outliers_weight.data());
    int * oi_dev_ptr = thrust::raw_pointer_cast(outlier_indices.data());

    computeOutlierWeights cow;
    cow.w_ = 0.1f;
    cow.problem_size_ = outlier_indices.size();
    cow.max_angle_ = 60.f;
    cow.angles_ = angles_with_vp_.ptr();
    cow.weights_ = ow_dev_ptr;
    cow.indices_ = oi_dev_ptr;

    {
        int threads = 512;
        int blocks = divUp(outlier_indices.size(), threads);

        computeOutlierWeightsKernel<<<blocks, threads>>>(cow);
    }

    //count outliers per hypotheses
    thrust::device_vector<int> hypotheses_ids = getHypothesesIdsFromPointExtended(outlier_indices, outlier_extended_points_);

    thrust::device_vector<float> outliers_per_hypothesis(outlier_indices.size(), 0);
    thrust::device_vector<int> output_keys(outlier_indices.size(), -1);
    typedef thrust::device_vector<float>::iterator IndexIterator_float;

    thrust::pair< IndexIterator , IndexIterator_float > new_end = thrust::reduce_by_key(  hypotheses_ids.begin(),
                                                                                            hypotheses_ids.end(),
                                                                                            outliers_weight.begin(),
                                                                                            output_keys.begin(),
                                                                                            outliers_per_hypothesis.begin());

    //careful, we need to iterate using output_keys in case there are hypotheses without outliers!!
    outliers_per_hypothesis.resize(new_end.first - output_keys.begin());
    output_keys.resize(new_end.first - output_keys.begin());
    outliers_per_hypothesis_.resize(hypotheses_.size(), 0);

    for(size_t i=0; i < output_keys.size(); i++)
        outliers_per_hypothesis_[output_keys[i]] = outliers_per_hypothesis[i];

    ///////////////////////////////////////////////////////////
    ///////////////scene explained points//////////////////////
    ///////////////////////////////////////////////////////////

    //1.) get inlier indices (the complementary of the outliers)
    thrust::device_vector<int> inlier_indices;
    inlier_indices.resize(visible_extended_points.size());

    {
        thrust::counting_iterator<int> first(0);
        thrust::counting_iterator<int> end( visible_extended_points.size() );
        typedef thrust::device_vector<int>::iterator IndexIterator;
        IndexIterator indices_end = thrust::copy_if(first, end, rs_dev_ptr, inlier_indices.begin(), is_not_zero());
        inlier_indices.resize(indices_end - inlier_indices.begin());
    }

    //std::cout << "inliers size:" << inlier_indices.size() << " " << outlier_indices.size() << " " << visible_extended_points.size() << std::endl;

    //2.) process them (compute values, using normals and inlier weight)
    //(results.sizes and results.data - vector of ints with neighbours)
    //in particular, results.data is a vector of max_results * #queries

    thrust::device_vector<float> inlier_weights(inlier_indices.size() * max_results);
    //thrust::device_vector<thrust::pair<int, int> > hyp_id_scene_p_keys(inlier_indices.size() * max_results);
    thrust::device_vector<int> scene_p_keys(inlier_indices.size() * max_results);
    thrust::device_vector<int> hyp_id_keys(inlier_indices.size() * max_results);

    int * ii_dev_ptr = thrust::raw_pointer_cast(inlier_indices.data());
    float * iw_dev_ptr = thrust::raw_pointer_cast(inlier_weights.data());
    //thrust::pair<int, int> * keys = thrust::raw_pointer_cast(hyp_id_scene_p_keys.data());
    int * scene_idx_ptr = thrust::raw_pointer_cast(scene_p_keys.data());
    int * hyp_idx_ptr = thrust::raw_pointer_cast(hyp_id_keys.data());

    {
        //CudaTiming time_eval("Inlier weight computation\n");

        //later, we can use this kernel to compute color weight and outliers because of color?
        int threads = 512;
        int blocks = divUp(inlier_weights.size(), threads);

        computeInlierWeights ciw(inlier_threshold);
        ciw.indices_ = ii_dev_ptr;
        ciw.weights_ = iw_dev_ptr;
        ciw.scene_ = scene_downsampled_.ptr();
        ciw.max_elems_ = max_results;
        ciw.visible_extended_points = visible_extended_points;
        ciw.nn_sizes_ = results.sizes;
        ciw.nn_ = results.data;
        //ciw.keys_ = keys; //necessary only if using chunks
        ciw.keys_scene_ = scene_idx_ptr;
        ciw.problem_size_ = inlier_weights.size();
        ciw.scene_colors_ = scene_rgb_values_.ptr();
        ciw.keys_hyp_id_ = hyp_idx_ptr;
        computeInlierWeightsKernel<<<blocks, threads>>>(ciw);
        cudaDeviceSynchronize();

    }

    scene_cloud_octree.release();
    visible_extended_points_float4.release();
    results.sizes.release();
    results.data.release();

    //3.) do a sort_by_key to group consecutive hypotheses_id, scene_point_id
    bool sort_by_chunks = false;

    PrintFreeMemory();
    std::cout << inlier_indices.size() << std::endl;

    if(!sort_by_chunks)
    {

        CudaTiming time_eval("Time for the whole stable_sort_by_key on GPU...");

        thrust::device_vector<int> permutation(inlier_indices.size() * max_results);
        thrust::sequence(permutation.begin(), permutation.end(), 0, 1);

        //sort permutation using scene_points
        thrust::device_vector<int> scene_p_keys_tmp(scene_p_keys.begin(), scene_p_keys.end());

        thrust::stable_sort_by_key(scene_p_keys_tmp.begin(),
                                   scene_p_keys_tmp.end(),
                                   permutation.begin());

        //gather hypotheses keys based on the scene point index permutation
        thrust::device_vector<int> hyp_id_keys_tmp(hyp_id_keys.size());
        thrust::gather(permutation.begin(), permutation.end(), hyp_id_keys.begin(), hyp_id_keys_tmp.begin());

        thrust::stable_sort_by_key(hyp_id_keys_tmp.begin(),
                                   hyp_id_keys_tmp.end(),
                                   permutation.begin());

        //gather hyp_id_keys and scene_p_keys in the correct order using final permutation
        {
            thrust::device_vector<int> scene_p_keys_tmp(scene_p_keys.begin(), scene_p_keys.end());
            thrust::device_vector<int> hyp_id_keys_tmp(hyp_id_keys.begin(), hyp_id_keys.end());
            thrust::device_vector<float> inlier_weights_tmp(inlier_weights.begin(), inlier_weights.end());

            thrust::gather(permutation.begin(), permutation.end(), scene_p_keys_tmp.begin(), scene_p_keys.begin());
            thrust::gather(permutation.begin(), permutation.end(), hyp_id_keys_tmp.begin(), hyp_id_keys.begin());
            thrust::gather(permutation.begin(), permutation.end(), inlier_weights_tmp.begin(), inlier_weights.begin());
        }

        //now keys (hyp_id, scene_id) are sorted, gather maximum
        typedef thrust::device_vector<float>::iterator IdxIter2;
        typedef thrust::device_vector< thrust::tuple<int, int> >::iterator IdxIter1;

        thrust::device_vector<float> inlier_weights_output(inlier_indices.size() * max_results);
        thrust::device_vector<thrust::tuple<int, int> > hyp_id_scene_p_keys_output(inlier_indices.size() * max_results);

        PrintFreeMemory();

        thrust::pair< IdxIter1 , IdxIter2 > new_end_hyp_points
            =  thrust::reduce_by_key(   thrust::make_zip_iterator(thrust::make_tuple(hyp_id_keys.begin(), scene_p_keys.begin())),
                                        thrust::make_zip_iterator(thrust::make_tuple(hyp_id_keys.end(), scene_p_keys.end())),
                                        inlier_weights.begin(),
                                        hyp_id_scene_p_keys_output.begin(),
                                        inlier_weights_output.begin(),
                                        binaryPredHypScenePointsTuple(),
                                        binaryOpHypScenePoints() );

        inlier_weights_output.resize(new_end_hyp_points.first - hyp_id_scene_p_keys_output.begin());
        hyp_id_scene_p_keys_output.resize(new_end_hyp_points.first - hyp_id_scene_p_keys_output.begin());

        //6.) for each hypotheses, extract the explained points (should be unique)
        {
            //CudaTiming t("saving explained points and stuff...\n");

            thrust::host_vector<thrust::tuple<int, int> > hyp_id_scene_p_keys_host;
            hyp_id_scene_p_keys_host = hyp_id_scene_p_keys_output;
            thrust::host_vector< float > inlier_weights_output_host = inlier_weights_output;

            explained_points_.resize(hypotheses_.size());
            explained_points_weights_.resize(hypotheses_.size());

            for(size_t i=0; i < hypotheses_.size(); i++)
            {
                explained_points_[i].reserve(20000);
                explained_points_weights_[i].reserve(20000);
            }

            for(size_t i=0; i < inlier_weights_output_host.size(); i++)
            {
                int scene_p_id = thrust::get<1>(hyp_id_scene_p_keys_host[i]);
                if(scene_p_id < 0)
                    continue;

                int hyp_id = thrust::get<0>(hyp_id_scene_p_keys_host[i]);
                float w = inlier_weights_output_host[i];

                explained_points_[hyp_id].push_back(scene_p_id);
                explained_points_weights_[hyp_id].push_back(w);
            }
        }
    }
    /*else
    {
        //sort chunks using only the scene_p_id as key
        CudaTiming time_eval("Time for the sort_by_key in chunks");

        thrust::device_vector<int> inliers_per_hypothesis(hypotheses_.size(), 0);

        {

            thrust::device_vector<int> inliers_per_hypothesis_local(inlier_indices.size(), 0);

            //count inliers per hypothesis
            thrust::device_vector<int> hypotheses_ids = getHypothesesIdsFromPointExtendedWithIndices(inlier_indices, visible_extended_points);

            thrust::constant_iterator<int> values(1);
            thrust::device_vector<int> output_keys(inlier_indices.size(), -1);

            thrust::pair< IndexIterator , IndexIterator > new_end = thrust::reduce_by_key(  hypotheses_ids.begin(),
                                                                                            hypotheses_ids.end(),
                                                                                            values,
                                                                                            output_keys.begin(),
                                                                                            inliers_per_hypothesis_local.begin());

            inliers_per_hypothesis_local.resize(new_end.first - output_keys.begin());
            output_keys.resize(new_end.first - output_keys.begin());

            std::cout << output_keys.size() << " " << hypotheses_.size() << std::endl;

            thrust::host_vector<int> inliers_per_hypothesis_local_host = inliers_per_hypothesis_local;

            for(size_t i=0; i < output_keys.size(); i++)
                inliers_per_hypothesis[output_keys[i]] = inliers_per_hypothesis_local_host[i];
        }

        thrust::host_vector<int> inliers_per_hypothesis_host(hypotheses_.size());
        inliers_per_hypothesis_host = inliers_per_hypothesis;


        thrust::device_vector<int> sorted_indices(inlier_indices.size() * max_results);
        thrust::sequence(sorted_indices.begin(), sorted_indices.end(), 0, 1);

        int start_pos = 0;
        for(size_t i=0; i < hypotheses_.size(); i++)
        {
            if(inliers_per_hypothesis_host[i] == 0)
            {
                std::cout << "WARN: this guy has no explained points, what is the effect? " << i << std::endl;
                continue;
            }

            int size = inliers_per_hypothesis_host[i] * max_results;

            thrust::sort_by_key(scene_p_keys.begin() + start_pos,
                                scene_p_keys.begin() + start_pos + size,
                                sorted_indices.begin() + start_pos);

            start_pos += size;
        }

        {
            CudaTiming t_eval("gathering and copy");

            thrust::device_vector<float> inlier_weights_tmp(inlier_indices.size() * max_results);
            thrust::device_vector<thrust::pair<int, int> > hyp_id_scene_p_keys_tmp(inlier_indices.size() * max_results);

            thrust::gather(sorted_indices.begin(), sorted_indices.end(), hyp_id_scene_p_keys.begin(), hyp_id_scene_p_keys_tmp.begin());
            thrust::gather(sorted_indices.begin(), sorted_indices.end(), inlier_weights.begin(), inlier_weights_tmp.begin());

            hyp_id_scene_p_keys = hyp_id_scene_p_keys_tmp;
            inlier_weights = inlier_weights_tmp;
        }

        //4.) reduce_by_key to find the maximum value ( key will be the pair(scene_point_id, hypotheses_id) )
            //the best hypotheses point explaining a scene point
            //i have weights, keys and

        typedef thrust::device_vector<float>::iterator IdxIter2;
        typedef thrust::device_vector< thrust::pair<int, int> >::iterator IdxIter1;

        thrust::device_vector<float> inlier_weights_output(inlier_indices.size() * max_results);
        thrust::device_vector<thrust::pair<int, int> > hyp_id_scene_p_keys_output(inlier_indices.size() * max_results);

        thrust::pair< IdxIter1 , IdxIter2 > new_end_hyp_points
                                    = thrust::reduce_by_key(    hyp_id_scene_p_keys.begin(),
                                                                hyp_id_scene_p_keys.end(),
                                                                inlier_weights.begin(),
                                                                hyp_id_scene_p_keys_output.begin(),
                                                                inlier_weights_output.begin(),
                                                                binaryPredHypScenePoints(),
                                                                binaryOpHypScenePoints() );


        inlier_weights_output.resize(new_end_hyp_points.first - hyp_id_scene_p_keys_output.begin());
        hyp_id_scene_p_keys_output.resize(new_end_hyp_points.first - hyp_id_scene_p_keys_output.begin());

        //6.) for each hypotheses, extract the explained points (should be unique)
        thrust::host_vector<thrust::pair<int, int> > hyp_id_scene_p_keys_host;
        hyp_id_scene_p_keys_host = hyp_id_scene_p_keys_output;
        thrust::host_vector< float > inlier_weights_output_host = inlier_weights_output;

        explained_points_.resize(hypotheses_.size());
        explained_points_weights_.resize(hypotheses_.size());

        for(size_t i=0; i < hypotheses_.size(); i+=1)
        {
            explained_points_[i].reserve(20000);
            explained_points_weights_[i].reserve(20000);
        }

        for(size_t i=0; i < inlier_weights_output_host.size(); i++)
        {
            int hyp_id = hyp_id_scene_p_keys_host[i].first;
            int scene_p_id =  hyp_id_scene_p_keys_host[i].second;
            float w = inlier_weights_output_host[i];

            if(scene_p_id < 0)
                continue;

            //std::cout << "weight is:" << w << std::endl;
            explained_points_[hyp_id].push_back(scene_p_id);
            explained_points_weights_[hyp_id].push_back(w);
        }
    }*/
}

//for all explained points in the scene (by any hypothesis), compute NN in clutter_radius
//this remain constant for all hypotheses!!
//smooth segmentation (provided from outside)
//and create the unexplained vector

void faat_pcl::recognition_cuda::GHV::computeClutterCue()
{

    /*if(!detect_clutter_)
        return;*/

    //i will need #hypotheses vectors to know which points are explained by an hypothesis to avoid counting them for the clutter term
    //create a huge vector of size scene_downsampled_.size() * hypotheses_.size()
    thrust::host_vector<bool> scene_point_explained_by_hypothesis_host(scene_downsampled_.size() * hypotheses_.size(), false);
    thrust::host_vector<explainedSPExtended> explainedSP(hypotheses_.size() * scene_downsampled_.size());

    //compute all points explained in the scene and vector
    std::set<int> explained_points;
    int num_exp = 0;
    for(size_t i=0; i < hypotheses_.size(); i+=1)
    {
        int idx_hyp = scene_downsampled_.size() * i;
        for(size_t k=0; k < explained_points_[i].size(); k++)
        {
            explained_points.insert(explained_points_[i][k]);
            scene_point_explained_by_hypothesis_host[idx_hyp + explained_points_[i][k]] = true;
            explainedSP[num_exp].hypotheses_id = i;
            explainedSP[num_exp].scene_idx = explained_points_[i][k];
            num_exp++;
        }
    }

    explainedSP.resize(num_exp);

    std::vector<int> explained_points_vec;
    explained_points_vec.assign(explained_points.begin(), explained_points.end());

    thrust::host_vector<int> scene_to_unique(scene_downsampled_.size(),-1);
    for(size_t i=0; i < explained_points_vec.size(); i++)
        scene_to_unique[explained_points_vec[i]] = i;

    //transform cloud to octree format using explained indices
    pcl::gpu::DeviceArray<float4> scene_cloud_octree;
    {
        thrust::device_vector<int> explained_points_indices = explained_points_vec;
        int size_cloud = explained_points_vec.size();
        scene_cloud_octree.create(size_cloud);
        int threads = 512;
        int blocks = divUp(size_cloud, threads);
        int * raw_ptr = thrust::raw_pointer_cast(explained_points_indices.data());
        xyzp_to_float4_with_indices<<<blocks, threads>>>(scene_downsampled_.ptr(), scene_cloud_octree.ptr(), raw_ptr, size_cloud);
        cudaDeviceSynchronize();
    }

    //get NN in radius search
    int max_results = 1000;
    float radius_clutter = 0.04f;

    pcl::gpu::NeighborIndices results;

    {
        CudaTiming t_radius_search_clutter("Radius search clutter GPU\n");
        results.create(static_cast<int> (explained_points_vec.size()), max_results);
        results.sizes.create(explained_points_vec.size());
        octree_device_.radiusSearch(scene_cloud_octree, radius_clutter, results);
        cudaDeviceSynchronize();
    }
    //PrintFreeMemory();

    //std::cout << "elements in results:" << results.data.size() << std::endl;
    //then we will process for all hypotheses their respective explained points (in total num_exp)
    //and the number of kernels would be num_exp * max_results

    //std::cout << "to proces::" << num_exp * max_results << std::endl; //i need the hypothesis id and the scene point id for each point

    //then, will have a kernel that processes this huge vector computing for each explained scene point by an hypothesis the clutter weight
    //a large amount of this points will be explained by the specific hypotheses, so might be worth looking
    //into first removing those results in the Neighborhood that are actually explained by the hypothesis and then compute the clutter cue
    //this would result in computing std::vector<int> unexplained_in_neighborhood; for each hypothesis

    //compute for each explained point by an hypothesis, the distance to unexplained_points (num_exp * max_results)
    /*

    thrust::device_vector<bool> scene_point_explained_by_hypothesis = scene_point_explained_by_hypothesis_host;
    thrust::device_vector<int> scene_to_unique_dev = scene_to_unique;

    thrust::device_vector<explainedSPExtended> explainedSP_dev = explainedSP;
    std::cout << "trying to allocate..." << explainedSP_dev.size() << " " << num_exp << std::endl;
    thrust::device_vector<float> distance_explained_to_unexplained( num_exp * max_results, 10.f);
    PrintFreeMemory();

    {

        CudaTiming t("GPU clutter");
        int size_cloud = num_exp * max_results;
        int threads = 256;
        int blocks = divUp(size_cloud, threads);

        explainedSPExtended * raw_ptr_expSP_dev = thrust::raw_pointer_cast(explainedSP_dev.data());
        bool * raw_ptr_scene_point_explained_by_hypothesis = thrust::raw_pointer_cast(scene_point_explained_by_hypothesis.data());
        float * raw_ptr_distance_explained_to_unexplained = thrust::raw_pointer_cast(distance_explained_to_unexplained.data());
        int * raw_ptr_scene_to_unique = thrust::raw_pointer_cast(scene_to_unique_dev.data());

        computeDistFromExplainedToUnexplainedInNeighborhood cdfeun;
        cdfeun.scene_ = scene_downsampled_.ptr();
        cdfeun.max_elems_ = max_results;
        cdfeun.explained_extended_ = raw_ptr_expSP_dev;
        cdfeun.nn_sizes_ = results.sizes;
        cdfeun.nn_ = results.data;
        cdfeun.scene_point_explained_by_hypothesis_ = raw_ptr_scene_point_explained_by_hypothesis;
        cdfeun.scene_size_ = scene_downsampled_.size();
        cdfeun.distance_explained_to_unexplained_ = raw_ptr_distance_explained_to_unexplained;
        cdfeun.scene_to_unique_exp_ = raw_ptr_scene_to_unique;
        cdfeun.total_size_ = size_cloud;
        computeDistFromExplainedToUnexplainedInNeighborhoodKernel<<<blocks, threads>>>(cdfeun);
        cudaDeviceSynchronize();
        std::cout << "Launched kernel apparently" << threads << " " << blocks << std::endl;
    }

    thrust::host_vector<float> distance_explained_to_unexplained_host = distance_explained_to_unexplained;
    int c = 0;
    int c1 = 0;
    int c2 = 0;
    for(size_t i=0; i < distance_explained_to_unexplained_host.size(); i++)
    {
        if(distance_explained_to_unexplained_host[i] < 0)
        {
            c++;
            continue;
        }

        if(distance_explained_to_unexplained_host[i] > 1.f && distance_explained_to_unexplained_host[i] < 10.f)
        {
            c1++;
            continue;
        }

        if(distance_explained_to_unexplained_host[i] < 1.f)
        {
            c2++;
            continue;
        }
    }

    std::cout << "c:" << c << " explained:" << c1 << " unexplained:" << c2 << " " << distance_explained_to_unexplained_host.size() << std::endl;*/

    //PrintFreeMemory();

    //find maximum

    {
        CudaTiming t("clutter part CPU");
        //download NNs
        std::vector<int> sizes;
        std::vector<int> rs_indices;
        xyz_p * scene_downsampled_host = new xyz_p[scene_downsampled_.size()];

        {
            CudaTiming t("downloanding data from the GPU");
            results.sizes.download(sizes);
            results.data.download(rs_indices);

            scene_downsampled_.download(scene_downsampled_host);
        }

        std::vector< std::vector< std::pair<int, float> > > unexplained_points_per_model;
        unexplained_points_per_model.resize(hypotheses_.size());
        std::pair<int, float> def_value = std::make_pair(-1, std::numeric_limits<float>::infinity());
        for(size_t i=0; i < hypotheses_.size(); i++)
            unexplained_points_per_model[i].resize(scene_downsampled_.size(), def_value);

        //CPU
        int explained_in_nn = 0;
        int unexplained_in_nn = 0;
        float inlier_thres_sqr = inlier_threshold * inlier_threshold;
        for(size_t i=0; i < hypotheses_.size(); i+=1)
        {
            for(size_t k=0; k < explained_points_[i].size(); k++)
            {
                int s_id_exp = explained_points_[i][k];
                int idx_to_unique = scene_to_unique[explained_points_[i][k]];
                int hyp_id = scene_downsampled_.size() * i;
                int beg = idx_to_unique * max_results;

                for(size_t kk=0; kk < sizes[idx_to_unique]; kk++) //sizes[i] is at most max_results
                {
                    int sidx = rs_indices[beg+kk];
                    bool exp = scene_point_explained_by_hypothesis_host[hyp_id + sidx];
                    if(!exp)
                        unexplained_in_nn++;
                    else
                    {
                        explained_in_nn++;
                        continue;
                    }

                    float sqrt_dist = sqrDist(scene_downsampled_host[s_id_exp], scene_downsampled_host[sidx]);
                    if( (sqrt_dist > (inlier_thres_sqr)) && (sqrt_dist < unexplained_points_per_model[i][sidx].second))
                    {
                        //there is an explained point which is closer to this unexplained point
                        unexplained_points_per_model[i][sidx].second = sqrt_dist;
                        unexplained_points_per_model[i][sidx].first = s_id_exp;
                    }
                }
            }
        }

        //std::cout << "explained:" << explained_in_nn << " unexplained:" << unexplained_in_nn << std::endl;

        unexplained_points_.resize(hypotheses_.size());
        unexplained_points_weights_.resize(hypotheses_.size());

        float rn_sqr = radius_clutter * radius_clutter;

        for(size_t kk=0; kk < hypotheses_.size(); kk++)
        {
            int p=0;
            unexplained_points_[kk].resize(unexplained_points_per_model[kk].size());
            unexplained_points_weights_[kk].resize(unexplained_points_per_model[kk].size());

            for(size_t i=0; i < unexplained_points_per_model[kk].size(); i++)
            {
                int sidx = unexplained_points_per_model[kk][i].first;
                if(sidx < 0)
                    continue;

                float d = unexplained_points_per_model[kk][i].second;
                float d_weight = -(d / rn_sqr) + 1; //points that are close have a strong weight
                float dot_p = dot<xyz_p>(scene_normals_host_->points[i], scene_normals_host_->points[sidx]);
                if(dot_p < 0.1)
                    dot_p = 0.1;

                d_weight *= dot_p;

                if(labels_[i] != 0 && (labels_[i] == labels_[sidx]))
                {
                    d_weight *= clutter_regularizer_;
                }

                unexplained_points_[kk][p] = i;
                unexplained_points_weights_[kk][p] = d_weight;
                p++;
            }

            unexplained_points_[kk].resize(p);
            unexplained_points_weights_[kk].resize(p);
        }
    }
}
