#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <stdio.h>
//#include <pcl/point_types.h>
#include <pcl/gpu/containers/device_array.h>
#include "internal.hpp"

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

namespace faat_pcl
{

namespace recognition_cuda
{

struct xyz_p
{
    float x;
    float y;
    float z;
};

struct pointExtended
{
    xyz_p p;
    int hypotheses_id;
    int model_id;
    int point_id;
    bool set;
    float3 color; //color space independent
    bool color_set_;
    int hypotheses_type_; //0 for objects, 1 for planes
};

struct explainedSPExtended
{
    int hypotheses_id;
    int scene_idx;
};

struct rgb
{
    float r;
    float g;
    float b;
};

struct Mat4f {
    float mat[4][4];
};

struct HypothesisGPU
{
  Mat4f transform_;
  int model_idx_;
  int size_;
  /*bool * p_visible_;
  int * visible_;*/
};

class XYZPointCloud
{
    public:

        __host__ __device__
        XYZPointCloud()
        {

        }

        __host__
        ~XYZPointCloud()
        {
            if(on_device_)
            {
                if(points)
                {
                    cudaFree(points);
                    cudaCheckErrors("cudaFree fail");
                }
            }
            else
            {
                if(points)
                    delete[] points;
            }
        }

        int width_;
        int height_;
        bool on_device_;
        xyz_p * points;

        __device__ __host__
        xyz_p at(int u, int v)
        {
            return points[u * width_ + v];
        }
};

//scene should be allocated in GPU memory
//points of the model should also be allocated in GPU

class GHV
{
    pcl::gpu::DeviceArray2D<xyz_p> scene_;
    pcl::gpu::DeviceArray<xyz_p> scene_downsampled_;
    pcl::gpu::DeviceArray<float3> scene_rgb_values_;
    XYZPointCloud * scene_downsampled_host_;
    XYZPointCloud * scene_normals_host_;

    pcl::gpu::DeviceArray<xyz_p> * model_clouds_device_array_;
    pcl::gpu::DeviceArray<xyz_p> * model_normals_device_array_;
    pcl::gpu::DeviceArray<float3> * model_colours_device_array_;

    xyz_p ** model_clouds_;
    xyz_p ** model_normals_;
    float3 ** model_rgb_values_;

    std::vector<HypothesisGPU> hypotheses_;
    pcl::gpu::DeviceArray<HypothesisGPU> hypotheses_dev_;
    size_t n_models_;
    std::vector<XYZPointCloud *> model_clouds_host_;
    int total_points_hypotheses_;

    std::vector<int> labels_; //scene_downsampled_.size()
    pcl::device::OctreeImpl octree_device_;

    //parameters
    float inlier_threshold;
    float clutter_regularizer_;
    float outlier_regularizer_;
    bool detect_clutter_;
    float clutter_radius_;
    float color_sigma_y_, color_sigma_ab_;

    //output variables
    pcl::gpu::DeviceArray<float> angles_with_vp_;
    std::vector< std::vector<int> > visible_points_; //point indices that are visible for each hypothesis
    pcl::gpu::DeviceArray<pointExtended> extended_points;
    pcl::gpu::DeviceArray<pointExtended> visible_extended_points;
    thrust::device_vector<int> visible_sizes_per_hypothesis_; //hypotheses_.size() vector
    thrust::device_vector<float> outliers_per_hypothesis_; //hypotheses_.size() vector
    pcl::gpu::DeviceArray<pointExtended> outlier_extended_points_;

    thrust::device_vector<int> getHypothesesIdsFromPointExtended(thrust::device_vector<int> & indices,
                                                                 pcl::gpu::DeviceArray<pointExtended> & points);

    thrust::device_vector<int> getHypothesesIdsFromPointExtendedWithIndices(thrust::device_vector<int> & indices,
                                                                 pcl::gpu::DeviceArray<pointExtended> & points);


    std::vector< std::vector<int> > explained_points_;
    std::vector< std::vector<float> > explained_points_weights_;
    std::vector< std::vector<int> > unexplained_points_;
    std::vector< std::vector<float> > unexplained_points_weights_;

    std::vector<int> solution_global_;
    bool color_exists_;
public:
    GHV()
    {
        outlier_regularizer_ = 2.f;
        inlier_threshold = 0.008f;
        clutter_regularizer_ = 5.f;
        detect_clutter_ = true;
        color_exists_ = false;
        clutter_radius_ = 0.03f;
        color_sigma_y_ = color_sigma_ab_ = 0.5f;
    }

    void setclutterRadius(float f)
    {
        clutter_radius_ = f;
    }

    void setInlierThreshold(float i)
    {
        inlier_threshold = i;
    }

    void setOutlierWewight(float i)
    {
        outlier_regularizer_ = i;
    }

    void setClutterWeight(float i)
    {
        clutter_regularizer_ = i;
    }

    void setColorSigmas(float cs_y, float cs_ab)
    {
        color_sigma_y_ = cs_y;
        color_sigma_ab_ = cs_ab;
    }

    ~GHV()
    {

    }

    void setDetectClutter(bool b)
    {
        detect_clutter_ = b;
    }

    void setSmoothLabels(std::vector<int> & labels)
    {
        labels_ = labels;
    }

    void freeMemory();

    //scene cloud used for visibility checking (organized)
    void setSceneCloud(XYZPointCloud * cloud);

    void setSceneRGBValues(float3 * rgb_scene_values, int size);

    //unorganized cloud used to compute cues and optimize
    void setScenePointCloud(XYZPointCloud * cloud);

    void setModelClouds(std::vector<XYZPointCloud *> & model_clouds,
                        std::vector<XYZPointCloud *> & model_normals);

    void setModelColors(std::vector<float3 *> & model_colors, int models_size, std::vector<int> & sizes_per_model);

    void setSceneNormals(XYZPointCloud * normals);

    void setHypotheses(std::vector<HypothesisGPU> & hypotheses);

    void computeExplainedAndModelCues();

    void computeClutterCue();

    void computeVisibility();

    void getVisible(std::vector< std::vector<int> > & visible_points)
    {
        visible_points = visible_points_;
    }

    void getVisibleAndOutlierSizes(std::vector< std::pair<int, int> > & visible_and_outlier_sizes)
    {
        for(size_t i=0; i < hypotheses_.size(); i++)
        {
            std::pair<int, int> p = std::make_pair(visible_sizes_per_hypothesis_[i], outliers_per_hypothesis_[i]);
            visible_and_outlier_sizes.push_back(p);
        }
    }

    void getExplainedPointsAndWeights(std::vector< std::vector<int> > & explained,
                                      std::vector< std::vector<float> > & explained_weights)
    {
        explained = explained_points_;
        explained_weights = explained_points_weights_;
    }

    void getUnexplainedPointsAndWeights(std::vector< std::vector<int> > & unexplained,
                                        std::vector< std::vector<float> > & unexplained_weights)
    {
        unexplained = unexplained_points_;
        unexplained_weights = unexplained_points_weights_;
    }

    void optimize();

    void getSolution(std::vector<bool> & solution)
    {
        solution.resize(solution_global_.size(), false);
        for(size_t i=0; i < solution_global_.size(); i++)
        {
            if(solution_global_[i] == 1)
                solution[i] = true;
        }
    }

};
}
}
