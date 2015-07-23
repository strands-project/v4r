#ifndef DO_LEARNING_H_
#define DO_LEARNING_H_

/*
 * Author: Thomas Faeulhammer
 * Date: June 2015
 *
 * */

#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/octree.h>

#include <3rdparty//SiftGPU/src/SiftGPU/SiftGPU.h>
#include <v4r/common/keypoint/ClusterNormalsToPlanes.hh>
#include <v4r/common/keypoint/impl/PointTypes.hpp>
#include "v4r/object_modelling/model_view.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/filesystem.hpp>
#include <boost/graph/graph_traits.hpp>

namespace v4r
{
namespace object_modelling
{

class CamConnect
{
public:
    CamConnect()
    {

    }
    Eigen::Matrix4f transformation_;
    double edge_weight_;
    std::string model_name_;
    std::string source_id_, target_id_;
};

class DOL
{
public:
    class Parameter{
    public:
        double radius_;
        double eps_angle_;
        double voxel_resolution_;
        double seed_resolution_;
        double ratio_;
        double chop_z_;
        bool do_erosion_;
        bool do_sift_based_camera_pose_estimation_;
        bool transfer_indices_from_latest_frame_only_;
        size_t min_points_for_transferring_;
        int normal_method_;
        Parameter (double radius = 0.005f, double eps_angle = 0.9f, double voxel_resolution = 0.005f,
                   double seed_resolution = 0.03f, double ratio = 0.25f,
                   double chop_z = std::numeric_limits<double>::quiet_NaN(), bool do_erosion = true,
                   bool do_sift_based_camera_pose_estimation = false, bool transfer_indices_from_latest_frame_only = false,
                   size_t min_points_for_transferring = 10, int normal_method = 1) :
            radius_(radius), eps_angle_(eps_angle), voxel_resolution_(voxel_resolution),
            seed_resolution_(seed_resolution), ratio_(ratio), chop_z_(chop_z), do_erosion_(do_erosion),
            do_sift_based_camera_pose_estimation_(do_sift_based_camera_pose_estimation),
            transfer_indices_from_latest_frame_only_(transfer_indices_from_latest_frame_only),
            min_points_for_transferring_(min_points_for_transferring), normal_method_(normal_method)
        {
        }

    };

protected:
    typedef pcl::PointXYZRGB PointT;
    typedef flann::L1<float> DistT;
    typedef pcl::Histogram<128> FeatureT;

    typedef boost::adjacency_list < boost::vecS, boost::vecS, boost::undirectedS,
             modelView, CamConnect> Graph;
    typedef boost::graph_traits < Graph >::vertex_descriptor Vertex;
    typedef boost::graph_traits < Graph >::edge_descriptor Edge;
    typedef boost::graph_traits<Graph>::vertex_iterator vertex_iter;
    typedef boost::graph_traits<Graph>::edge_iterator edge_iter;

//    typedef boost::property_map<Graph, boost::vertex_index_t>::type IndexMap;

    Parameter param_;
    kp::ClusterNormalsToPlanes::Parameter p_param_;

    pcl::PointCloud<PointT>::Ptr big_cloud_;
    pcl::PointCloud<PointT>::Ptr big_cloud_segmented_;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> vis_, vis_reconstructed_;
    std::vector<int> vis_reconstructed_viewpoint_;
    std::vector<int> vis_viewpoint_;

    std::vector<size_t> LUT_new2old_indices;
    cv::Ptr<SiftGPU> sift_;
    Graph grph_;
    size_t counter_;
    pcl::octree::OctreePointCloudSearch<PointT> octree_;

    ///radius to select points in other frames to belong to the same object
    /// bootstraps region growing

public:

    DOL () : octree_(0.005f)
    {
        // Parameters for smooth clustering / plane segmentation
        p_param_.thrAngle=45;
        p_param_.inlDist=0.05;
        p_param_.minPoints=5000;    // minimum number for a plane to be segmented
        p_param_.least_squares_refinement=true;
        p_param_.smooth_clustering=true;
        p_param_.thrAngleSmooth=30;
        p_param_.inlDistSmooth=0.02;
        p_param_.minPointsSmooth=20;    // minimum number for a segment other than a plane

        big_cloud_.reset(new pcl::PointCloud<PointT>);
        big_cloud_segmented_.reset(new pcl::PointCloud<PointT>);

        counter_ = 0;
    }

    void extractEuclideanClustersSmooth (const pcl::PointCloud<PointT>::ConstPtr &cloud,
                const pcl::PointCloud<pcl::Normal> &normals_,
                const std::vector<size_t> &initial,
                std::vector<size_t> &cluster) const;

    void updatePointNormalsFromSuperVoxels(const pcl::PointCloud<PointT>::Ptr & cloud,
                                           pcl::PointCloud<pcl::Normal>::Ptr & normals_,
                                           const std::vector< size_t > &object_points,
                                           std::vector< size_t > & good_neighbours,
                                           pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &supervoxel_cloud);


    void erodeInitialIndices(const pcl::PointCloud<PointT> & cloud,
                             const std::vector< size_t > & initial_indices,
                             std::vector< size_t > & eroded_indices);

    static void createDirIfNotExist(const std::string & dirs)
    {
        boost::filesystem::path dir = dirs;
        if(!boost::filesystem::exists(dir))
        {
            boost::filesystem::create_directory(dir);
        }
    }

    bool save_model (const std::string &models_dir = "/tmp/dynamic_models/",
                     const std::string &recognition_structure_dir = "/tmp/recognition_structure_dir/",
                     const std::string &model_name = "new_dynamic_model.pcd");

    bool learn_object (const pcl::PointCloud<PointT> &cloud,
                       const Eigen::Matrix4f &camera_pose = Eigen::Matrix4f::Identity(),
                       const std::vector<size_t> &initial_indices = std::vector<size_t>());

    void initialize (int argc, char ** argv);

    void clearMem()
    {
        big_cloud_->points.clear();
        big_cloud_segmented_->points.clear();
        LUT_new2old_indices.clear();
    }

    /**
     * @brief given a point cloud and a normal cloud, this function computes points belonging to a table
     *  (optional: computes smooth clusters for points not belonging to table)
     * @param cloud
     * @param normals
     * @param p_param
     * @param planes
     */
    static void extractPlanePoints(const pcl::PointCloud<PointT>::ConstPtr &cloud,
                            const pcl::PointCloud<pcl::Normal>::ConstPtr &normals,
                            const kp::ClusterNormalsToPlanes::Parameter p_param,
                            std::vector<kp::ClusterNormalsToPlanes::Plane::Ptr> &planes);

    static void getPlanesNotSupportedByObjectMask(const std::vector<kp::ClusterNormalsToPlanes::Plane::Ptr> &planes,
                                                const std::vector< size_t > object_mask,
                                                std::vector<kp::ClusterNormalsToPlanes::Plane::Ptr> &planes_dst,
                                                std::vector< size_t > &all_plane_indices_wo_object,
                                                float ratio=0.25);
    void visualize();

    /**
     * @brief transforms each keyframe to global coordinate system using given camera
     *  pose and does segmentation based on the computed object indices to create
     *  reconstructed point cloud of scene and object model
     */
    void createBigCloud();


    /**
     * @brief given indices of an image or pointcloud, this function create a boolean mask of the indices
     * @param objectIndices
     * @param image_size
     * @param object_mask (output)
     */
    void createMaskFromIndices(const std::vector<size_t> &objectIndices,
                                size_t image_size,
                                std::vector<bool> &object_mask);

    /**
     * @brief this function returns the indices of foreground given the foreground mask and considering all true pixels in background mask are being removed
     * @param background_mask (if true, pixel is neglected / not being part of the scene)
     * @param foreground_mask
     * @param new_foreground_indices (output)
     * @param indices of original image which belong to the scene (output)
     */
    void updateIndicesConsideringMask(const std::vector<bool> &background_mask,
                                           const std::vector<bool> &foreground_mask,
                                           std::vector<size_t> &new_foreground_indices,
                                           std::vector<size_t> &old_bg_indices);

    void computeNormals(const pcl::PointCloud<PointT>::ConstPtr &cloud,
                        pcl::PointCloud<pcl::Normal>::Ptr &normals, int method);

    bool calcSiftFeatures (const pcl::PointCloud<PointT>::Ptr &cloud_src,
                      pcl::PointCloud<PointT>::Ptr &sift_keypoints,
                      std::vector< size_t > &sift_keypoint_indices,
                      pcl::PointCloud<FeatureT>::Ptr &sift_signatures,
                      std::vector<float> &sift_keypoint_scales);

    void
    estimateViewTransformationBySIFT(const pcl::PointCloud<PointT> &src_cloud,
                                          const pcl::PointCloud<PointT> &dst_cloud,
                                          const std::vector<size_t> &src_sift_keypoint_indices,
                                          const std::vector<size_t> &dst_sift_keypoint_indices,
                                          const pcl::PointCloud<FeatureT> &src_sift_signatures,
                                          boost::shared_ptr< flann::Index<DistT> > &src_flann_index,
                                          std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transformations,
                                          bool use_gc = false);

    float calcEdgeWeightAndRefineTf (const pcl::PointCloud<PointT>::ConstPtr &cloud_src,
                          const pcl::PointCloud<PointT>::ConstPtr &cloud_dst,
                          Eigen::Matrix4f &transformation);

    void printParams(std::ostream &ostr = std::cout) const;

    /**
     * @brief transfers object points nearest neighbor search in dest frame.
     * @param points which are transferred and looked for in search cloud
     * @param search space for transferred points
     * @param nn... nearest neighbors points highlighted (true) in object mask
     * @param homogeneous transformation matrix for object_points into search cloud
     */
    void nnSearch(const pcl::PointCloud<PointT> &object_points, pcl::octree::OctreePointCloudSearch<PointT> &octree,  std::vector<bool> &obj_mask);
    void nnSearch(const pcl::PointCloud<PointT> &object_points, const pcl::PointCloud<PointT>::ConstPtr &search_cloud,  std::vector<bool> &obj_mask);
};

}
}

#endif //DO_LEARNING_H_
