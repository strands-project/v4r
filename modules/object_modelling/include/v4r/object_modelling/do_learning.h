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

#include <3rdparty/SiftGPU/src/SiftGPU/SiftGPU.h>
#include <v4r/common/keypoint/ClusterNormalsToPlanes.hh>
#include <v4r/common/keypoint/impl/PointTypes.hpp>
#include <v4r/common/noise_model_based_cloud_integration.h>
#include <v4r/object_modelling/model_view.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/filesystem.hpp>
#include <boost/graph/graph_traits.hpp>

namespace v4r
{
namespace object_modelling
{

struct CamConnect
{
    Eigen::Matrix4f transformation_;
    float edge_weight;
    std::string model_name_;
    size_t source_id_, target_id_;

    explicit CamConnect(float w) :
        edge_weight(w)
    {

    }

    CamConnect() : edge_weight(0.f)
    {

    }

    bool operator<(const CamConnect& e) const {
        if(edge_weight < e.edge_weight)
            return true;

        return false;
    }

    bool operator<=(const CamConnect& e) const {
        if(edge_weight <= e.edge_weight)
            return true;

        return false;
    }

    bool operator>(const CamConnect& e) const {
        if(edge_weight > e.edge_weight)
            return true;

        return false;
    }

    bool operator>=(const CamConnect& e) const {
        if(edge_weight >= e.edge_weight)
            return true;

        return false;
    }
};

class DOL
{
public:
    class Parameter{
    public:
        double radius_;
        double eps_angle_;
        double dist_threshold_growing_;
        double voxel_resolution_;
        double seed_resolution_;
        double ratio_;
        double chop_z_;
        bool do_erosion_;
        bool do_sift_based_camera_pose_estimation_;
        bool transfer_indices_from_latest_frame_only_;
        size_t min_points_for_transferring_;
        int normal_method_;
        bool do_mst_refinement_;
        Parameter (double radius = 0.005f,
                   double eps_angle = 0.9f,
                   double dist_threshold_growing = 0.05f,
                   double voxel_resolution = 0.005f,
                   double seed_resolution = 0.03f,
                   double ratio = 0.25f,
                   double chop_z = std::numeric_limits<double>::quiet_NaN(),
                   bool do_erosion = true,
                   bool do_sift_based_camera_pose_estimation = false,
                   bool transfer_indices_from_latest_frame_only = false,
                   size_t min_points_for_transferring = 10,
                   int normal_method = 1,
                   bool do_mst_refinement = true) :
            radius_(radius),
            eps_angle_(eps_angle),
            dist_threshold_growing_(dist_threshold_growing),
            voxel_resolution_(voxel_resolution),
            seed_resolution_(seed_resolution),
            ratio_(ratio),
            chop_z_(chop_z),
            do_erosion_(do_erosion),
            do_sift_based_camera_pose_estimation_(do_sift_based_camera_pose_estimation),
            transfer_indices_from_latest_frame_only_(transfer_indices_from_latest_frame_only),
            min_points_for_transferring_(min_points_for_transferring),
            normal_method_(normal_method),
            do_mst_refinement_(do_mst_refinement)
        {
        }

    };

    enum MASK_OPERATOR{
        AND,
        AND_N, // this will negate the second argument
        OR,
        XOR
    };

    struct {
        int meanK_ = 50;
        double std_mul_ = 1.0f;
    }sor_params_;

protected:
    typedef pcl::PointXYZRGB PointT;
    typedef flann::L1<float> DistT;
    typedef pcl::Histogram<128> FeatureT;

    typedef boost::property<boost::edge_weight_t, CamConnect> EdgeWeightProperty;
    typedef boost::adjacency_list < boost::vecS, boost::vecS, boost::undirectedS, size_t, EdgeWeightProperty> Graph;
    typedef boost::graph_traits < Graph >::vertex_descriptor Vertex;
    typedef boost::graph_traits < Graph >::edge_descriptor Edge;
    typedef boost::graph_traits<Graph>::vertex_iterator vertex_iter;
    typedef boost::property_map<Graph, boost::vertex_index_t>::type IndexMap;

    Graph gs_;

    Parameter param_;
    kp::ClusterNormalsToPlanes::Parameter p_param_;
    v4r::utils::NMBasedCloudIntegration<pcl::PointXYZRGB>::Parameter nm_int_param_;

    pcl::PointCloud<PointT>::Ptr big_cloud_;
    pcl::PointCloud<PointT>::Ptr big_cloud_segmented_;
    pcl::PointCloud<PointT>::Ptr big_cloud_refined_;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr big_cloud_segmented_refined_;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> vis_, vis_reconstructed_;
    std::vector<int> vis_reconstructed_viewpoint_;
    std::vector<int> vis_viewpoint_;

    std::vector<size_t> LUT_new2old_indices;
    cv::Ptr<SiftGPU> sift_;
    std::vector<modelView> grph_;
    size_t counter_;
    pcl::octree::OctreePointCloudSearch<PointT> octree_;

    void computeAbsolutePoses(const Graph & grph,
                              std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & absolute_poses);

    void
    computeAbsolutePosesRecursive (const Graph &grph,
                                  const Vertex start,
                                  const Eigen::Matrix4f &accum,
                                  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &absolute_poses,
                                  std::vector<bool> &hop_list);

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

        nm_int_param_.final_resolution_ = 0.002f;
        nm_int_param_.min_points_per_voxel_ = 1;
        nm_int_param_.min_weight_ = 0.5f;
        nm_int_param_.octree_resolution_ = 0.002f;
        nm_int_param_.threshold_ss_ = 0.01f;

        big_cloud_.reset(new pcl::PointCloud<PointT>);
        big_cloud_segmented_.reset(new pcl::PointCloud<PointT>);
        big_cloud_refined_.reset(new pcl::PointCloud<PointT>);
        big_cloud_segmented_refined_.reset(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        vis_.reset();
        vis_reconstructed_.reset();
        counter_ = 0;
    }

    void extractEuclideanClustersSmooth (const pcl::PointCloud<PointT>::ConstPtr &cloud,
                const pcl::PointCloud<pcl::Normal> &normals_,
                const std::vector<size_t> &initial, const std::vector<bool> &bg_mask,
                std::vector<size_t> &cluster) const;

    void updatePointNormalsFromSuperVoxels(const pcl::PointCloud<PointT>::Ptr & cloud,
                                           pcl::PointCloud<pcl::Normal>::Ptr & normals_,
                                           const std::vector< size_t > &object_points,
                                           std::vector< size_t > & good_neighbours,
                                           pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &supervoxel_cloud);

    std::vector<bool>
    erodeIndices(const std::vector< bool > &obj_mask,
                      const pcl::PointCloud<PointT> & cloud);


    bool save_model (const std::string &models_dir = "/tmp/dynamic_models/",
                     const std::string &recognition_structure_dir = "/tmp/recognition_structure_dir/",
                     const std::string &model_name = "new_dynamic_model.pcd");

    bool learn_object (const pcl::PointCloud<PointT> &cloud,
                       const Eigen::Matrix4f &camera_pose = Eigen::Matrix4f::Identity(),
                       const std::vector<size_t> &initial_indices = std::vector<size_t>());

    void initialize (int argc, char ** argv);

    void clear()
    {
        big_cloud_->points.clear();
        big_cloud_segmented_->points.clear();
        big_cloud_refined_->points.clear();
        big_cloud_segmented_refined_->points.clear();
        LUT_new2old_indices.clear();
        grph_.clear();
        counter_=0;
        gs_.clearing_graph();
        gs_.clear();
        vis_viewpoint_.clear();
        vis_reconstructed_viewpoint_.clear();
    }

    /**
     * @brief given a point cloud and a normal cloud, this function computes points belonging to a table
     *  (optional: computes smooth clusters for points not belonging to table)
     * @param cloud
     * @param normals
     * @param p_param
     * @param planes
     */
    void extractPlanePoints(const pcl::PointCloud<PointT>::ConstPtr &cloud,
                            const pcl::PointCloud<pcl::Normal>::ConstPtr &normals,
                            std::vector<kp::ClusterNormalsToPlanes::Plane::Ptr> &planes);

    static void getPlanesNotSupportedByObjectMask(const std::vector<kp::ClusterNormalsToPlanes::Plane::Ptr> &planes,
                                                const std::vector<size_t> object_mask,
                                                std::vector<std::vector<int> > &planes_not_on_object,
                                                float ratio=0.25);

    void getPlanesNotSupportedByObjectMask(const std::vector<kp::ClusterNormalsToPlanes::Plane::Ptr> &planes,
                                           const std::vector< bool > &object_mask,
                                           const std::vector< bool > &occlusion_mask,
                                           const pcl::PointCloud<PointT>::ConstPtr &cloud,
                                           std::vector<std::vector<int> > &planes_not_on_object,
                                           float ratio=0.25,
                                           float ratio_occ=0.75) const;
    void visualize();

    /**
     * @brief performs bit wise logical operations
     * @param bit mask1
     * @param bit mask2
     * @param operation (AND, AND_N, OR, XOR)
     * @return output bit mask
     */
    static std::vector<bool> logical_operation(const std::vector<bool> &mask1, const std::vector<bool> &mask2, int operation=MASK_OPERATOR::OR);

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
    static std::vector<bool> createMaskFromIndices(const std::vector<size_t> &indices,
                                size_t image_size);
    static std::vector<bool> createMaskFromIndices(const std::vector<int> &indices,
                               size_t image_size);
    static std::vector<bool> createMaskFromVecIndices(const std::vector<std::vector<int> > &indices,
                                size_t image_size);

    static std::vector<size_t>
    createIndicesFromMask(const std::vector<bool> &mask, bool invert=false);

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
