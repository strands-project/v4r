/******************************************************************************
 * Copyright (c) 2015 Thomas Faeulhammer
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


#ifndef V4R_INCREMENTAL_OBJECT_LEARNING_H_
#define V4R_INCREMENTAL_OBJECT_LEARNING_H_
#include <v4r_config.h>
#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/octree.h>

#include <v4r/core/macros.h>
#include <v4r/keypoints/ClusterNormalsToPlanes.h>
#include <v4r/common/PointTypes.h>
#include <v4r/registration/noise_model_based_cloud_integration.h>
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

    explicit CamConnect(float w) : edge_weight(w) { }
    CamConnect() : edge_weight(0.f) { }
    bool operator<(const CamConnect& e) const { return edge_weight < e.edge_weight; }
    bool operator<=(const CamConnect& e) const { return edge_weight <= e.edge_weight; }
    bool operator>(const CamConnect& e) const { return edge_weight > e.edge_weight; }
    bool operator>=(const CamConnect& e) const { return edge_weight >= e.edge_weight; }
};

/**
 * @brief Dynamic object modelling over a sequence of point clouds. The seed is given by
 * the user by means of an initial object mask (indices of the point cloud belonging to the object).
 * The method projects the incrementally learnt object cloud to each view by a transformation given
 * from the camera pose and looks for nearest neighbors. After some filtering, these points are then
 * used for growing the object over the points in the current view. The final model is obtained
 * by a noise-model based cloud integration of all the points labelled as objects.
 *
 * @author Thomas Faeulhammer
 * @date July 2015
 * @ref Robotics and Automation Letters 2016, Faeulhammer et al
 */
class V4R_EXPORTS IOL
{
public:
    class Parameter{
    public:
        double radius_;
        double eps_angle_;
        double dist_threshold_growing_;
        double voxel_resolution_;
        double seed_resolution_;
        double ratio_supervoxel_;
        double chop_z_; /// @brief cut-off distance in meters
        bool do_erosion_;
        bool transfer_indices_from_latest_frame_only_;
        size_t min_points_for_transferring_;
        int normal_method_;
        bool do_mst_refinement_;
        double ratio_cluster_obj_supported_;
        double ratio_cluster_occluded_;
        Parameter (double radius = 0.005f,
                   double eps_angle = 0.95f,
                   double dist_threshold_growing = 0.05f,
                   double voxel_resolution = 0.005f,
                   double seed_resolution = 0.03f,
                   double ratio = 0.25f,
                   double chop_z = std::numeric_limits<double>::quiet_NaN(),
                   bool do_erosion = true,
                   bool transfer_indices_from_latest_frame_only = false,
                   size_t min_points_for_transferring = 10,
                   int normal_method = 0,
                   bool do_mst_refinement = true,
                   double ratio_cluster_obj_supported = 0.25,
                   double ratio_cluster_occluded = 0.75):
            radius_(radius),
            eps_angle_(eps_angle),
            dist_threshold_growing_(dist_threshold_growing),
            voxel_resolution_(voxel_resolution),
            seed_resolution_(seed_resolution),
            ratio_supervoxel_(ratio),
            chop_z_(chop_z),
            do_erosion_(do_erosion),
            transfer_indices_from_latest_frame_only_(transfer_indices_from_latest_frame_only),
            min_points_for_transferring_(min_points_for_transferring),
            normal_method_(normal_method),
            do_mst_refinement_(do_mst_refinement),
            ratio_cluster_obj_supported_ (ratio_cluster_obj_supported),
            ratio_cluster_occluded_(ratio_cluster_occluded)
        {
        }

    }param_;

protected:
    typedef pcl::PointXYZRGB PointT;

public:
    struct {
        int meanK_ = 10;
        double std_mul_ = 2.0f;
    }sor_params_;

    v4r::ClusterNormalsToPlanes::Parameter p_param_;
    v4r::NMBasedCloudIntegration<PointT>::Parameter nm_int_param_;

protected:

    typedef boost::property<boost::edge_weight_t, CamConnect> EdgeWeightProperty;
    typedef boost::adjacency_list < boost::vecS, boost::vecS, boost::undirectedS, size_t, EdgeWeightProperty> Graph;
    typedef boost::graph_traits < Graph >::vertex_descriptor Vertex;
    typedef boost::graph_traits < Graph >::edge_descriptor Edge;
    typedef boost::graph_traits<Graph>::vertex_iterator vertex_iter;
    typedef boost::property_map<Graph, boost::vertex_index_t>::type IndexMap;

    std::vector< pcl::PointCloud<PointT>::Ptr > keyframes_used_;  /// @brief all keyframes containing the object with sufficient number of points
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > cameras_used_;  /// @brief camera pose belonging to the keyframes containing the object with sufficient number of points
    std::vector<std::vector<size_t> > object_indices_clouds_used_;  /// @brief indices of the object in all keyframes containing the object with sufficient number of points

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_normals_oriented_;

    Graph gs_;

    pcl::PointCloud<PointT>::Ptr big_cloud_;
    pcl::PointCloud<PointT>::Ptr big_cloud_segmented_;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr big_cloud_segmented_refined_;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> vis_, vis_reconstructed_, vis_seg_;
    std::vector<int> vis_reconstructed_viewpoint_;
    std::vector<int> vis_viewpoint_;

    std::vector<modelView> grph_;
    pcl::octree::OctreePointCloudSearch<PointT> octree_;

    void computeAbsolutePoses(const Graph & grph,
                              std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & absolute_poses);

    void
    computeAbsolutePosesRecursive (const Graph &grph,
                                  const Vertex start,
                                  const Eigen::Matrix4f &accum,
                                  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &absolute_poses,
                                  std::vector<bool> &hop_list);

    /**
     * @brief checks if there sufficient points of the plane are labelled as object
     * @param Plane Model
     * @return true if plane has enough labelled object points
     */
    bool
    plane_has_object (const modelView::SuperPlane &sp) const
    {
        const size_t num_pts = sp.within_chop_z_indices.size();
        const size_t num_obj = sp.object_indices.size();

        return num_pts > 0 && static_cast<double>(num_obj)/num_pts > param_.ratio_cluster_obj_supported_;
    }

    /**
     * @brief checks if there sufficient points of the plane that are visible in the first view
     * @param Plane Model
     * @return true if plane is visibly enough in first view point
     */
    bool
    plane_is_visible (const modelView::SuperPlane &sp) const
    {
        const size_t num_pts = sp.within_chop_z_indices.size();
        const size_t num_vis = sp.visible_indices.size();

        return num_pts > 0 && static_cast<double>(num_vis)/num_pts > param_.ratio_cluster_occluded_;
    }

    bool
    plane_is_filtered (const modelView::SuperPlane &sp) const
    {
        return plane_is_visible(sp) && !plane_has_object(sp);
    }

    /**
     * @brief checks if two planes can be merged based on orientation
     * @param First plane model
     * @param Second plane model
     * @return true if planes are similar
     */
    bool merging_planes_reasonable(const modelView::SuperPlane &sp1, const modelView::SuperPlane &sp2) const;

    /**
     * @brief sets pixels in given mask belonging to invalid points to false
     * @param input cloud
     * @param mask
     */
    void
    remove_nan_points(const pcl::PointCloud<PointT> &cloud, std::vector<bool> &mask)
    {
        assert(mask.size()==cloud.size());
        for (size_t i=0; i<mask.size(); i++)
        {
            if(mask[i] && !pcl::isFinite(cloud.points[i]))
                mask[i] = false;
        }
    }

    /**
     * @brief given a point cloud and a normal cloud, this function computes points belonging to a table
     *  (optional: computes smooth clusters for points not belonging to table)
     * @param[in] cloud The input cloud from which smooth clusters / planes are being calculated
     * @param[in] normals The normal cloud corresponding to the input cloud
     * @param[out] planes The resulting smooth clusters and planes
     */
    void extractPlanePoints(const pcl::PointCloud<PointT>::ConstPtr &cloud,
                            const pcl::PointCloud<pcl::Normal>::ConstPtr &normals,
                            std::vector<v4r::ClusterNormalsToPlanes::Plane::Ptr> &planes);

    /**
     * @brief given a set of clusters, this function returns the clusters which have less than ratio% object points or ratio_occ% occlusion points
     * Points further away than param_.chop_z_ are neglected
     * @param[in] planes - set of input clusters
     * @param[in] object_mask - binary mask of object pixels
     * @param[in] occlusion_mask - binary mask of pixels which are neither object nor background (e.g. pixels that are occluded when transferred into the labelled frame)
     * @param[in] cloud - point cloud of the scene
     * @param[out] planes_not_on_object - output set of clusters
     * @param ratio - threshold percentage when a cluster is considered as belonging to an object
     * @param ratio_occ - threshold percentage when a cluster is considered as being occluded
     */
    void computePlaneProperties(const std::vector<v4r::ClusterNormalsToPlanes::Plane::Ptr> &planes,
                                           const std::vector< bool > &object_mask,
                                           const std::vector< bool > &occlusion_mask,
                                           const pcl::PointCloud<PointT>::ConstPtr &cloud,
                                           std::vector<modelView::SuperPlane> &super_planes) const;

    /**
     * @brief transfers object points nearest neighbor search in dest frame.
     * @param[in] object_points points which are transferred and looked for in search cloud
     * @param[in] octree search space for transferred points
     * @param[out] obj_mask nearest neighbors points within a specified radius highlighted (true) in object mask
     */
    void nnSearch(const pcl::PointCloud<PointT> &object_points, pcl::octree::OctreePointCloudSearch<PointT> &octree,  std::vector<bool> &obj_mask);

    /**
     * @brief Nearest Neighbor Search for points transferred into search cloud
     * @param transferred object points
     * @param[in] point cloud to be checked for proximity to transferred points
     * @param[out] mask with nearest neighbors points within a specified radius highlighted (true) in object mask
     */
    void nnSearch(const pcl::PointCloud<PointT> &object_points, const pcl::PointCloud<PointT>::ConstPtr &search_cloud,  std::vector<bool> &obj_mask);

    /**
     * @brief extracts smooth Euclidean clusters of a given point cloud
     * @param input cloud
     * @param normals_
     * @param initial_mask
     * @param bg_mask
     * @return
     */
    std::vector<bool>
    extractEuclideanClustersSmooth (const pcl::PointCloud<PointT>::ConstPtr &cloud,
                                    const pcl::PointCloud<pcl::Normal> &normals_,
                                    const std::vector<bool> &initial_mask,
                                    const std::vector<bool> &bg_mask) const;

    void updatePointNormalsFromSuperVoxels(const pcl::PointCloud<PointT>::Ptr & cloud,
                                           pcl::PointCloud<pcl::Normal>::Ptr & normals_,
                                           const std::vector<bool> &obj_mask,
                                           std::vector<bool> &obj_mask_out,
                                           pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &supervoxel_cloud,
                                           pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &supervoxel_cloud_organized);

    std::vector<bool>
    erodeIndices(const std::vector< bool > &obj_mask, const pcl::PointCloud<PointT> & cloud);

public:

    IOL (const Parameter &p=Parameter()) : octree_(0.005f)
    {
        param_ = p;
        // Parameters for smooth clustering / plane segmentation
        p_param_.thrAngle=45;
        p_param_.inlDist=0.05;
        p_param_.minPoints=5000;    // minimum number for a plane to be segmented
        p_param_.least_squares_refinement=true;
        p_param_.smooth_clustering=false;
        p_param_.thrAngleSmooth=30;
        p_param_.inlDistSmooth=0.02;
        p_param_.minPointsSmooth=50;    // minimum number for a segment other than a plane

        nm_int_param_.min_points_per_voxel_ = 1;
        nm_int_param_.octree_resolution_ = 0.002f;
        nm_int_param_.average_ = false;

        big_cloud_.reset(new pcl::PointCloud<PointT>);
        big_cloud_segmented_.reset(new pcl::PointCloud<PointT>);
        big_cloud_segmented_refined_.reset(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        vis_.reset();
        vis_reconstructed_.reset();
        cloud_normals_oriented_.reset (new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    }

    /**
     * @brief saves the learned model to disk
     * @param[in] directory where to save the object model
     * @param[in] name of the object model
     * @return
     */
    bool save_model (const std::string &models_dir = "/tmp/dynamic_models/",
                     const std::string &model_name = "new_dynamic_model",
                     bool save_individual_views = true);

    bool write_model_to_disk (const std::string &models_dir = "/tmp/dynamic_models/",
                     const std::string &model_name = "new_dynamic_model",
                              bool save_individual_views = true);

    bool learn_object (const pcl::PointCloud<PointT> &cloud,
                       const Eigen::Matrix4f &camera_pose = Eigen::Matrix4f::Identity(),
                       const std::vector<size_t> &initial_indices = std::vector<size_t>());

    void initSIFT ();

    /**
     * @brief clears the memory from the currently learned object.
     * Needs to be called before learning a new object model.
     */
    void clear()
    {
        big_cloud_->points.clear();
        big_cloud_segmented_->points.clear();
        big_cloud_segmented_refined_->points.clear();
        grph_.clear();
        gs_.clearing_graph();
        gs_.clear();
        vis_viewpoint_.clear();
        vis_reconstructed_viewpoint_.clear();
    }

    /**
     * @brief This shows the learned object together with all the intermediate steps using pcl viewer
     */
    void visualize();
    void visualize_clusters();


    /**
     * @brief This creates images of all intermediate steps of the object learning and writes them to disk
     * @param[in] path - folder where to write the files to
     */
    void writeImagesToDisk(const std::string &path = std::string("/tmp/dol_images/"), bool crop=false);

    /**
     * @brief transforms each keyframe to global coordinate system using given camera
     *  pose and does segmentation based on the computed object indices to create
     *  reconstructed point cloud of scene and object model
     */
    void createBigCloud();

    /**
     * @brief prints parameters to the given output stream
     * @param output stream
     */
    void printParams(std::ostream &ostr = std::cout) const;
};

}
}

#endif //DO_LEARNING_H_
