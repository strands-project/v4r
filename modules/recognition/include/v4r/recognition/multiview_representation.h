#ifndef MULTI_VIEW_REPRESENTATION_H
#define MULTI_VIEW_REPRESENTATION_H
#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <v4r/common/plane_model.h>
#include <v4r/recognition/model.h>
#include <v4r/recognition/local_rec_object_hypotheses.h>

namespace v4r
{

template<typename PointT>
class V4R_EXPORTS View
{
protected:
    typedef Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;

public:
    View();
    typename boost::shared_ptr< pcl::PointCloud<PointT> > scene_;
//    typename boost::shared_ptr< pcl::PointCloud<PointT> > scene_f_;
    boost::shared_ptr< pcl::PointCloud<pcl::Normal> > scene_normals_;
    std::vector<int> filtered_scene_indices_;
    Eigen::Matrix4f absolute_pose_;
    typename pcl::PointCloud<PointT>::Ptr scene_kp_;
    pcl::PointCloud<pcl::Normal>::Ptr scene_kp_normals_;
    typename std::map<std::string, LocalObjectHypothesis<PointT> > hypotheses_;
    std::vector<std::vector<float> > sift_signatures_;
//    std::vector<float> sift_keypoints_scales_;
    std::vector<int> sift_kp_indices_;
    Eigen::Matrix4f transform_to_world_co_system_;
    bool has_been_hopped_;
    double cumulative_weight_to_new_vrtx_;
    pcl::PointIndices kp_indices_;
    size_t id_;

    /** @brief: generated object hypotheses from correspondence grouping (before verification) */
    std::vector<ModelTPtr> models_;
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_;
    std::vector<PlaneModel<PointT> > planes_;

    /** @brief boolean vector defining if model or plane is verified (models are first in the vector and its size is equal to models_) */
    std::vector<bool> model_or_plane_is_verified_;

    /** @brief vector defining from which view the object hypothesis comes from */
    std::vector<size_t> origin_view_id_;

    std::vector<std::vector<float> >  pt_properties_; /// @brief noise properties for each point
};

struct V4R_EXPORTS CamConnect
{
    Eigen::Matrix4f transformation_;
    float edge_weight_;
    std::string model_name_;
    size_t source_id_, target_id_;

    explicit CamConnect(float w) : edge_weight_(w) { }
    CamConnect() : edge_weight_(std::numeric_limits<float>::max ()) { }
    bool operator<(const CamConnect& e) const { return edge_weight_ < e.edge_weight_; }
    bool operator<=(const CamConnect& e) const { return edge_weight_ <= e.edge_weight_; }
    bool operator>(const CamConnect& e) const { return edge_weight_ > e.edge_weight_; }
    bool operator>=(const CamConnect& e) const { return edge_weight_ >= e.edge_weight_; }
};

}
#endif
