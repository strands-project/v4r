#ifndef BOOST_GRAPH_EXTENSION_H
#define BOOST_GRAPH_EXTENSION_H
#include <boost/graph/adjacency_list.hpp>
#include <boost/filesystem.hpp>
#include <boost/graph/graph_traits.hpp>
#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <v4r/recognition/source.h>
#include <v4r/recognition/recognizer.h>

typedef pcl::Histogram<128> FeatureT;

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
    typename boost::shared_ptr< pcl::PointCloud<PointT> > scene_f_;
    boost::shared_ptr< pcl::PointCloud<pcl::Normal> > scene_normals_;
    std::vector<int> filtered_scene_indices_;
    Eigen::Matrix4f absolute_pose_;
//    typename boost::shared_ptr< pcl::PointCloud<PointT> > pKeypointsMultipipe_;
//    boost::shared_ptr< pcl::PointCloud<pcl::Normal> > kp_normals_;
    typename std::map<std::string, ObjectHypothesis<PointT> > hypotheses_;
    boost::shared_ptr< pcl::PointCloud<FeatureT > > sift_signatures_;
//    std::vector<float> sift_keypoints_scales_;
    pcl::PointIndices sift_kp_indices_;
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

    //GO3D
    std::vector<float> nguyens_noise_model_weights_;
    std::vector<int> nguyens_kept_indices_;
};

struct V4R_EXPORTS CamConnect
{
    Eigen::Matrix4f transformation_;
    float edge_weight_;
    std::string model_name_;
    size_t source_id_, target_id_;

    explicit CamConnect(float w) :
        edge_weight_(w)
    {

    }

    CamConnect() : edge_weight_(std::numeric_limits<float>::max ())
    {

    }

    bool operator<(const CamConnect& e) const {
        if(edge_weight_ < e.edge_weight_)
            return true;

        return false;
    }

    bool operator<=(const CamConnect& e) const {
        if(edge_weight_ <= e.edge_weight_)
            return true;

        return false;
    }

    bool operator>(const CamConnect& e) const {
        if(edge_weight_ > e.edge_weight_)
            return true;

        return false;
    }

    bool operator>=(const CamConnect& e) const {
        if(edge_weight_ >= e.edge_weight_)
            return true;

        return false;
    }
};

using namespace boost;
}
#endif
