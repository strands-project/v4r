#ifndef WORLD_REPRESENTATION_H
#define WORLD_REPRESENTATION_H

#ifndef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#endif

#include "multiview_object_recognizer_service.h"
#include "singleview_object_recognizer.h"

class worldRepresentation
{
private:
    boost::shared_ptr<Recognizer> pSingleview_recognizer_;
    std::vector <multiviewGraph > graph_v_;
    bool visualize_output_, scene_to_scene_, use_robot_pose_;
    int extension_mode_; // defines the method for extending hypotheses (0 = keypoint correspondences; 1 = full hypotheses only)
    int icp_iter_;
    int opt_type_;
    double chop_at_z_;
    std::string models_dir_;
    cv::Ptr<SiftGPU> sift_;
    size_t max_vertices_in_graph_;
    double distance_keypoints_get_discarded_;

public:
    worldRepresentation()
    {
        distance_keypoints_get_discarded_ = 0.005*0.005;
        max_vertices_in_graph_ = 4;
        visualize_output_ = false;
        scene_to_scene_ = true;
        use_robot_pose_ = false;
        extension_mode_ = 0;
    }

    void clear()
    {
        graph_v_.clear();
    }

    bool recognize (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &pInput,
                                         const std::string &scene_name,
                                         const std::string &view_name,
                                         const size_t &timestamp,
                                         const std::vector<double> &global_trans_v,
                                         std::vector<ModelTPtr> &models_mv,
                                         std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transforms_mv,
                                         const std::string &filepath_or_results_mv = std::string(""),
                                         const std::string &filepath_or_results_sv = std::string("")
                                         );

    multiviewGraph& get_current_graph(const std::string scene_name);

    int icp_iter() const;
    void setIcp_iter(int icp_iter);
    bool visualize_output() const;
    void setVisualize_output(bool visualize_output);
    int opt_type() const;
    void setOpt_type(int opt_type);
    double chop_at_z() const;
    void setChop_at_z(double chop_at_z);
    std::string models_dir() const;
    void setModels_dir(const std::string &models_dir);
    void setPSingleview_recognizer(const boost::shared_ptr<Recognizer> &value);
    cv::Ptr<SiftGPU> sift() const;
    void setSift(const cv::Ptr<SiftGPU> &sift);
    void set_scene_to_scene(const bool scene_to_scene)
    {
        scene_to_scene_ = scene_to_scene;
    }
    void set_max_vertices_in_graph(const size_t num)
    {
        max_vertices_in_graph_ = num;
    }

    void set_distance_keypoints_get_discarded(const double distance)
    {
        distance_keypoints_get_discarded_ = distance;
    }

    void set_visualize_output(const bool vis_output)
    {
        visualize_output_ = vis_output;
    }

    void set_use_robot_pose(const bool use_robot_pose)
    {
        use_robot_pose_ = use_robot_pose;
    }

    void set_extension_mode(const int extension_mode)
    {
        extension_mode_ = extension_mode;
    }
};

#endif
