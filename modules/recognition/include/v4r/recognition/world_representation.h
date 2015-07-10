#ifndef WORLD_REPRESENTATION_H
#define WORLD_REPRESENTATION_H

#ifndef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#endif

#ifndef USE_SIFT_GPU
#define USE_SIFT_GPU
#endif

#include "multiview_object_recognizer_service.h"
#include "singleview_object_recognizer.h"

namespace v4r
{
class worldRepresentation
{
protected:
    typedef faat_pcl::rec_3d_framework::Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;

    boost::shared_ptr<Recognizer> pSingleview_recognizer_;
//    std::vector <multiviewGraph > graph_v_;
    std::map<std::string, v4r::MultiviewRecognizer> mv_environment_;
    bool visualize_output_;
    bool use_robot_pose_;
    std::string models_dir_;


public:
    worldRepresentation()
    {
        visualize_output_ = false;
        use_robot_pose_ = true;
    }

    void clear()
    {
//        graph_v_.clear();
        mv_environment_.clear();
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

//    multiviewGraph& get_current_graph(const std::string &scene_name);

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
//    void setPSingleview_recognizer(const boost::shared_ptr<Recognizer> &value);
//    cv::Ptr<SiftGPU> sift() const;
//    void setSift(const cv::Ptr<SiftGPU> &sift);

    void set_visualize_output(const bool vis_output)
    {
        visualize_output_ = vis_output;
    }


};
}

#endif
