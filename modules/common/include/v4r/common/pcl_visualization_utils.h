#pragma once

#include <v4r/core/macros.h>
#include <pcl/visualization/cloud_viewer.h>
#include <string>

namespace v4r
{
class V4R_EXPORTS pcl_visualizer
{
public:
    static std::vector<int> visualization_framework (pcl::visualization::PCLVisualizer &vis,
                                                     size_t number_of_views,
                                                     size_t number_of_subwindows_per_view,
                                                     const std::vector<std::string> &title_subwindows = std::vector<std::string>());
};

class V4R_EXPORTS PCLVisualizationParams
{
public:
    typedef boost::shared_ptr< PCLVisualizationParams > Ptr;
    typedef boost::shared_ptr< PCLVisualizationParams const> ConstPtr;

    bool no_text_; ///< optimizes visualization for paper (no text labels...)
    int vis_pt_size_;
    Eigen::Vector3f text_color_;
    Eigen::Vector3i bg_color_;
    int fontsize_;
    float coordinate_axis_scale_;
    PCLVisualizationParams() :
        no_text_(false),
        vis_pt_size_ (10),
        text_color_ (Eigen::Vector3f(0.f,0.f,0.f)),
        bg_color_ ( Eigen::Vector3i(255,255,255)),
        fontsize_ (12),
        coordinate_axis_scale_(0.4f)
    { }
};
}

