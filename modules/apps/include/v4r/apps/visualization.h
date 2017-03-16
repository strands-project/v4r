#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <v4r/core/macros.h>
#include <v4r/common/pcl_visualization_utils.h>
#include <v4r/recognition/local_feature_matching.h>
#include <v4r/recognition/object_hypothesis.h>
#include <v4r/recognition/source.h>

namespace v4r
{

/**
 * @brief Visualization framework for object recognition
 * @author Thomas Faeulhammer
 * @date May 2016
 */
template<typename PointT>
class V4R_EXPORTS ObjectRecognitionVisualizer
{
private:
    typename pcl::PointCloud<PointT>::ConstPtr cloud_; ///< input cloud
    typename pcl::PointCloud<PointT>::ConstPtr processed_cloud_; ///< input cloud
    typename pcl::PointCloud<pcl::Normal>::ConstPtr normals_; ///< input normals

    std::vector< ObjectHypothesesGroup<PointT> > generated_object_hypotheses_;   ///< generated object hypotheses
    std::vector< ObjectHypothesesGroup<PointT> > generated_object_hypotheses_refined_;   ///< (ICP refined) generated object hypotheses
    std::vector< typename ObjectHypothesis<PointT>::Ptr > verified_object_hypotheses_; ///< verified object hypotheses
    mutable boost::shared_ptr<pcl::visualization::PCLVisualizer> vis_;
    mutable int vp1a_, vp2_, vp3_, vp1b_, vp2b_;
    mutable std::vector<std::string> coordinate_axis_ids_;

    typename Source<PointT>::ConstPtr m_db_;  ///< model data base
    std::map<std::string, typename LocalObjectModel::ConstPtr> model_keypoints_; ///< pointer to local model database (optional: required if visualization of feature matching is desired)

    PCLVisualizationParams::ConstPtr vis_param_; ///< visualization parameters
    void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event) const;
    void pointPickingEventOccured (const pcl::visualization::PointPickingEvent &event) const;
    void flipOpacity(const std::string& cloud_name, double max_opacity = 1.) const;

    int model_resolution_mm_; ///< resolution of the visualized object model in mm

    mutable pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree_;


    /**
     * @brief The Line class is a utility class to visualize and toggle the correspondences between model and scene keypoints
     */
    class Line
    {
    private:
        bool is_visible_;
        pcl::visualization::PCLVisualizer::Ptr visLine_;

    public:
        PointT p_, q_;
        double r_, g_, b_;
        std::string id_;
        int viewport_;

        Line(pcl::visualization::PCLVisualizer::Ptr vis, const PointT &p, const PointT &q, double r, double g, double b, const std::string &id, int viewport = 0)
            :
              is_visible_ (false),
              visLine_(vis),
              p_(p),
              q_(q),
              r_(r),
              g_(g),
              b_(b),
              id_(id),
              viewport_(viewport)
        {}

        void operator()()
        {
            if(!is_visible_)
            {
                visLine_->addLine( p_, q_, r_, g_, b_, id_, viewport_);
                is_visible_ = true;
                ;
            }
            else
            {
                visLine_->removeShape(id_, viewport_);
                visLine_->removeShape(id_);
                is_visible_ = false;
            }
        }
    };
    mutable std::vector<Line> corrs_;
    mutable std::vector<Line> corrs2_;

public:
    ObjectRecognitionVisualizer( const PCLVisualizationParams::ConstPtr &vis_param)
        : vis_param_(vis_param)
    {
    }

    ObjectRecognitionVisualizer( )
    {
        PCLVisualizationParams::Ptr vis_param ( new PCLVisualizationParams() );
        vis_param_ = vis_param;
    }

    /**
     * @brief visualize
     */
    void visualize() const;

    /**
     * @brief setCloud
     * @param[in] cloud input cloud
     */
    void
    setCloud ( const typename pcl::PointCloud<PointT>::ConstPtr cloud )
    {
        cloud_ = cloud;
    }

    /**
     * @brief setProcessedCloud
     * @param[in] cloud processed cloud
     */
    void
    setProcessedCloud ( const typename pcl::PointCloud<PointT>::ConstPtr cloud )
    {
        processed_cloud_ = cloud;
    }

    /**
     * @brief setNormals visualizes normals of input cloud
     * @param normal cloud
     */
    void
    setNormals ( const pcl::PointCloud<pcl::Normal>::ConstPtr &normals )
    {
        normals_ = normals;
    }

    /**
     * @brief setGeneratedObjectHypotheses
     * @param[in] goh generated hypotheses
     */
    void
    setGeneratedObjectHypotheses ( const std::vector< ObjectHypothesesGroup<PointT> > &goh )
    {
        generated_object_hypotheses_ = goh;
    }

    /**
     * @brief setRefinedGeneratedObjectHypotheses
     * @param[in] goh (ICP refined) generated hypotheses
     */
    void
    setRefinedGeneratedObjectHypotheses ( const std::vector< ObjectHypothesesGroup<PointT> > &goh )
    {
        generated_object_hypotheses_refined_ = goh;
    }

    /**
     * @brief setVerifiedObjectHypotheses
     * @param[in] voh verified hypotheses
     */
    void
    setVerifiedObjectHypotheses ( const std::vector<typename ObjectHypothesis<PointT>::Ptr > &voh )
    {
        verified_object_hypotheses_ = voh;
    }

    /**
     * @brief setLocalModelDatabase this function allows to additionally show the keypoint correspondences between scene and model
     * @param lomdb Local ModelDatabase
     */
    void
    setLocalModelDatabase(const std::map<std::string, typename LocalObjectModel::ConstPtr> &model_keypoints)
    {
        model_keypoints_ = model_keypoints;
    }


    /**
     * @brief setModelDatabase
     * @param m_db model database
     */
    void
    setModelDatabase(const typename Source<PointT>::ConstPtr &m_db)
    {
        m_db_ = m_db;
    }


    typedef boost::shared_ptr< ObjectRecognitionVisualizer<PointT> > Ptr;
    typedef boost::shared_ptr< ObjectRecognitionVisualizer<PointT> const> ConstPtr;
};

}
