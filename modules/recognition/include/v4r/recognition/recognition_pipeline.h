/******************************************************************************
 * Copyright (c) 2013 Aitor Aldoma, Thomas Faeulhammer
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


#pragma once

#include <pcl/common/common.h>

#include <v4r_config.h>
#include <v4r/common/normals.h>
#include <v4r/common/pcl_visualization_utils.h>
#include <v4r/core/macros.h>
#include <v4r/recognition/object_hypothesis.h>
#include <v4r/recognition/source.h>

namespace v4r
{

/**
 * @brief The recognition pipeline class is an abstract class that represents a
 * pipeline for object recognition. It will generated groups of object hypotheses.
 * For a global recognition pipeline, each segmented cluster from the input cloud will store its object hypotheses into one group.
 * For all other pipelines, each group will only contain one object hypothesis.
 * @author Thomas Faeulhammer, Aitor Aldoma
 */
template<typename PointT>
class V4R_EXPORTS RecognitionPipeline
{
public:
    typedef boost::shared_ptr< RecognitionPipeline<PointT> > Ptr;
    typedef boost::shared_ptr< RecognitionPipeline<PointT> const> ConstPtr;

protected:
    typedef Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;

    typename pcl::PointCloud<PointT>::ConstPtr scene_; ///< Point cloud to be recognized
    pcl::PointCloud<pcl::Normal>::ConstPtr scene_normals_; ///< associated normals
    typename Source<PointT>::ConstPtr m_db_;  ///< model data base
    std::vector< ObjectHypothesesGroup<PointT> > obj_hypotheses_;   ///< generated object hypotheses
    typename NormalEstimator<PointT>::Ptr normal_estimator_;    ///< normal estimator used for computing surface normals (currently only used at training)
    Eigen::Vector4f table_plane_;
    bool table_plane_set_;

    PCLVisualizationParams::ConstPtr vis_param_;

public:
    RecognitionPipeline() :
        table_plane_(Eigen::Vector4f::Identity()),
        table_plane_set_(false)
    {}

    virtual ~RecognitionPipeline(){}

    virtual size_t getFeatureType() const = 0;

    virtual bool needNormals() const = 0;

    /**
     * @brief initialize the recognizer (extract features, create FLANN,...)
     * @param[in] path to model database. If training directory exists, will load trained model from disk; if not, computed features will be stored on disk (in each
     * object model folder, a feature folder is created with data)
     * @param[in] retrain if set, will re-compute features and store to disk, no matter if they already exist or not
     */
    virtual void
    initialize(const std::string &trained_dir = "", bool retrain = false)
    {
        (void) retrain;
        (void) trained_dir;
        PCL_WARN("initialize is not implemented for this class.");
    }

    /**
     * @brief setInputCloud
     * @param cloud to be recognized
     */
    void
    setInputCloud (const typename pcl::PointCloud<PointT>::ConstPtr cloud)
    {
        scene_ = cloud;
    }


    /**
     * @brief getObjectHypothesis
     * @return generated object hypothesis
     */
    std::vector<ObjectHypothesesGroup<PointT> >
    getObjectHypothesis() const
    {
        return obj_hypotheses_;
    }

    /**
     * @brief setSceneNormals
     * @param normals normals of the input cloud
     */
    void
    setSceneNormals(const pcl::PointCloud<pcl::Normal>::ConstPtr &normals)
    {
        scene_normals_ = normals;
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

    /**
     * @brief getModelDatabase
     * @return model database
     */
    typename Source<PointT>::ConstPtr
    getModelDatabase() const
    {
        return m_db_;
    }

    void
    setTablePlane( const Eigen::Vector4f &table_plane)
    {
        table_plane_ = table_plane;
        table_plane_set_ = true;
    }


    /**
     * @brief setNormalEstimator sets the normal estimator used for computing surface normals (currently only used at training)
     * @param normal_estimator
     */
    void
    setNormalEstimator(const typename NormalEstimator<PointT>::Ptr &normal_estimator)
    {
        normal_estimator_ = normal_estimator;
    }


    /**
     * @brief setVisualizationParameter sets the PCL visualization parameter (only used if some visualization is enabled)
     * @param vis_param
     */
    void
    setVisualizationParameter(const PCLVisualizationParams::ConstPtr &vis_param)
    {
        vis_param_ = vis_param;
    }

    virtual bool requiresSegmentation() const = 0;
    virtual void recognize () = 0;
};
}
