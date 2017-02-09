/******************************************************************************
 * Copyright (c) 2017 Thomas Faeulhammer
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

/**
*
*      @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
*      @date January, 2017
*      @brief multiview object instance recognizer
*/

#pragma once

#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <v4r_config.h>
#include <v4r/recognition/recognition_pipeline.h>

namespace v4r
{

template<typename PointT>
class V4R_EXPORTS MultiviewRecognizer : public RecognitionPipeline<PointT>
{
private:
    using RecognitionPipeline<PointT>::scene_;
    using RecognitionPipeline<PointT>::scene_normals_;
    using RecognitionPipeline<PointT>::m_db_;
    using RecognitionPipeline<PointT>::obj_hypotheses_;

    typename RecognitionPipeline<PointT>::Ptr recognition_pipeline_;

    struct View
    {
        typename pcl::PointCloud<PointT>::ConstPtr cloud_; ///< Point cloud of the scene
        pcl::PointCloud<pcl::Normal>::ConstPtr cloud_normals_; ///< associated scene normals
        Eigen::Matrix4f camera_pose_;   ///< camera pose of the view which aligns cloud in registered cloud when multiplied

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        View(const typename pcl::PointCloud<PointT>::ConstPtr cloud,
             const pcl::PointCloud<pcl::Normal>::ConstPtr cloud_normals,
             const Eigen::Matrix4f &camera_pose = Eigen::Matrix4f::Identity() )
            :
              cloud_ (cloud),
              cloud_normals_ (cloud_normals),
              camera_pose_ (camera_pose)
        {}
    };

    std::vector<View> views_;

public:
    MultiviewRecognizer() { }

    void
    initialize(const std::string &trained_dir = "", bool retrain = false)
    {
        recognition_pipeline_->initialize( trained_dir, retrain );
    }

    /**
    * @brief recognize
    */
    void
    recognize();

    /**
    * @brief oh_tmp
    * @param rec recognition pipeline (local or global)
    */
    void
    setSingleViewRecognitionPipeline(typename RecognitionPipeline<PointT>::Ptr & rec)
    {
        recognition_pipeline_ = rec;
    }


    /**
    * @brief needNormals
    * @return true if normals are needed, false otherwise
    */
    bool
    needNormals() const
    {
        return recognition_pipeline_->needNormals();
    }

    /**
         * @brief getFeatureType
         * @return
         */
    size_t
    getFeatureType() const
    {
        return recognition_pipeline_->getFeatureType();
    }

    /**
    * @brief requiresSegmentation
    * @return
    */
    bool
    requiresSegmentation() const
    {
        return recognition_pipeline_->requiresSegmentation();
    }

    /**
     * @brief addView add a view to the recognizer
     * @param cloud
     * @param cloud_normals
     * @param camera_pose
     */
    void
    addView(const typename pcl::PointCloud<PointT>::ConstPtr cloud,
            const pcl::PointCloud<pcl::Normal>::ConstPtr cloud_normals = pcl::PointCloud<pcl::Normal>::ConstPtr(),
            const Eigen::Matrix4f &camera_pose = Eigen::Matrix4f::Identity() )
    {
        View v(cloud, cloud_normals, camera_pose);
        views_.push_back(v);
    }

    typedef boost::shared_ptr< MultiviewRecognizer<PointT> > Ptr;
    typedef boost::shared_ptr< MultiviewRecognizer<PointT> const> ConstPtr;
};

}

