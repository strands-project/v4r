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

class V4R_EXPORTS MultiviewRecognizerParameter
{
public:
    bool transfer_only_verified_hypotheses_;
    size_t max_views_;  ///<

    MultiviewRecognizerParameter() :
        transfer_only_verified_hypotheses_ (true),
        max_views_(3)
    {}
};

template<typename PointT>
class V4R_EXPORTS MultiviewRecognizer : public RecognitionPipeline<PointT>
{
private:
    using RecognitionPipeline<PointT>::scene_;
    using RecognitionPipeline<PointT>::scene_normals_;
    using RecognitionPipeline<PointT>::m_db_;
    using RecognitionPipeline<PointT>::obj_hypotheses_;
    using RecognitionPipeline<PointT>::table_plane_;
    using RecognitionPipeline<PointT>::table_plane_set_;

    typename RecognitionPipeline<PointT>::Ptr recognition_pipeline_;
    MultiviewRecognizerParameter param_;

    struct View
    {
        Eigen::Matrix4f camera_pose_;   ///< camera pose of the view which aligns cloud in registered cloud when multiplied
        std::vector< ObjectHypothesesGroup > obj_hypotheses_;   ///< generated object hypotheses
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        View() :
              camera_pose_ (Eigen::Matrix4f::Identity())
        {}
    };

    std::vector<View> views_;

public:
    MultiviewRecognizer(const MultiviewRecognizerParameter &p = MultiviewRecognizerParameter() )
        : param_ (p)
    { }

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

    void
    clear()
    {
        views_.clear();
    }

    typedef boost::shared_ptr< MultiviewRecognizer<PointT> > Ptr;
    typedef boost::shared_ptr< MultiviewRecognizer<PointT> const> ConstPtr;
};

}

