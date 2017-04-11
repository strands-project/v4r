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

#include <v4r/recognition/recognition_pipeline.h>
#include <omp.h>

namespace v4r
{
template<typename PointT>
class V4R_EXPORTS MultiRecognitionPipeline : public RecognitionPipeline<PointT>
{
private:
    using RecognitionPipeline<PointT>::scene_;
    using RecognitionPipeline<PointT>::scene_normals_;
    using RecognitionPipeline<PointT>::m_db_;
    using RecognitionPipeline<PointT>::normal_estimator_;
    using RecognitionPipeline<PointT>::obj_hypotheses_;
    using RecognitionPipeline<PointT>::table_plane_;
    using RecognitionPipeline<PointT>::table_plane_set_;
    using RecognitionPipeline<PointT>::vis_param_;

    std::vector<typename RecognitionPipeline<PointT>::Ptr > recognition_pipelines_;

    omp_lock_t rec_lock_;

public:
    MultiRecognitionPipeline() { }

    /**
         * @brief initialize the recognizer (extract features, create FLANN,...)
         * @param[in] path to model database. If training directory exists, will load trained model from disk; if not, computed features will be stored on disk (in each
         * object model folder, a feature folder is created with data)
         * @param[in] retrain if set, will re-compute features and store to disk, no matter if they already exist or not
         */
    void
    initialize(const std::string &trained_dir = "", bool retrain = false);

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
    addRecognitionPipeline(typename RecognitionPipeline<PointT>::Ptr & rec)
    {
        recognition_pipelines_.push_back(rec);
    }


    /**
         * @brief needNormals
         * @return true if normals are needed, false otherwise
         */
    bool
    needNormals() const
    {
        for(size_t r_id=0; r_id < recognition_pipelines_.size(); r_id++)
        {
            if(recognition_pipelines_[r_id]->needNormals())
                return true;
        }
        return false;
    }

    /**
         * @brief getFeatureType
         * @return
         */
    size_t
    getFeatureType() const
    {
        size_t feat_type = 0;
        for(size_t r_id=0; r_id < recognition_pipelines_.size(); r_id++)
            feat_type += recognition_pipelines_[r_id]->getFeatureType();

        return feat_type;
    }

    /**
         * @brief requiresSegmentation
         * @return
         */
    bool
    requiresSegmentation() const
    {
        bool ret_value = false;
        for(size_t i=0; (i < recognition_pipelines_.size()) && !ret_value; i++)
            ret_value = recognition_pipelines_[i]->requiresSegmentation();

        return ret_value;
    }

    typedef boost::shared_ptr< MultiRecognitionPipeline<PointT> > Ptr;
    typedef boost::shared_ptr< MultiRecognitionPipeline<PointT> const> ConstPtr;
};
}
