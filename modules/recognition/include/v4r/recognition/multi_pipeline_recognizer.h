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

#ifndef MULTI_PIPELINE_RECOGNIZER_H_
#define MULTI_PIPELINE_RECOGNIZER_H_

#include <v4r/recognition/recognizer.h>
#include <v4r/recognition/local_recognizer.h>
#include <v4r/common/graph_geometric_consistency.h>
#include <omp.h>

namespace v4r
{
    template<typename PointT>
    class V4R_EXPORTS MultiRecognitionPipeline : public Recognizer<PointT>
    {
      protected:
        std::vector<typename boost::shared_ptr<Recognizer<PointT> > > recognizers_;

      private:
        using Recognizer<PointT>::scene_;
        using Recognizer<PointT>::scene_normals_;
        using Recognizer<PointT>::obj_hypotheses_;
        using Recognizer<PointT>::hv_algorithm_;
        using Recognizer<PointT>::hypothesisVerification;
        using Recognizer<PointT>::getDataSource;
        using Recognizer<PointT>::param_;
        using Recognizer<PointT>::source_;

        typedef typename pcl::PointCloud<PointT>::Ptr PointTPtr;
        typedef typename pcl::PointCloud<PointT>::ConstPtr ConstPointTPtr;

        typedef Model<PointT> ModelT;
        typedef boost::shared_ptr<ModelT> ModelTPtr;
        omp_lock_t rec_lock_;

        typename boost::shared_ptr<GraphGeometricConsistencyGrouping<PointT, PointT> > cg_algorithm_;  /// @brief algorithm for correspondence grouping
        typename pcl::PointCloud<PointT>::Ptr scene_keypoints_; /// @brief keypoints of the scene
        pcl::PointCloud<pcl::Normal>::Ptr scene_kp_normals_;
        std::map<std::string, LocalObjectHypothesis<PointT> > local_obj_hypotheses_;   /// @brief stores feature correspondences

        /**
         * @brief removes all scene keypoints not having a correspondence in the model database and adapt correspondences indices accordingly
         */
        void
        compress()
        {
            std::vector<bool> kp_has_correspondence(scene_keypoints_->points.size(), false);
            for (const auto &oh : local_obj_hypotheses_) {
                for (const auto &corr : oh.second.model_scene_corresp_) {
                    kp_has_correspondence[corr.index_match] = true;
                }
            }

            std::vector<int> lut (scene_keypoints_->points.size(), -1);
            size_t kept=0;
            for(size_t i=0; i<scene_keypoints_->points.size(); i++) {
                if( kp_has_correspondence[i] ) {
                    lut[i] = kept;
                    scene_keypoints_->points[kept] = scene_keypoints_->points[i];
                    scene_kp_normals_->points[kept] = scene_kp_normals_->points[i];
                    kept++;
                }
            }
            scene_keypoints_->points.resize(kept);
            scene_keypoints_->width = kept;
            scene_kp_normals_->points.resize(kept);
            scene_kp_normals_->width = kept;

            // adapt correspondences
            for (auto &oh : local_obj_hypotheses_) {
                for (auto &corr : oh.second.model_scene_corresp_) {
                    corr.index_match = lut[corr.index_match];
                }
            }
        }

        void
        callIndiviualRecognizer(boost::shared_ptr<Recognizer<PointT> > &rec);

        void mergeStuff(const std::vector<ObjectHypothesesGroup<PointT> > &global_hypotheses,
                         std::map<std::string, LocalObjectHypothesis<PointT> > &loh_m,
                         const pcl::PointCloud<PointT> &scene_kps,
                         const pcl::PointCloud<pcl::Normal> &scene_kp_normals);
      public:
        MultiRecognitionPipeline (const typename Recognizer<PointT>::Parameter &p = Recognizer<PointT>::Parameter()) : Recognizer<PointT>(p)
        { }

        MultiRecognitionPipeline(std::vector<std::string> &arguments);


        void
        getSavedHypotheses(std::map<std::string, LocalObjectHypothesis<PointT> > & hypotheses) const
        {
          hypotheses = local_obj_hypotheses_;
        }


        void
        getKeypointCloud(PointTPtr & keypoint_cloud) const
        {
          keypoint_cloud = scene_keypoints_;
        }

        void
        getKeyPointNormals(pcl::PointCloud<pcl::Normal>::Ptr & kp_normals) const
        {
            kp_normals = scene_kp_normals_;
        }

        bool
        initialize(bool force_retrain = false);

        /**
         * @brief updates the model database (checks if new training views are added or existing ones deleted)
         * @return
         */
        bool
        update();

        /**
         * @brief retrain the model database ( removes all trained descriptors and retrains from scratch )
         * @param model_name name of the model to retrain. If empty, all models will be retrained.
         * @return
         */
        bool
        retrain(const std::string &model_name = "");

        void
        correspondenceGrouping();

        void
        getPoseRefinement( const std::vector<ModelTPtr> &models,
                           std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transforms);

        void
        recognize();

        void
        addRecognizer(const typename Recognizer<PointT>::Ptr & rec)
        {
            recognizers_.push_back(rec);
        }

        void
        clearRecognizers()
        {
            recognizers_.clear();
            local_obj_hypotheses_.clear();
        }

        void
        setFeatAndKeypoints(const std::vector<std::vector<float> > & signatures,
                            const std::vector<int> & keypoint_indices,
                            size_t feature_type)
        {
          for (size_t i=0; i < recognizers_.size(); i++)
          {
              if(recognizers_[i]->getFeatureType() == feature_type)
              {
                  boost::shared_ptr<LocalRecognitionPipeline<PointT> > cast_local_recognizer
                          = boost::static_pointer_cast<LocalRecognitionPipeline<PointT> > (recognizers_[i]);
                  cast_local_recognizer->setFeatAndKeypoints(signatures, keypoint_indices);
              }
          }
        }

        void
        setCGAlgorithm (const typename boost::shared_ptr<GraphGeometricConsistencyGrouping<PointT, PointT> > & alg)
        {
          cg_algorithm_ = alg;
        }

        bool
        isSegmentationRequired() const
        {
            bool ret_value = false;
            for(size_t i=0; (i < recognizers_.size()) && !ret_value; i++)
                ret_value = recognizers_[i]->requiresSegmentation();

            return ret_value;
        }

        bool
        needNormals() const
        {
            for(size_t r_id=0; r_id < recognizers_.size(); r_id++)
            {
                if(recognizers_[r_id]->needNormals())
                    return true;
            }
            if(cg_algorithm_ && cg_algorithm_->getRequiresNormals())
                return true;

            if(hv_algorithm_)
                return true;

            return false;
        }

        size_t
        getFeatureType() const
        {
            size_t feat_type = 0;
            for(size_t r_id=0; r_id < recognizers_.size(); r_id++)
                feat_type += recognizers_[r_id]->getFeatureType();

            return feat_type;
        }

        typedef boost::shared_ptr< MultiRecognitionPipeline<PointT> > Ptr;
        typedef boost::shared_ptr< MultiRecognitionPipeline<PointT> const> ConstPtr;
    };
}
#endif /* MULTI_PIPELINE_RECOGNIZER_H_ */
