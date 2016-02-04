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

#include "recognizer.h"
#include "local_recognizer.h"
#include <v4r/common/graph_geometric_consistency.h>
#include <omp.h>

namespace v4r
{
    template<typename PointT>
    class V4R_EXPORTS MultiRecognitionPipeline : public Recognizer<PointT>
    {
    public:
        class V4R_EXPORTS Parameter : public Recognizer<PointT>::Parameter
        {
        public:
            using Recognizer<PointT>::Parameter::voxel_size_icp_;
            using Recognizer<PointT>::Parameter::normal_computation_method_;
            using Recognizer<PointT>::Parameter::merge_close_hypotheses_;
            using Recognizer<PointT>::Parameter::merge_close_hypotheses_dist_;
            using Recognizer<PointT>::Parameter::merge_close_hypotheses_angle_;

            bool save_hypotheses_;

            Parameter(
                    bool save_hypotheses = false
                    )
                : Recognizer<PointT>::Parameter(),
                  save_hypotheses_ ( save_hypotheses )
            {}
        }param_;

      protected:
        std::vector<typename boost::shared_ptr<Recognizer<PointT> > > recognizers_;

      private:
        using Recognizer<PointT>::scene_;
        using Recognizer<PointT>::scene_normals_;
        using Recognizer<PointT>::models_;
        using Recognizer<PointT>::transforms_;
        using Recognizer<PointT>::hv_algorithm_;
        using Recognizer<PointT>::hypothesisVerification;

        typedef typename pcl::PointCloud<PointT>::Ptr PointTPtr;
        typedef typename pcl::PointCloud<PointT>::ConstPtr ConstPointTPtr;

        typedef Model<PointT> ModelT;
        typedef boost::shared_ptr<ModelT> ModelTPtr;
        std::vector<pcl::PointIndices> segmentation_indices_;
        omp_lock_t rec_lock_;

        typename boost::shared_ptr<GraphGeometricConsistencyGrouping<PointT, PointT> > cg_algorithm_;  /// @brief algorithm for correspondence grouping
        typename pcl::PointCloud<PointT>::Ptr scene_keypoints_; /// @brief keypoints of the scene
        pcl::PointCloud<pcl::Normal>::Ptr scene_kp_normals_;
        std::map<std::string, ObjectHypothesis<PointT> > obj_hypotheses_;   /// @brief stores feature correspondences

        /**
         * @brief removes all scene keypoints not having a correspondence in the model database and adapt correspondences indices accordingly
         */
        void
        compress()
        {
            std::vector<bool> kp_has_correspondence(scene_keypoints_->points.size(), false);
            for (const auto &oh : obj_hypotheses_) {
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
            for (auto &oh : obj_hypotheses_) {
                for (auto &corr : oh.second.model_scene_corresp_) {
                    corr.index_match = lut[corr.index_match];
                }
            }
        }

        void
        callIndiviualRecognizer(Recognizer<PointT> &rec);

        void mergeStuff(std::map<std::string, ObjectHypothesis<PointT> > &oh_m,
                         const std::vector<ModelTPtr> &models,
                         const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transforms,
                         const pcl::PointCloud<PointT> &scene_kps,
                         const pcl::PointCloud<pcl::Normal> &scene_kp_normals);
      public:
        MultiRecognitionPipeline (const Parameter &p = Parameter()) : Recognizer<PointT>(p)
        {
            param_ = p;
        }

        MultiRecognitionPipeline(int argc, char **argv);

        void setSaveHypotheses(bool set_save_hypotheses)
        {
            param_.save_hypotheses_ = set_save_hypotheses;
        }

        void
        getSavedHypotheses(std::map<std::string, ObjectHypothesis<PointT> > & hypotheses) const
        {
          hypotheses = obj_hypotheses_;
        }

        bool
        getSaveHypothesesParam() const
        {
            return param_.save_hypotheses_;
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

        void
        correspondenceGrouping();

        void
        getPoseRefinement( const std::vector<ModelTPtr> &models,
                           std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transforms);

        void recognize();

        void addRecognizer(typename boost::shared_ptr<Recognizer<PointT> > & rec)
        {
          recognizers_.push_back(rec);
        }

        void clearRecognizers()
        {
            recognizers_.clear();
            obj_hypotheses_.clear();
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

        bool isSegmentationRequired() const;

        typename boost::shared_ptr<Source<PointT> >
        getDataSource () const;

        void
        setSegmentation(const std::vector<pcl::PointIndices> & ind)
        {
          segmentation_indices_ = ind;
        }

    };
}
#endif /* MULTI_PIPELINE_RECOGNIZER_H_ */
