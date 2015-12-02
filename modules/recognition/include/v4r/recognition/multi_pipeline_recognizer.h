/*
 * multi_pipeline_recognizer.h
 *
 *  Created on: Feb 24, 2013
 *      Author: aitor
 */

#ifndef MULTI_PIPELINE_RECOGNIZER_H_
#define MULTI_PIPELINE_RECOGNIZER_H_

#include <v4r/common/faat_3d_rec_framework_defines.h>
#include "recognizer.h"
#include "local_recognizer.h"
#include <v4r/common/graph_geometric_consistency.h>

namespace v4r
{
    template<typename PointT>
    class V4R_EXPORTS MultiRecognitionPipeline : public Recognizer<PointT>
    {
    public:
        class V4R_EXPORTS Parameter : public Recognizer<PointT>::Parameter
        {
        public:
            using Recognizer<PointT>::Parameter::icp_iterations_;
            using Recognizer<PointT>::Parameter::icp_type_;
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
        std::vector<typename boost::shared_ptr<v4r::Recognizer<PointT> > > recognizers_;

      private:
        using Recognizer<PointT>::scene_;
        using Recognizer<PointT>::scene_normals_;
        using Recognizer<PointT>::models_;
        using Recognizer<PointT>::transforms_;
        using Recognizer<PointT>::hv_algorithm_;

        using Recognizer<PointT>::poseRefinement;
        using Recognizer<PointT>::hypothesisVerification;

        typedef typename pcl::PointCloud<PointT>::Ptr PointTPtr;
        typedef typename pcl::PointCloud<PointT>::ConstPtr ConstPointTPtr;

        typedef Model<PointT> ModelT;
        typedef boost::shared_ptr<ModelT> ModelTPtr;
        std::vector<pcl::PointIndices> segmentation_indices_;

        typename boost::shared_ptr<v4r::GraphGeometricConsistencyGrouping<PointT, PointT> > cg_algorithm_;
        typename pcl::PointCloud<PointT>::Ptr scene_keypoints_;
        pcl::PointIndices scene_kp_indices_;

        std::map<std::string, ObjectHypothesis<PointT> > saved_object_hypotheses_;
        std::map<std::string, ObjectHypothesis<PointT> > obj_hypotheses_;


      public:
        MultiRecognitionPipeline (const Parameter &p = Parameter()) : Recognizer<PointT>(p)
        {
            param_ = p;
        }

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
        getKeypointIndices(pcl::PointIndices & indices) const
        {
            indices = scene_kp_indices_;
        }

        void initialize();

        void reinitialize();

        void reinitialize(const std::vector<std::string> & load_ids);

        void correspondenceGrouping();

        void getPoseRefinement(const std::vector<ModelTPtr> &models,
                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transforms);

        void recognize();

        void addRecognizer(typename boost::shared_ptr<v4r::Recognizer<PointT> > & rec)
        {
          recognizers_.push_back(rec);
        }

        void clearRecognizers()
        {
            recognizers_.clear();
        }

        template <typename FeatureT>
        void
        setFeatAndKeypoints(const typename pcl::PointCloud<FeatureT>::Ptr & signatures,
                            const pcl::PointIndices & keypoint_indices,
                            size_t feature_type)
        {
          for (size_t i=0; i < recognizers_.size(); i++)
          {
              if(recognizers_[i]->getFeatureType() == feature_type)
              {
                  boost::shared_ptr<LocalRecognitionPipeline<flann::L1, PointT, FeatureT> > cast_local_recognizer
                          = boost::static_pointer_cast<LocalRecognitionPipeline<flann::L1, PointT, FeatureT> > (recognizers_[i]);
                  cast_local_recognizer->setFeatAndKeypoints(signatures, keypoint_indices);
              }
          }
        }

        void
        setCGAlgorithm (const typename boost::shared_ptr<v4r::GraphGeometricConsistencyGrouping<PointT, PointT> > & alg)
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
