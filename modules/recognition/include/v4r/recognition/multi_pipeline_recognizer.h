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
    template<typename PointInT>
    class V4R_EXPORTS MultiRecognitionPipeline : public Recognizer<PointInT>
    {
      protected:
        std::vector<typename boost::shared_ptr<v4r::Recognizer<PointInT> > > recognizers_;

      private:
        using Recognizer<PointInT>::input_;
        using Recognizer<PointInT>::models_;
        using Recognizer<PointInT>::transforms_;
        using Recognizer<PointInT>::ICP_iterations_;
        using Recognizer<PointInT>::icp_type_;
        using Recognizer<PointInT>::VOXEL_SIZE_ICP_;
        using Recognizer<PointInT>::indices_;
        using Recognizer<PointInT>::hv_algorithm_;
        using Recognizer<PointInT>::setSceneNormals;

        using Recognizer<PointInT>::poseRefinement;
        using Recognizer<PointInT>::hypothesisVerification;
        using Recognizer<PointInT>::icp_scene_indices_;

        typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;
        typedef typename pcl::PointCloud<PointInT>::ConstPtr ConstPointInTPtr;

        typedef Model<PointInT> ModelT;
        typedef boost::shared_ptr<ModelT> ModelTPtr;
        std::vector<pcl::PointIndices> segmentation_indices_;

        typename boost::shared_ptr<v4r::GraphGeometricConsistencyGrouping<PointInT, PointInT> > cg_algorithm_;
        pcl::PointCloud<pcl::Normal>::Ptr scene_normals_;
        bool normals_set_;

        bool multi_object_correspondence_grouping_;

        typename std::map<std::string, ObjectHypothesis<PointInT> > saved_object_hypotheses_;
        boost::shared_ptr<std::map<std::string, ObjectHypothesis<PointInT> > > pObjectHypotheses_;
        typename pcl::PointCloud<PointInT>::Ptr keypoints_cloud_;

        pcl::PointIndices keypoint_indices_;

        bool set_save_hypotheses_;

      public:
        MultiRecognitionPipeline () : Recognizer<PointInT>()
        {
            normals_set_ = false;
            multi_object_correspondence_grouping_ = false;
            set_save_hypotheses_ = false;
        }

        void setMultiObjectCG(bool b)
        {
            multi_object_correspondence_grouping_ = b;
        }

        void setSaveHypotheses(bool set_save_hypotheses)
        {
            set_save_hypotheses_ = set_save_hypotheses;
        }

        void
        getSavedHypotheses(std::map<std::string, ObjectHypothesis<PointInT> > & hypotheses) const
        {
          hypotheses = *pObjectHypotheses_;
        }

        void
        getKeypointCloud(PointInTPtr & keypoint_cloud) const
        {
          keypoint_cloud = keypoints_cloud_;
        }


        void
        getKeypointIndices(pcl::PointIndices & indices) const
        {
            indices.header = keypoint_indices_.header;
            indices.indices = keypoint_indices_.indices;
        }

        void initialize();

        void reinitialize();

        void reinitialize(const std::vector<std::string> & load_ids);

        void correspondenceGrouping();

        void getPoseRefinement(
                boost::shared_ptr<std::vector<ModelTPtr> > models,
                boost::shared_ptr<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > transforms);

        void recognize();

        void addRecognizer(const typename boost::shared_ptr<v4r::Recognizer<PointInT> > & rec)
        {
          recognizers_.push_back(rec);
        }

        template <template<class > class Distance, typename FeatureT>
        void setISPK(const typename pcl::PointCloud<FeatureT>::Ptr & signatures,
                     const PointInTPtr & p,
                     const pcl::PointIndices & keypoint_indices,
                     size_t feature_type)
        {
          for (size_t i=0; i < recognizers_.size(); i++)
          {
              std::cout << "Checking recognizer type: " << recognizers_[i]->getFeatureType() << std::endl;
              if(recognizers_[i]->getFeatureType() == feature_type)
              {
                  typename boost::shared_ptr<v4r::LocalRecognitionPipeline<Distance, PointInT, FeatureT> > cast_local_recognizer;
                  cast_local_recognizer = boost::static_pointer_cast<v4r::LocalRecognitionPipeline<Distance, PointInT, FeatureT> > (recognizers_[i]);
                  cast_local_recognizer->setISPK(signatures, p, keypoint_indices);
              }
          }
        }

        void
        setCGAlgorithm (const typename boost::shared_ptr<v4r::GraphGeometricConsistencyGrouping<PointInT, PointInT> > & alg)
        {
          cg_algorithm_ = alg;
        }

        bool isSegmentationRequired() const;

        typename boost::shared_ptr<Source<PointInT> >
        getDataSource () const;

        void
        setSegmentation(const std::vector<pcl::PointIndices> & ind)
        {
          segmentation_indices_ = ind;
        }

        void setSceneNormals(const pcl::PointCloud<pcl::Normal>::Ptr &normals)
        {
            scene_normals_ = normals;
            normals_set_ = true;
        }

        void clear()
        {
            recognizers_.clear();
            saved_object_hypotheses_.clear();
            segmentation_indices_.clear();
        }
    };
}
#endif /* MULTI_PIPELINE_RECOGNIZER_H_ */
