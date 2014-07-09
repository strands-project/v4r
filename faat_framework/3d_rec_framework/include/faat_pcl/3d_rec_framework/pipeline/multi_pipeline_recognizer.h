/*
 * multi_pipeline_recognizer.h
 *
 *  Created on: Feb 24, 2013
 *      Author: aitor
 */

#ifndef MULTI_PIPELINE_RECOGNIZER_H_
#define MULTI_PIPELINE_RECOGNIZER_H_

#include <faat_pcl/3d_rec_framework/defines/faat_3d_rec_framework_defines.h>
#include <faat_pcl/3d_rec_framework/pipeline/recognizer.h>
#include <faat_pcl/recognition/cg/graph_geometric_consistency.h>

namespace faat_pcl
{
  namespace rec_3d_framework
  {
    template<typename PointInT>
    class FAAT_3D_FRAMEWORK_API MultiRecognitionPipeline : public Recognizer<PointInT>
    {
      protected:
        std::vector<typename boost::shared_ptr<faat_pcl::rec_3d_framework::Recognizer<PointInT> > > recognizers_;

      private:
        using Recognizer<PointInT>::input_;
        using Recognizer<PointInT>::models_;
        using Recognizer<PointInT>::transforms_;
        using Recognizer<PointInT>::ICP_iterations_;
        using Recognizer<PointInT>::icp_type_;
        using Recognizer<PointInT>::VOXEL_SIZE_ICP_;
        using Recognizer<PointInT>::indices_;
        using Recognizer<PointInT>::hv_algorithm_;

        using Recognizer<PointInT>::poseRefinement;
        using Recognizer<PointInT>::hypothesisVerification;
        using Recognizer<PointInT>::icp_scene_indices_;

        typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;
        typedef typename pcl::PointCloud<PointInT>::ConstPtr ConstPointInTPtr;

        typedef Model<PointInT> ModelT;
        typedef boost::shared_ptr<ModelT> ModelTPtr;
        std::vector<pcl::PointIndices> segmentation_indices_;

        typename boost::shared_ptr<faat_pcl::GraphGeometricConsistencyGrouping<PointInT, PointInT> > cg_algorithm_;
        pcl::PointCloud<pcl::Normal>::Ptr scene_normals_;
        bool normals_set_;

        bool multi_object_correspondence_grouping_;

      public:
        MultiRecognitionPipeline () : Recognizer<PointInT>()
        {
            normals_set_ = false;
            multi_object_correspondence_grouping_ = false;
        }

        void setMultiObjectCG(bool b)
        {
            multi_object_correspondence_grouping_ = b;
        }

        void initialize();

        void recognize();

        void addRecognizer(typename boost::shared_ptr<faat_pcl::rec_3d_framework::Recognizer<PointInT> > & rec)
        {
          recognizers_.push_back(rec);
        }

        void
        setCGAlgorithm (typename boost::shared_ptr<faat_pcl::GraphGeometricConsistencyGrouping<PointInT, PointInT> > & alg)
        {
          cg_algorithm_ = alg;
        }

        bool isSegmentationRequired();

        typename boost::shared_ptr<Source<PointInT> >
        getDataSource ();

        void
        setSegmentation(std::vector<pcl::PointIndices> & ind)
        {
          segmentation_indices_ = ind;
        }

        void setSceneNormals(pcl::PointCloud<pcl::Normal>::Ptr & normals)
        {
            scene_normals_ = normals;
            normals_set_ = true;
        }
    };
  }
}
#endif /* MULTI_PIPELINE_RECOGNIZER_H_ */
