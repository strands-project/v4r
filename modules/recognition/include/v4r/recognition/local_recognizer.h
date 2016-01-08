/******************************************************************************
 * Copyright (c) 2012 Aitor Aldoma, Thomas Faeulhammer
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
 * local_recognizer.h
 *
 *      @date Mar 24, 2012
 *      @author Aitor Aldoma, Thomas Faeulhammer
 */

#ifndef V4R_LOCAL_RECOGNIZER_H_
#define V4R_LOCAL_RECOGNIZER_H_

#include <flann/flann.h>
#include <pcl/common/common.h>

#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/common/correspondence_grouping.h>
#include <v4r/features/local_estimator.h>
#include <v4r/recognition/hypotheses_verification.h>
#include <v4r/recognition/recognizer.h>
#include <v4r/recognition/source.h>

namespace v4r
{
    /**
     * \brief Object recognition + 6DOF pose based on local features, GC and HV
     * Contains keypoints/local features computation, matching using FLANN,
     * point-to-point correspondence grouping, pose refinement and hypotheses verification
     * Available features: SHOT, FPFH
     * See apps/3d_rec_framework/tools/apps for usage
     * \author Aitor Aldoma, Federico Tombari
     */

    template<template<class > class Distance, typename PointT, typename FeatureT>
      class V4R_EXPORTS LocalRecognitionPipeline : public Recognizer<PointT>
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

              bool use_cache_;
              int kdtree_splits_;
              int knn_;
              float distance_same_keypoint_;
              float max_descriptor_distance_;
              float correspondence_distance_constant_weight_;
              bool save_hypotheses_;

              Parameter(
                      bool use_cache = false,
                      int kdtree_splits = 512,
                      int knn = 1,
                      float distance_same_keypoint = 0.001f * 0.001f,
                      float max_descriptor_distance = std::numeric_limits<float>::infinity(),
                      float correspondence_distance_constant_weight = 1.f,
                      bool save_hypotheses = false
                      )
                  : Recognizer<PointT>::Parameter(),
                    use_cache_(use_cache),
                    kdtree_splits_ (kdtree_splits),
                    knn_ ( knn ),
                    distance_same_keypoint_ ( distance_same_keypoint ),
                    max_descriptor_distance_ ( max_descriptor_distance ),
                    correspondence_distance_constant_weight_ ( correspondence_distance_constant_weight ),
                    save_hypotheses_ ( save_hypotheses )
              {}
          }param_;

        protected:
          typedef typename pcl::PointCloud<PointT>::Ptr PointTPtr;
          typedef typename pcl::PointCloud<PointT>::ConstPtr ConstPointTPtr;
          typedef typename Recognizer<PointT>::symHyp symHyp;

          typedef Distance<float> DistT;
          typedef Model<PointT> ModelT;
          typedef boost::shared_ptr<ModelT> ModelTPtr;

          using Recognizer<PointT>::scene_;
          using Recognizer<PointT>::scene_normals_;
          using Recognizer<PointT>::models_;
          using Recognizer<PointT>::transforms_;
          using Recognizer<PointT>::hv_algorithm_;
          using Recognizer<PointT>::poseRefinement;
          using Recognizer<PointT>::hypothesisVerification;
          using Recognizer<PointT>::models_dir_;

          class flann_model
          {
          public:
            ModelTPtr model;
            std::string view_id;
            size_t keypoint_id;
            std::vector<float> descr;
          };

          /** \brief Model data source */
          typename boost::shared_ptr<Source<PointT> > source_;

          /** \brief Computes a feature */
          typename boost::shared_ptr<LocalEstimator<PointT, FeatureT> > estimator_;

          /** \brief Point-to-point correspondence grouping algorithm */
          typename boost::shared_ptr<v4r::CorrespondenceGrouping<PointT, PointT> > cg_algorithm_;

          /** \brief Descriptor name */
          std::string descr_name_;

          /** \brief Id of the model to be used */
          std::string search_model_;

          bool feat_kp_set_from_outside_;

          flann::Matrix<float> flann_data_;
          boost::shared_ptr<flann::Index<DistT> > flann_index_;

          std::map< std::pair< ModelTPtr, size_t >, std::vector<size_t> > model_view_id_to_flann_models_;
          std::vector<flann_model> flann_models_;

          typename pcl::PointCloud<FeatureT>::Ptr signatures_;
          typename pcl::PointCloud<PointT>::Ptr scene_keypoints_;
          pcl::PointIndices scene_kp_indices_;

          std::string flann_data_fn_;

          /** \brief stores keypoint correspondences */
          typename std::map<std::string, ObjectHypothesis<PointT> > obj_hypotheses_;

          //load features from disk and create flann structure
          bool loadFeaturesAndCreateFLANN();

          template <typename Type>
          inline void
          convertToFLANN (const std::vector<Type> &models, flann::Matrix<float> &data)
          {
            data.rows = models.size ();
            data.cols = models[0].descr.size (); // number of histogram bins

            float *empty_data = new float[models.size () * models[0].descr.size ()];
            flann::Matrix<float> flann_data (empty_data, models.size (), models[0].descr.size ());

            for (size_t i = 0; i < data.rows; ++i)
              for (size_t j = 0; j < data.cols; ++j)
              {
                flann_data.ptr ()[i * data.cols + j] = models[i].descr[j];
              }

            data = flann_data;
          }

          void nearestKSearch (boost::shared_ptr<flann::Index<DistT> > &index, flann::Matrix<float> & p, int k, flann::Matrix<int> &indices, flann::Matrix<float> &distances)
          {
              index->knnSearch (p, indices, distances, k, flann::SearchParams (param_.kdtree_splits_));
          }

          pcl::Normal getKpNormal (const ModelT &model, size_t keypoint_id, const std::string &view_id=0);

          PointT getKeypoint (const ModelT & model, size_t keypoint_id, const std::string &view_id=0);

          void getView (const ModelT & model, const std::string &view_id, typename pcl::PointCloud<PointT>::Ptr & view);

          void correspondenceGrouping();

          virtual void specificLoadFeaturesAndCreateFLANN()
          {
            std::cout << "specificLoadFeaturesAndCreateFLANN => this function does nothing..." << std::endl;
          }

      public:

        LocalRecognitionPipeline (const Parameter &p = Parameter()) : Recognizer<PointT>(p)
        {
          param_ = p;
          search_model_ = "";
          feat_kp_set_from_outside_ = false;
        }

        size_t getFeatureType() const
        {
            return estimator_->getFeatureType();
        }

        void setCorrespondenceDistanceConstantWeight(float w)
        {
            param_.correspondence_distance_constant_weight_ = w;
        }

        void setMaxDescriptorDistance(float d)
        {
            param_.max_descriptor_distance_ = d;
        }

        void setDistanceSameKeypoint(float d)
        {
            param_.distance_same_keypoint_ = d*d;
        }

        ~LocalRecognitionPipeline ()
        {

        }

        void
        setKnn(int k)
        {
          param_.knn_ = k;
        }

        void
        setSaveHypotheses(bool set)
        {
          param_.save_hypotheses_ = set;
        }

        bool
        getSaveHypothesesParam() const
        {
            return param_.save_hypotheses_;
        }

        virtual
        void
        getSavedHypotheses(std::map<std::string, ObjectHypothesis<PointT> > & hypotheses) const
        {
          hypotheses = obj_hypotheses_;
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

        void
        setFeatAndKeypoints(const typename pcl::PointCloud<FeatureT>::Ptr & signatures,
                                 const pcl::PointIndices & keypoint_indices)
        {  
          if(!signatures || signatures->points.size()==0 ||
                  (signatures->points.size()!=keypoint_indices.indices.size()))
              throw std::runtime_error("Provided signatures and keypoints are not valid!");

          feat_kp_set_from_outside_ = true;
          scene_kp_indices_ = keypoint_indices;
          signatures_ = signatures;
        }


        void
        setSearchModel (const std::string & id)
        {
          search_model_ = id;
        }

        void
        setKdtreeSplits (int n)
        {
          param_.kdtree_splits_ = n;
        }

        void
        setUseCache (bool u)
        {
          param_.use_cache_ = u;
        }

        /**
         * \brief Sets the model data source_
         */
        void
        setDataSource (const typename boost::shared_ptr<Source<PointT> > & source)
        {
          source_ = source;
        }

        typename boost::shared_ptr<Source<PointT> >
        getDataSource () const
        {
          return source_;
        }

        /**
         * \brief Sets the local feature estimator
         */
        void
        setFeatureEstimator (const typename boost::shared_ptr<LocalEstimator<PointT, FeatureT> > & feat)
        {
          estimator_ = feat;
        }

        /**
         * \brief Sets the CG algorithm
         */
        void
        setCGAlgorithm (const typename boost::shared_ptr<v4r::CorrespondenceGrouping<PointT, PointT> > & alg)
        {
          cg_algorithm_ = alg;
        }

        /**
         * \brief Initializes the FLANN structure from the provided source
         * It does training for the models that havent been trained yet
         */
        bool
        initialize(bool force_retrain = false);

        void
        reinitialize()
        {
            reinitializeSourceOnly();
            reinitializeRecOnly();
        }

        void
        reinitializeSourceOnly()
        {
            flann_models_.clear();
            source_->generate();
        }

        void
        reinitializeRecOnly()
        {
            initialize(false);
        }

        /**
         * @brief Visualizes all found correspondences between scene and model
         * @param object model to be visualized
         */
        void
        drawCorrespondences (const ObjectHypothesis<PointT> & oh)
        {
          oh.visualize(*scene_);
        }

        /**
         * @brief Visualizes all found correspondences between scene and models
         */
        void
        drawCorrespondences()
        {
            typename std::map<std::string, ObjectHypothesis<PointT> >::iterator it;
            for (it = obj_hypotheses_.begin(); it != obj_hypotheses_.end (); it++) {
                it->second.visualize(*scene_);
            }
        }

        /**
         * \brief Performs recognition and pose estimation on the input cloud
         */
        void
        recognize ();
      };
}

#endif /* REC_FRAMEWORK_LOCAL_RECOGNIZER_H_ */
