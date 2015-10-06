/*
 * local_recognizer.h
 *
 *      Created on: Mar 24, 2012
 *      Author: Aitor Aldoma
 *      Maintainer: Thomas Faeulhammer
 */

#ifndef FAAT_PCL_REC_FRAMEWORK_LOCAL_RECOGNIZER_H_
#define FAAT_PCL_REC_FRAMEWORK_LOCAL_RECOGNIZER_H_

#include <flann/flann.h>
#include <pcl/common/common.h>
#include "source.h"
#include <v4r/features/local_estimator.h>
#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/common/correspondence_grouping.h>
#include <v4r/recognition//hypotheses_verification.h>
#include "recognizer.h"

inline bool
correspSorter (const pcl::Correspondence & i, const pcl::Correspondence & j)
{
  return (i.distance < j.distance);
}

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
        protected:
          typedef typename pcl::PointCloud<PointT>::Ptr PointTPtr;
          typedef typename pcl::PointCloud<PointT>::ConstPtr ConstPointTPtr;

          typedef Distance<float> DistT;
          typedef Model<PointT> ModelT;
          typedef boost::shared_ptr<ModelT> ModelTPtr;

          using Recognizer<PointT>::scene_;
          using Recognizer<PointT>::models_;
          using Recognizer<PointT>::transforms_;
          using Recognizer<PointT>::ICP_iterations_;
          using Recognizer<PointT>::icp_type_;
          using Recognizer<PointT>::VOXEL_SIZE_ICP_;
          using Recognizer<PointT>::indices_;
          using Recognizer<PointT>::hv_algorithm_;
          using Recognizer<PointT>::poseRefinement;
          using Recognizer<PointT>::hypothesisVerification;
          using Recognizer<PointT>::training_dir_;

          class flann_model
          {
          public:
            ModelTPtr model;
            size_t keypoint_id;
            std::vector<float> descr;
            size_t view_id;
          };


          /** \brief Model data source */
          typename boost::shared_ptr<Source<PointT> > source_;

          /** \brief Computes a feature */
          typename boost::shared_ptr<LocalEstimator<PointT, FeatureT> > estimator_;

          /** \brief Point-to-point correspondence grouping algorithm */
          typename boost::shared_ptr<v4r::CorrespondenceGrouping<PointT, PointT> > cg_algorithm_;

          /** \brief Descriptor name */
          std::string descr_name_;

          /** \brief defines number of leading zeros in view filenames (e.g. cloud_00001.pcd -> length_ = 5) */
          size_t view_id_length_;

          /** \brief Id of the model to be used */
          std::string search_model_;

          bool feat_kp_set_from_outside_;

          flann::Matrix<float> flann_data_;
          boost::shared_ptr<flann::Index<DistT> > flann_index_;

          std::map< std::pair< ModelTPtr, size_t >, std::vector<size_t> > model_view_id_to_flann_models_;
          std::vector<flann_model> flann_models_;

          bool use_cache_;

          float threshold_accept_model_hypothesis_;
          int kdtree_splits_;

          typename pcl::PointCloud<FeatureT>::Ptr signatures_;
          typename pcl::PointCloud<PointT>::Ptr scene_keypoints_;
          pcl::PointIndices scene_kp_indices_;

          std::string flann_data_fn_;

          int normal_computation_method_;

          bool save_hypotheses_;
          typename std::map<std::string, ObjectHypothesis<PointT> > obj_hypotheses_;
          int knn_;
          float distance_same_keypoint_;
          float max_descriptor_distance_;
          float correspondence_distance_constant_weight_;

          //load features from disk and create flann structure
          void loadFeaturesAndCreateFLANN ();

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

          void nearestKSearch (boost::shared_ptr<flann::Index<DistT> > &index, flann::Matrix<float> & p, int k, flann::Matrix<int> &indices, flann::Matrix<float> &distances);

          pcl::Normal getKpNormal (const ModelT &model, size_t keypoint_id, size_t view_id=0);

          PointT getKeypoint (const ModelT & model, size_t keypoint_id, size_t view_id=0);

          void getView (const ModelT & model, size_t view_id, typename pcl::PointCloud<PointT>::Ptr & view);


          virtual void specificLoadFeaturesAndCreateFLANN()
          {
            std::cout << "specificLoadFeaturesAndCreateFLANN => this function does nothing..." << std::endl;
          }

          virtual void prepareSpecificCG(PointTPtr & scene_cloud, PointTPtr & scene_keypoints)
          {
                (void)scene_cloud;
                (void)scene_keypoints;
                std::cerr << "This is a virtual function doing nothing!" << std::endl;
          }

          virtual void specificCG(PointTPtr & scene_cloud, PointTPtr & scene_keypoints, ObjectHypothesis<PointT> & oh)
          {
              (void)scene_cloud;
              (void)scene_keypoints;
              (void)oh;
              std::cerr << "This is a virtual function doing nothing!" << std::endl;
          }

          virtual void clearSpecificCG() {

          }

      public:

        LocalRecognitionPipeline () : Recognizer<PointT>()
        {
          use_cache_ = false;
          threshold_accept_model_hypothesis_ = 0.2f;
          kdtree_splits_ = 512;
          search_model_ = "";
          save_hypotheses_ = false;
          knn_ = 1;
          distance_same_keypoint_ = 0.001f * 0.001f;
          max_descriptor_distance_ = std::numeric_limits<float>::infinity();
          correspondence_distance_constant_weight_ = 1.f;
          normal_computation_method_ = 1;
          feat_kp_set_from_outside_ = false;
        }

        size_t getFeatureType() const
        {
            return estimator_->getFeatureType();
        }

        void setCorrespondenceDistanceConstantWeight(float w)
        {
            correspondence_distance_constant_weight_ = w;
        }

        void setMaxDescriptorDistance(float d)
        {
            max_descriptor_distance_ = d;
        }

        void setDistanceSameKeypoint(float d)
        {
            distance_same_keypoint_ = d*d;
        }

        ~LocalRecognitionPipeline ()
        {

        }

        void
        setKnn(int k)
        {
          knn_ = k;
        }

        void
        setSaveHypotheses(bool set)
        {
          save_hypotheses_ = set;
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
        setThresholdAcceptHyp (float t)
        {
          threshold_accept_model_hypothesis_ = t;
        }

        void
        setKdtreeSplits (int n)
        {
          kdtree_splits_ = n;
        }

        void
        setIndices (const std::vector<int> & indices)
        {
          indices_ = indices;
        }

        void
        setUseCache (bool u)
        {
          use_cache_ = u;
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
         * \brief Sets the descriptor name
         */
        void
        setDescriptorName (const std::string & name)
        {
          descr_name_ = name;
        }


        /**
         * \brief Initializes the FLANN structure from the provided source
         * It does training for the models that havent been trained yet
         */
        void
        initialize (bool force_retrain = false);

        void
        reinitialize(const std::vector<std::string> & load_ids = std::vector<std::string>());

        /**
         * @brief Visualizes all found correspondences between scene and model
         * @param object model to be visualized
         */
        void
        drawCorrespondences (const ObjectHypothesis<PointT> & oh)
        {
          oh.visualize();
        }

        /**
         * @brief Visualizes all found correspondences between scene and models
         */
        void
        drawCorrespondences()
        {
            typename std::map<std::string, ObjectHypothesis<PointT> >::iterator it;
            for (it = obj_hypotheses_.begin(); it != obj_hypotheses_.end (); it++) {
                it->second.visualize();
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
