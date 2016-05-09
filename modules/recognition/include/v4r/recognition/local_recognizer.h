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
#include <v4r/features/local_estimator.h>
#include <v4r/keypoints/keypoint_extractor.h>
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

    template<typename PointT>
      class V4R_EXPORTS LocalRecognitionPipeline : public Recognizer<PointT>
      {
      public:
          class V4R_EXPORTS Parameter : public Recognizer<PointT>::Parameter
          {
          public:
              using Recognizer<PointT>::Parameter::voxel_size_icp_;
              using Recognizer<PointT>::Parameter::max_corr_distance_;
              using Recognizer<PointT>::Parameter::max_distance_;
              using Recognizer<PointT>::Parameter::normal_computation_method_;
              using Recognizer<PointT>::Parameter::merge_close_hypotheses_;
              using Recognizer<PointT>::Parameter::merge_close_hypotheses_dist_;
              using Recognizer<PointT>::Parameter::merge_close_hypotheses_angle_;

              int kdtree_splits_;
              size_t knn_;  /// @brief nearest neighbors to search for when checking feature descriptions of the scene
              float distance_same_keypoint_;
              float max_descriptor_distance_;
              float correspondence_distance_weight_; /// @brief weight factor for correspondences distances. This is done to favour correspondences from different pipelines that are more reliable than other (SIFT and SHOT corr. simultaneously fed into CG)
              int distance_metric_; /// @brief defines the norm used for feature matching (1... L1 norm, 2... L2 norm)
              bool use_3d_model_; /// @brief if true, it learns features directly from the reconstructed 3D model instead of on the individual training views

              bool use_codebook_;   /// @brief if true, performs K-Means clustering on all signatures being trained.
              size_t codebook_size_; /// @brief number of clusters being computed for the codebook (K-Means)
              float codebook_filter_ratio_; /// @brief signatures clustered into a cluster which occures more often (w.r.t the total number of signatures) than this threshold, will be rejected.
              float kernel_sigma_;

              bool filter_planar_; /// @brief Filter keypoints with a planar surface
              bool filter_points_above_plane_; /// @brief Filter only points above table plane
              int min_plane_size_; /// @brief Minimum number of points for a plane to be checked if filter only points above table plane
              int planar_computation_method_; /// @brief defines the method used to check for planar points. 0... based on curvate value after normalestimationomp, 1... with eigenvalue check of scatter matrix
              float planar_support_radius_; /// @brief Radius used to check keypoints for planarity.
              float threshold_planar_; /// @brief threshold ratio used for deciding if patch is planar. Ratio defined as largest eigenvalue to all others.

              bool filter_border_pts_; /// @brief Filter keypoints at the boundary
              int boundary_width_; /// @brief Width in pixel of the depth discontinuity

              bool adaptative_MLS_;

              bool visualize_keypoints_;    /// @brief if true, visualizes the extracted keypoints

              Parameter(
                      int kdtree_splits = 512,
                      size_t knn = 1,
                      float distance_same_keypoint = 0.001f * 0.001f,
                      float max_descriptor_distance = std::numeric_limits<float>::infinity(),
                      float correspondence_distance_weight = 1.f,
                      int distance_metric = 1,
                      bool use_3d_model = false,
                      bool use_codebook = false,
                      size_t codebook_size = 50,
                      float codebook_filter_ratio = 0.3f,
                      float kernel_sigma = 1.f,
                      bool filter_planar = false,
                      bool filter_points_above_plane = false,
                      int min_plane_size = 1000,
                      float planar_support_radius = 0.04f,
                      float threshold_planar = 0.02f,
                      bool filter_border_pts = false,
                      int boundary_width = 5,
                      bool adaptive_MLS = false,
                      bool visualize_keypoints = false
                      )
                  : Recognizer<PointT>::Parameter(),
                    kdtree_splits_ (kdtree_splits),
                    knn_ ( knn ),
                    distance_same_keypoint_ ( distance_same_keypoint ),
                    max_descriptor_distance_ ( max_descriptor_distance ),
                    correspondence_distance_weight_ ( correspondence_distance_weight ),
                    distance_metric_ (distance_metric),
                    use_3d_model_ (use_3d_model),
                    use_codebook_ (use_codebook),
                    codebook_size_ (codebook_size),
                    codebook_filter_ratio_ (codebook_filter_ratio),
                    kernel_sigma_ (kernel_sigma),
                    filter_planar_ (filter_planar),
                    filter_points_above_plane_ ( filter_points_above_plane ),
                    min_plane_size_ (min_plane_size),
                    planar_support_radius_ (planar_support_radius),
                    threshold_planar_ (threshold_planar),
                    filter_border_pts_ (filter_border_pts),
                    boundary_width_ (boundary_width),
                    adaptative_MLS_ (adaptive_MLS),
                    visualize_keypoints_ (visualize_keypoints)
              {}
          }param_;

      private:
          void extractKeypoints (); /// @brief extracts keypoints from the scene
          void featureMatching (); /// @brief matches all scene keypoints with model signatures
          void featureEncoding (); /// @brief describes each keypoint with corresponding feature descriptor
//          bool checkFLANN () const;
//          void computeCodebook (); /// @brief computes a dictionary based on all model signatures
//          int getCodebookLabel (const Eigen::VectorXf &signature) const; /// @brief returns label of codebook that is closest to given signature
          void filterKeypoints (bool filter_signatures = false); /// @brief filters keypoints based on planarity and closeness to depth discontinuity (if according parameters are set)
//          void filterKeypointsWithCodebook (); /// @brief filters scene keypoints based on uniqueness of the signatures given by the codebook
//          void createFLANN();   /// @brief create the FLANN index used for matching signatures
          void loadFeaturesFromDisk(); /// @brief load features from disk
//          void computeFeatureProbabilities();
//          float computePriorProbability(const Eigen::VectorXf &query_sig);
//          void visualizeModelProbabilities() const;
//          void filterModelKeypointsBasedOnPriorProbability();

          float max_prior_prob_;
          bool initialization_phase_;

        protected:
          typedef typename pcl::PointCloud<PointT>::Ptr PointTPtr;
          typedef typename pcl::PointCloud<PointT>::ConstPtr ConstPointTPtr;
          typedef typename Recognizer<PointT>::symHyp symHyp;

          typedef Model<PointT> ModelT;
          typedef boost::shared_ptr<ModelT> ModelTPtr;

          using Recognizer<PointT>::scene_;
          using Recognizer<PointT>::scene_normals_;
          using Recognizer<PointT>::indices_;
          using Recognizer<PointT>::source_;
          using Recognizer<PointT>::hv_algorithm_;
          using Recognizer<PointT>::hypothesisVerification;
          using Recognizer<PointT>::models_dir_;


          class flann_model
          {
          public:
            ModelTPtr model;
            std::string view_id;
            size_t keypoint_id;
          };

          /** \brief Computes a feature */
          typename boost::shared_ptr<LocalEstimator<PointT> > estimator_;

          /** \brief Descriptor name */
          std::string descr_name_;

          bool feat_kp_set_from_outside_;

          boost::shared_ptr<flann::Index<flann::L1<float> > > flann_index_l1_;
          boost::shared_ptr<flann::Index<flann::L2<float> > > flann_index_l2_;
          std::vector<flann_model> flann_models_;
          boost::shared_ptr<flann::Matrix<float> > flann_data_;

          std::vector<std::vector<float> > scene_signatures_;
          std::vector<int> keypoint_indices_;
          std::vector<int> keypoint_indices_unfiltered_;    ///@brief only for visualization

          /** \brief stores keypoint correspondences */
          typename std::map<std::string, LocalObjectHypothesis<PointT> > local_obj_hypotheses_;

          void visualizeKeypoints() const;
          mutable pcl::visualization::PCLVisualizer::Ptr vis_;

          bool computeFeatures();

          std::vector<typename KeypointExtractor<PointT>::Ptr > keypoint_extractor_;
      public:

        LocalRecognitionPipeline (const Parameter &p = Parameter()) : Recognizer<PointT>(p)
        {
            param_ = p;
            feat_kp_set_from_outside_ = false;
            initialization_phase_ = false;
        }

        size_t
        getFeatureType() const
        {
            return estimator_->getFeatureType();
        }

        std::string
        getFeatureName() const
        {
            return estimator_->getFeatureDescriptorName();
        }

        void
        getSavedHypotheses(std::map<std::string, LocalObjectHypothesis<PointT> > &oh) const
        {
          oh = local_obj_hypotheses_;
        }

        typename pcl::PointCloud<PointT>::Ptr
        getKeypointCloud() const
        {
            typename pcl::PointCloud<PointT>::Ptr scene_keypoints (new pcl::PointCloud<PointT>);
            pcl::copyPointCloud(*scene_, keypoint_indices_, *scene_keypoints);
            return scene_keypoints;
        }

        void
        getKeypointIndices(std::vector<int> &indices) const
        {
            indices = keypoint_indices_;
        }

        void
        setFeatAndKeypoints(const std::vector<std::vector<float> > & signatures,
                                 const std::vector<int> & keypoint_indices)
        {  
          if( signatures.empty() ||
                  (signatures.size()!=keypoint_indices.size()))
              throw std::runtime_error("Provided signatures and keypoints are not valid!");

          feat_kp_set_from_outside_ = true;
          keypoint_indices_ = keypoint_indices;
          scene_signatures_ = signatures;
        }


        /**
         * \brief Sets the model data source_
         */
        void
        setDataSource (const typename Source<PointT>::Ptr & source)
        {
          source_ = source;
        }

        typename Source<PointT>::Ptr
        getDataSource () const
        {
          return source_;
        }

        /**
         * \brief Sets the local feature estimator
         */
        void
        setFeatureEstimator (const typename LocalEstimator<PointT>::Ptr & feat)
        {
            estimator_ = feat;
        }

        /**
         * \brief Initializes the FLANN structure from the provided source
         * It does training for the models that havent been trained yet
         */
        bool
        initialize(bool force_retrain = false);

        /**
         * @brief Visualizes all found correspondences between scene and model
         * @param object model to be visualized
         */
        void
        drawCorrespondences (const LocalObjectHypothesis<PointT> & oh)
        {
            pcl::PointCloud<PointT> scene_keypoints;
            pcl::copyPointCloud(*scene_, keypoint_indices_, scene_keypoints);
            oh.visualize(*scene_, scene_keypoints);
        }

        /**
         * @brief adds a keypoint extractor
         * @param keypoint extractor object
         */
        void
        addKeypointExtractor (boost::shared_ptr<KeypointExtractor<PointT> > & ke)
        {
            keypoint_extractor_.push_back (ke);
        }

        /**
         * @brief Visualizes all found correspondences between scene and models
         */
        void
        drawCorrespondences()
        {
            typename std::map<std::string, LocalObjectHypothesis<PointT> >::iterator it;
            for (it = local_obj_hypotheses_.begin(); it != local_obj_hypotheses_.end (); it++) {
                pcl::PointCloud<PointT> scene_keypoints;
                pcl::copyPointCloud(*scene_, keypoint_indices_, scene_keypoints);
                it->second.visualize(*scene_, scene_keypoints);
            }
        }

        virtual bool
        needNormals() const
        {
            if (estimator_ && estimator_->needNormals())
                return true;

            if (!keypoint_extractor_.empty())
            {
                for (size_t i=0; i<keypoint_extractor_.size(); i++)
                {
                    if ( keypoint_extractor_[i]->needNormals() )
                        return true;
                }
            }
            return false;
        }

        /**
         * \brief Performs recognition and pose estimation on the input cloud
         */
        void
        recognize ();


        typedef boost::shared_ptr< LocalRecognitionPipeline<PointT> > Ptr;
        typedef boost::shared_ptr< LocalRecognitionPipeline<PointT> const> ConstPtr;
      };
}

#endif /* REC_FRAMEWORK_LOCAL_RECOGNIZER_H_ */
