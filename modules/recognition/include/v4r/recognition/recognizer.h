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

/**
*
*      @author Aitor Aldoma
*      @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
*      @date Feb, 2013
*      @brief object instance recognizer
*/

#ifndef V4R_RECOGNIZER_H_
#define V4R_RECOGNIZER_H_

#include <v4r_config.h>
#include <v4r/core/macros.h>
#include <v4r/recognition/hypotheses_verification.h>
#include <v4r/recognition/local_rec_object_hypotheses.h>
#include <v4r/recognition/source.h>

#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace v4r
{
    template<typename PointT>
    class V4R_EXPORTS Recognizer
    {
      public:
        class V4R_EXPORTS Parameter
        {
        public:
            double voxel_size_icp_;
            double max_corr_distance_; /// @brief defines the margin for the bounding box used when doing pose refinement with ICP of the cropped scene to the model
            int normal_computation_method_; /// @brief chosen normal computation method of the V4R library
            bool merge_close_hypotheses_; /// @brief if true, close correspondence clusters (object hypotheses) of the same object model are merged together and this big cluster is refined
            double merge_close_hypotheses_dist_; /// @brief defines the maximum distance of the centroids in meter for clusters to be merged together
            double merge_close_hypotheses_angle_; /// @brief defines the maximum angle in degrees for clusters to be merged together
            int resolution_mm_model_assembly_; /// @brief the resolution in millimeters of the model when it gets assembled into a point cloud
            double max_distance_; /// @brief max distance in meters for recognition

            Parameter(
                    double voxel_size_icp = 0.0025f,
                    double max_corr_distance = 0.03f,
                    int normal_computation_method = 2,
                    bool merge_close_hypotheses = true,
                    double merge_close_hypotheses_dist = 0.02f,
                    double merge_close_hypotheses_angle = 10.f,
                    int resolution_mm_model_assembly = 3,
                    double max_distance = std::numeric_limits<double>::max())
                : voxel_size_icp_ (voxel_size_icp),
                  max_corr_distance_ (max_corr_distance),
                  normal_computation_method_ (normal_computation_method),
                  merge_close_hypotheses_ (merge_close_hypotheses),
                  merge_close_hypotheses_dist_ (merge_close_hypotheses_dist),
                  merge_close_hypotheses_angle_ (merge_close_hypotheses_angle),
                  resolution_mm_model_assembly_ (resolution_mm_model_assembly),
                  max_distance_ (max_distance)
            {}
        }param_;

      protected:
        typedef Model<PointT> ModelT;
        typedef boost::shared_ptr<ModelT> ModelTPtr;

        typedef typename std::map<std::string, ObjectHypothesis<PointT> > symHyp;

        typedef typename pcl::PointCloud<PointT>::Ptr PointTPtr;
        typedef typename pcl::PointCloud<PointT>::ConstPtr ConstPointTPtr;

        PointTPtr scene_; /// \brief Point cloud to be classified
        std::vector<int> indices_; /// @brief segmented cloud to be recognized (if empty, all points will be processed)
        pcl::PointCloud<pcl::Normal>::Ptr scene_normals_; /// \brief Point cloud to be classified
        typename Source<PointT>::Ptr source_;  /// \brief Model data source

        mutable boost::shared_ptr<pcl::visualization::PCLVisualizer> vis_;
        mutable int vp1_, vp2_, vp3_;

        /** @brief: generated object hypotheses from correspondence grouping (before verification) */
        std::vector<ModelTPtr> models_;
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_;

        /** @brief boolean vector defining if model or plane is verified (models are first in the vector and its size is equal to models_) */
        std::vector<bool> hypothesis_is_verified_;

        bool requires_segmentation_;

        std::string models_dir_; /// \brief Directory containing the object models

        /** \brief Hypotheses verification algorithm */
        typename boost::shared_ptr<HypothesisVerification<PointT, PointT> > hv_algorithm_;

        void hypothesisVerification ();


      public:

        Recognizer(const Parameter &p = Parameter())
        {
          param_ = p;
          requires_segmentation_ = false;
        }

        virtual size_t getFeatureType() const = 0;

        virtual bool
        needNormals() const = 0;

        /**
         * \brief Sets the model data source_
         */
        void
        setDataSource (const typename Source<PointT>::Ptr & source)
        {
          source_ = source;
        }

        typename Source<PointT>::Ptr
        getDataSource() const
        {
            return source_;
        }

        virtual bool
        initialize(bool force_retrain)
        {
            (void) force_retrain;
            PCL_WARN("initialize is not implemented for this class.");
            return true;
        }

        /**
         * @brief sets the Hypotheses Verification algorithm
         * @param alg
         */
        void
        setHVAlgorithm (const typename HypothesisVerification<PointT, PointT>::Ptr & alg)
        {
          hv_algorithm_ = alg;
        }

        void
        setInputCloud (const PointTPtr cloud)
        {
          scene_ = cloud;
        }

        /**
         * @brief return all generated object hypotheses
         * @return potential object model in the scene (not aligned to the scene)
         */
        std::vector<ModelTPtr>
        getModels() const
        {
          return models_;
        }


        /**
         * @brief returns only the models for the objects that have been verified
         * @return verified object model of the scene (not aligned to the scene)
         */
        std::vector<ModelTPtr>
        getVerifiedModels () const
        {
            std::vector<ModelTPtr> models_verified;

            if(hypothesis_is_verified_.size() < models_.size()) {
                std::cout << "Verification vector is not valid. Did you run hyphotheses verification?" << std::endl;
                return models_verified;
            }

            for(size_t i=0; i<models_.size(); i++) {
                if (hypothesis_is_verified_[i])
                    models_verified.push_back(models_[i]);
            }
            return models_verified;
        }


        /**
         * @brief return all transforms of generated object hypotheses with respect to the scene
         * @return 4x4 homogenous transformation matrix
         */
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >
        getTransforms () const
        {
          return transforms_;
        }


        /**
         * @brief returns only the transformations for the objects that have been verified
         * @return 4x4 homogenous transformation matrix of verified model to the scene
         */
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >
        getVerifiedTransforms () const
        {
          std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_verified;

          if(hypothesis_is_verified_.size() < transforms_.size()) {
              std::cout << "Verification vector is not valid. Did you run hyphotheses verification?" << std::endl;
              return transforms_verified;
          }

          for(size_t i=0; i<transforms_.size(); i++) {
              if (hypothesis_is_verified_[i])
                  transforms_verified.push_back(transforms_[i]);
          }
          return transforms_verified;
        }


        /**
         * @brief Filesystem dir containing training files
         */
        void
        setModelsDir (const std::string & dir)
        {
          models_dir_ = dir;
        }

        void setSceneNormals(const pcl::PointCloud<pcl::Normal>::Ptr &normals)
        {
            scene_normals_ = normals;
        }

        virtual bool requiresSegmentation() const
        {
          return requires_segmentation_;
        }

        void visualize () const;

        virtual void recognize () = 0;

        typedef boost::shared_ptr< Recognizer<PointT> > Ptr;
        typedef boost::shared_ptr< Recognizer<PointT> const> ConstPtr;
    };
}
#endif /* RECOGNIZER_H_ */
