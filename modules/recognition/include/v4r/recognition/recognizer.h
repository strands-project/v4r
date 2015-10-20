/*
 * recognizer.h
 *
 *  Created on: Feb 24, 2013
 *      Author: Aitor Aldoma
 *      Maintainer: Thomas Faeulhammer
 */

#ifndef RECOGNIZER_H_
#define RECOGNIZER_H_

#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/core/macros.h>
#include <v4r/recognition/hypotheses_verification.h>
#include <v4r/recognition/voxel_based_correspondence_estimation.h>
#include <v4r/recognition/source.h>

#include <pcl/common/common.h>
#include <pcl/common/time.h>
#include <pcl/filters/crop_box.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>

namespace v4r
{
    template<typename PointT>
    class V4R_EXPORTS ObjectHypothesis
    {
      typedef Model<PointT> ModelT;
      typedef boost::shared_ptr<ModelT> ModelTPtr;

      private:
        mutable boost::shared_ptr<pcl::visualization::PCLVisualizer> vis_;
        int vp1_;

      public:
        ModelTPtr model_;

        ObjectHypothesis()
        {
            model_scene_corresp_.reset(new pcl::Correspondences);
        }

        pcl::CorrespondencesPtr model_scene_corresp_; //indices between model keypoints (index query) and scene cloud (index match)
        std::vector<int> indices_to_flann_models_;

        void visualize(const typename pcl::PointCloud<PointT> & scene) const;

//        ObjectHypothesis & operator+=(const ObjectHypothesis &rhs);

        ObjectHypothesis & operator=(const ObjectHypothesis &rhs)
        {
            *(this->model_scene_corresp_) = *rhs.model_scene_corresp_;
            this->indices_to_flann_models_ = rhs.indices_to_flann_models_;
            this->model_ = rhs.model_;
            return *this;
        }
    };

    template<typename PointT>
    class V4R_EXPORTS Recognizer
    {
      public:
        class V4R_EXPORTS Parameter
        {
        public:
            int icp_iterations_;
            int icp_type_;
            float voxel_size_icp_;
            float max_corr_distance_;
            int normal_computation_method_;

            Parameter(
                    int icp_iterations = 30,
                    int icp_type = 1,
                    float voxel_size_icp = 0.0025f,
                    float max_corr_distance = 0.02f,
                    int normal_computation_method = 2)
                : icp_iterations_ (icp_iterations),
                  icp_type_ (icp_type),
                  voxel_size_icp_ (voxel_size_icp),
                  max_corr_distance_ (max_corr_distance),
                  normal_computation_method_ (normal_computation_method)
            {}
        }param_;

      protected:
        typedef Model<PointT> ModelT;
        typedef boost::shared_ptr<ModelT> ModelTPtr;

        typedef typename std::map<std::string, ObjectHypothesis<PointT> > symHyp;

        typedef typename pcl::PointCloud<PointT>::Ptr PointTPtr;
        typedef typename pcl::PointCloud<PointT>::ConstPtr ConstPointTPtr;

        /** \brief Point cloud to be classified */
        PointTPtr scene_;
        pcl::PointCloud<pcl::Normal>::Ptr scene_normals_;
        mutable boost::shared_ptr<pcl::visualization::PCLVisualizer> vis_;
        mutable int vp1_, vp2_, vp3_;

        std::vector<ModelTPtr> models_, models_verified_;
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_, transforms_verified_;
        std::vector<typename pcl::PointCloud<PointT>::Ptr > planes_verified_;
        bool requires_segmentation_;
        std::vector<int> indices_;
        pcl::PointIndicesPtr icp_scene_indices_;

        /** \brief Directory containing views of the object */
        std::string training_dir_;

        /** \brief Hypotheses verification algorithm */
        typename boost::shared_ptr<v4r::HypothesisVerification<PointT, PointT> > hv_algorithm_;

        void poseRefinement();
        void hypothesisVerification ();


      public:

        Recognizer(const Parameter &p = Parameter())
        {
          param_ = p;
          requires_segmentation_ = false;
        }

        virtual size_t getFeatureType() const
        {
            std::cout << "Get feature type is not implemented for this recognizer. " << std::endl;
            return 0;
        }

        virtual bool acceptsNormals() const
        {
            return false;
        }

        virtual void
        setSaveHypotheses(bool b)
        {
            (void)b;
            PCL_WARN("Set save hypotheses is not implemented for this class.");
        }

        virtual
        void
        getSavedHypotheses(symHyp &oh) const
        {
            (void)oh;
            PCL_WARN("getSavedHypotheses is not implemented for this class.");
        }

        virtual
        bool
        getSaveHypothesesParam() const
        {
            PCL_WARN("getSaveHypotheses is not implemented for this class.");
            return false;
        }

        virtual
        void
        getKeypointCloud(PointTPtr & cloud) const
        {
            (void)cloud;
            PCL_WARN("getKeypointCloud is not implemented for this class.");
        }

        virtual
        void
        getKeypointIndices(pcl::PointIndices & indices) const
        {
            (void)indices;
            PCL_WARN("Get keypoint indices is not implemented for this class.");
        }

        virtual void recognize () = 0;

        virtual typename boost::shared_ptr<Source<PointT> >
        getDataSource () const = 0;

        virtual void reinitialize(const std::vector<std::string> &load_ids = std::vector<std::string>())
        {
            (void)load_ids;
            PCL_WARN("Reinitialize is not implemented for this class.");
        }

        void setHVAlgorithm (const typename boost::shared_ptr<v4r::HypothesisVerification<PointT, PointT> > & alg)
        {
          hv_algorithm_ = alg;
        }

        void setInputCloud (const PointTPtr cloud)
        {
          scene_ = cloud;
        }

        std::vector<ModelTPtr>
        getModels () const
        {
          return models_;
        }


        std::vector<ModelTPtr>
        getVerifiedModels () const
        {
          return models_verified_;
        }


        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >
        getTransforms () const
        {
          return transforms_;
        }


        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >
        getVerifiedTransforms () const
        {
          return transforms_verified_;
        }


        void
        setICPIterations (int it)
        {
          param_.icp_iterations_ = it;
        }


        void setICPType(int t)
        {
          param_.icp_type_ = t;
        }


        void setVoxelSizeICP(float s)
        {
          param_.voxel_size_icp_ = s;
        }

        /**
         * \brief Filesystem dir containing training files
         */
        void
        setTrainingDir (const std::string & dir)
        {
          training_dir_ = dir;
        }

        void setSceneNormals(const pcl::PointCloud<pcl::Normal>::Ptr &normals)
        {
            scene_normals_ = normals;
        }

        virtual bool requiresSegmentation() const
        {
          return requires_segmentation_;
        }

        virtual void
        setIndices (const std::vector<int> & indices)
        {
          indices_ = indices;
        }

        void visualize () const;
    };
}
#endif /* RECOGNIZER_H_ */
