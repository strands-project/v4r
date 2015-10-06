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

        typename pcl::PointCloud<PointT>::Ptr scene_; // input point cloud of the scene
        pcl::PointCloud<pcl::Normal>::Ptr scene_normals_;
        pcl::CorrespondencesPtr model_scene_corresp_; //indices between model keypoints (index query) and scene cloud (index match)
        std::vector<int> indices_to_flann_models_;

        void visualize() const;

//        ObjectHypothesis & operator+=(const ObjectHypothesis &rhs);
    };

    template<typename PointInT>
    class V4R_EXPORTS Recognizer
    {
      typedef Model<PointInT> ModelT;
      typedef boost::shared_ptr<ModelT> ModelTPtr;

      typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;
      typedef typename pcl::PointCloud<PointInT>::ConstPtr ConstPointInTPtr;

      protected:
        /** \brief Point cloud to be classified */
        PointInTPtr scene_;
        pcl::PointCloud<pcl::Normal>::Ptr scene_normals_;
        mutable boost::shared_ptr<pcl::visualization::PCLVisualizer> vis_;
        mutable int vp1_, vp2_, vp3_;

        std::vector<ModelTPtr> models_, models_verified_;
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_, transforms_verified_;

        int ICP_iterations_;
        int icp_type_;
        float VOXEL_SIZE_ICP_;
        float max_corr_distance_;
        bool requires_segmentation_;
        std::vector<int> indices_;
        bool recompute_hv_normals_;
        pcl::PointIndicesPtr icp_scene_indices_;

        /** \brief Directory containing views of the object */
        std::string training_dir_;

        /** \brief Hypotheses verification algorithm */
        typename boost::shared_ptr<v4r::HypothesisVerification<PointInT, PointInT> > hv_algorithm_;

        void poseRefinement();
        void hypothesisVerification ();


      public:

        Recognizer()
        {
          ICP_iterations_ = 30;
          VOXEL_SIZE_ICP_ = 0.0025f;
          icp_type_ = 1;
          max_corr_distance_ = 0.02f;
          requires_segmentation_ = false;
          recompute_hv_normals_ = true;
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

        void setSceneNormals(const pcl::PointCloud<pcl::Normal>::Ptr &normals)
        {
            scene_normals_ = normals;
        }

        virtual void
        setSaveHypotheses(bool b)
        {
            (void)b;
            PCL_WARN("Set save hypotheses is not implemented for this class.");
        }

        virtual
        void
        getSavedHypotheses(std::map<std::string, ObjectHypothesis<PointInT> > & hypotheses) const
        {
            (void)hypotheses;
            PCL_WARN("Get saved hypotheses is not implemented for this class.");
        }

        virtual
        void
        getKeypointCloud(PointInTPtr & keypoint_cloud) const
        {
            (void)keypoint_cloud;
            PCL_WARN("Get keypoint cloud is not implemented for this class.");
        }

        virtual
        void
        getKeypointIndices(pcl::PointIndices & indices) const
        {
            (void)indices;
            PCL_WARN("Get keypoint indices is not implemented for this class.");
        }

        virtual void recognize () = 0;

        virtual typename boost::shared_ptr<Source<PointInT> >
        getDataSource () const = 0;

        virtual void reinitialize(const std::vector<std::string> & load_ids = std::vector<std::string>())
        {
            (void)load_ids;
            PCL_WARN("Reinitialize is not implemented for this class.");
        }

        /*virtual void
        setHVAlgorithm (typename boost::shared_ptr<pcl::HypothesisVerification<PointInT, PointInT> > & alg) = 0;*/

        void
        setHVAlgorithm (const typename boost::shared_ptr<v4r::HypothesisVerification<PointInT, PointInT> > & alg)
        {
          hv_algorithm_ = alg;
        }

        void
        setInputCloud (const PointInTPtr & cloud)
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
          ICP_iterations_ = it;
        }

        void setICPType(int t) {
          icp_type_ = t;
        }

        void setVoxelSizeICP(float s) {
          VOXEL_SIZE_ICP_ = s;
        }

        /**
         * \brief Filesystem dir containing training files
         */
        void
        setTrainingDir (const std::string & dir)
        {
          training_dir_ = dir;
        }

        virtual bool requiresSegmentation() const
        {
          return requires_segmentation_;
        }

        virtual void
        setIndices (const std::vector<int> & indices) {
          indices_ = indices;
        }

        void visualize () const;
    };
}
#endif /* RECOGNIZER_H_ */
