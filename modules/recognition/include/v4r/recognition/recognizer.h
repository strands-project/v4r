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

        typename pcl::PointCloud<PointT>::Ptr scene; // input point cloud of the scene
        typename pcl::PointCloud<PointT>::Ptr scene_keypoints; // keypoints of the scene
        typename pcl::PointCloud<PointT>::Ptr model_keypoints; //keypoints of model
        pcl::PointCloud<pcl::Normal>::Ptr model_kp_normals; //keypoint normals of model
        pcl::CorrespondencesPtr model_scene_corresp; //indices between model keypoints (index query) and scene cloud (index match)
        std::vector<int> indices_to_flann_models_;

        void visualize() const;
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
        PointInTPtr input_;

        std::vector<ModelTPtr> models_;
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_;

        int ICP_iterations_;
        int icp_type_;
        float VOXEL_SIZE_ICP_;
        float max_corr_distance_;
        bool requires_segmentation_;
        std::vector<int> indices_;
        bool recompute_hv_normals_;
        pcl::PointIndicesPtr icp_scene_indices_;

        /** \brief Hypotheses verification algorithm */
        typename boost::shared_ptr<v4r::HypothesisVerification<PointInT, PointInT> > hv_algorithm_;

        void poseRefinement()
        {
          PointInTPtr scene_voxelized (new pcl::PointCloud<PointInT> ());
          pcl::VoxelGrid<PointInT> voxel_grid_icp;
          voxel_grid_icp.setInputCloud (input_);
          if(icp_scene_indices_ && icp_scene_indices_->indices.size() > 0)
          {
            voxel_grid_icp.setIndices(icp_scene_indices_);
          }
          voxel_grid_icp.setLeafSize (VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_);
          voxel_grid_icp.filter (*scene_voxelized);

          switch (icp_type_)
          {
            case 0:
            {
    #pragma omp parallel for schedule(dynamic,1) num_threads(omp_get_num_procs())
              for (int i = 0; i < static_cast<int> (models_.size ()); i++)
              {
                ConstPointInTPtr model_cloud;
                PointInTPtr model_aligned (new pcl::PointCloud<PointInT>);
                model_cloud = models_[i]->getAssembled (VOXEL_SIZE_ICP_);
                pcl::transformPointCloud (*model_cloud, *model_aligned, transforms_[i]);

                typename pcl::registration::CorrespondenceRejectorSampleConsensus<PointInT>::Ptr
                                        rej (new pcl::registration::CorrespondenceRejectorSampleConsensus<PointInT> ());

                rej->setInputTarget (scene_voxelized);
                rej->setMaximumIterations (1000);
                rej->setInlierThreshold (0.005f);
                rej->setInputSource (model_aligned);

                pcl::IterativeClosestPoint<PointInT, PointInT> reg;
                reg.addCorrespondenceRejector (rej);
                reg.setInputTarget (scene_voxelized);
                reg.setInputSource (model_aligned);
                reg.setMaximumIterations (ICP_iterations_);
                reg.setMaxCorrespondenceDistance (max_corr_distance_);

                typename pcl::PointCloud<PointInT>::Ptr output_ (new pcl::PointCloud<PointInT> ());
                reg.align (*output_);

                Eigen::Matrix4f icp_trans = reg.getFinalTransformation ();
                transforms_[i] = icp_trans * transforms_[i];
              }
            }
              break;
            default:
            {
              #pragma omp parallel for schedule(dynamic,1) num_threads(omp_get_num_procs())
              for (int i = 0; i < static_cast<int> (models_.size ()); i++)
              {
                typename VoxelBasedCorrespondenceEstimation<PointInT, PointInT>::Ptr
                            est (new VoxelBasedCorrespondenceEstimation<PointInT, PointInT> ());

                typename pcl::registration::CorrespondenceRejectorSampleConsensus<PointInT>::Ptr
                            rej (new pcl::registration::CorrespondenceRejectorSampleConsensus<PointInT> ());

                Eigen::Matrix4f scene_to_model_trans = transforms_[i].inverse ();
                boost::shared_ptr<distance_field::PropagationDistanceField<PointInT> > dt;
                models_[i]->getVGDT (dt);

                PointInTPtr model_aligned (new pcl::PointCloud<PointInT>);
                PointInTPtr cloud_voxelized_icp_cropped (new pcl::PointCloud<PointInT>);
                typename pcl::PointCloud<PointInT>::ConstPtr cloud;
                dt->getInputCloud(cloud);
                model_aligned.reset(new pcl::PointCloud<PointInT>(*cloud));

                pcl::transformPointCloud (*scene_voxelized, *cloud_voxelized_icp_cropped, scene_to_model_trans);

                PointInT minPoint, maxPoint;
                pcl::getMinMax3D(*cloud, minPoint, maxPoint);
                minPoint.x -= max_corr_distance_;
                minPoint.y -= max_corr_distance_;
                minPoint.z -= max_corr_distance_;

                maxPoint.x += max_corr_distance_;
                maxPoint.y += max_corr_distance_;
                maxPoint.z += max_corr_distance_;

                pcl::CropBox<PointInT> cropFilter;
                cropFilter.setInputCloud (cloud_voxelized_icp_cropped);
                cropFilter.setMin(minPoint.getVector4fMap());
                cropFilter.setMax(maxPoint.getVector4fMap());
                cropFilter.filter (*cloud_voxelized_icp_cropped);

                est->setVoxelRepresentationTarget (dt);
                est->setInputSource (cloud_voxelized_icp_cropped);
                est->setInputTarget (model_aligned);
                est->setMaxCorrespondenceDistance (max_corr_distance_);

                rej->setInputTarget (model_aligned);
                rej->setMaximumIterations (1000);
                rej->setInlierThreshold (0.005f);
                rej->setInputSource (cloud_voxelized_icp_cropped);

                pcl::IterativeClosestPoint<PointInT, PointInT, float> reg;
                reg.setCorrespondenceEstimation (est);
                reg.addCorrespondenceRejector (rej);
                reg.setInputTarget (model_aligned); //model
                reg.setInputSource (cloud_voxelized_icp_cropped); //scene
                reg.setMaximumIterations (ICP_iterations_);
                reg.setEuclideanFitnessEpsilon(1e-5);
                reg.setTransformationEpsilon(0.001f * 0.001f);

                pcl::registration::DefaultConvergenceCriteria<float>::Ptr convergence_criteria;
                convergence_criteria = reg.getConvergeCriteria();
                convergence_criteria->setAbsoluteMSE(1e-12);
                convergence_criteria->setMaximumIterationsSimilarTransforms(15);
                convergence_criteria->setFailureAfterMaximumIterations(false);

                typename pcl::PointCloud<PointInT>::Ptr output_ (new pcl::PointCloud<PointInT> ());
                reg.align (*output_);

                Eigen::Matrix4f icp_trans;
                icp_trans = reg.getFinalTransformation () * scene_to_model_trans;
                transforms_[i] = icp_trans.inverse ();
              }
            }
          }
        }

        void
        hypothesisVerification ()
        {
          pcl::ScopeTime thv ("HV verification");

          std::vector<typename pcl::PointCloud<PointInT>::ConstPtr> aligned_models;
          std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr> aligned_normals;
          aligned_models.resize (models_.size ());
          aligned_normals.resize (models_.size ());

#pragma omp parallel for schedule(dynamic,1) num_threads(omp_get_num_procs())
          for (size_t i = 0; i < models_.size (); i++)
          {
            //we should get the resolution of the hv_algorithm here... then we can avoid to voxel grid again when computing the cues...
            //ConstPointInTPtr model_cloud = models_->at (i)->getAssembled (0.005f);
            ConstPointInTPtr model_cloud = models_[i]->getAssembled (VOXEL_SIZE_ICP_);

            PointInTPtr model_aligned (new pcl::PointCloud<PointInT>);
            pcl::transformPointCloud (*model_cloud, *model_aligned, transforms_[i]);
            aligned_models[i] = model_aligned;

            if (hv_algorithm_->getRequiresNormals () && !recompute_hv_normals_)
            {
              pcl::PointCloud<pcl::Normal>::ConstPtr normals_cloud = models_[i]->getNormalsAssembled (VOXEL_SIZE_ICP_);
              pcl::PointCloud<pcl::Normal>::Ptr normals_aligned (new pcl::PointCloud<pcl::Normal>);
              normals_aligned->points.resize (normals_cloud->points.size ());
              normals_aligned->width = normals_cloud->width;
              normals_aligned->height = normals_cloud->height;
              for (size_t k = 0; k < normals_cloud->points.size (); k++)
              {
                Eigen::Vector3f nt (normals_cloud->points[k].normal_x, normals_cloud->points[k].normal_y, normals_cloud->points[k].normal_z);
                normals_aligned->points[k].normal_x = static_cast<float> (transforms_[i] (0, 0) * nt[0] + transforms_[i] (0, 1) * nt[1]
                    + transforms_[i] (0, 2) * nt[2]);
                normals_aligned->points[k].normal_y = static_cast<float> (transforms_[i] (1, 0) * nt[0] + transforms_[i] (1, 1) * nt[1]
                    + transforms_[i] (1, 2) * nt[2]);
                normals_aligned->points[k].normal_z = static_cast<float> (transforms_[i] (2, 0) * nt[0] + transforms_[i] (2, 1) * nt[1]
                    + transforms_[i] (2, 2) * nt[2]);

                //flip here based on vp?
                pcl::flipNormalTowardsViewpoint (model_aligned->points[k], 0, 0, 0, normals_aligned->points[k].normal[0],
                                                 normals_aligned->points[k].normal[1], normals_aligned->points[k].normal[2]);
              }

              aligned_normals[i] = normals_aligned;
            }
          }

          std::vector<bool> mask_hv;
          hv_algorithm_->setSceneCloud (input_);
          if (hv_algorithm_->getRequiresNormals () && !recompute_hv_normals_)
          {
            hv_algorithm_->addNormalsClouds (aligned_normals);
          }

          hv_algorithm_->addModels (aligned_models, true);
          hv_algorithm_->verify ();
          hv_algorithm_->getMask (mask_hv);

          std::vector<ModelTPtr>  models_temp;
          std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_temp;

          for (size_t i = 0; i < models_.size (); i++)
          {
            if (!mask_hv[i])
              continue;

            models_temp.push_back (models_[i]);
            transforms_temp.push_back (transforms_[i]);
          }

          models_ = models_temp;
          transforms_ = transforms_temp;
        }

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

        virtual void setSceneNormals(const pcl::PointCloud<pcl::Normal>::Ptr & /*normals*/)
        {
            PCL_WARN("Set scene normals is not implemented for this class.");
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
        setHVAlgorithm (const typename boost::shared_ptr<const v4r::HypothesisVerification<PointInT, PointInT> > & alg)
        {
          hv_algorithm_ = alg;
        }

        void
        setInputCloud (const PointInTPtr & cloud)
        {
          input_ = cloud;
        }

        std::vector<ModelTPtr>
        getModels () const
        {
          return models_;
        }

        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >
        getTransforms () const
        {
          return transforms_;
        }

        std::vector<ModelTPtr>
        getModelsBeforeHV () const
        {
          return models_;
        }

        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >
        getTransformsBeforeHV () const
        {
          return transforms_;
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

        virtual bool requiresSegmentation() const
        {
          return requires_segmentation_;
        }

        virtual void
        setIndices (const std::vector<int> & indices) {
          indices_ = indices;
        }
    };
}
#endif /* RECOGNIZER_H_ */
