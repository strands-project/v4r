/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id: icp.h 8409 2013-01-03 10:02:16Z aaldoma $
 *
 */

#ifndef FAAT_PCL_ICP_H_
#define FAAT_PCL_ICP_H_

// PCL includes
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_registration.h>
#include <pcl/registration/registration.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/default_convergence_criteria.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/angles.h>
#include "faat_pcl/registration/cuda/icp_with_gc.h"

using namespace pcl;

namespace faat_pcl
{

  template<typename T>
  struct transformToGPUCloudFormat
  {
    inline void
    operator () (const typename pcl::PointCloud<T> & pcl_cloud,
                  std::vector<faat_pcl::xyz_p> & gpu_cloud)
    {
      gpu_cloud.resize(pcl_cloud.points.size());
      for(size_t i=0; i < gpu_cloud.size(); i++)
      {
        gpu_cloud[i].x = pcl_cloud.points[i].x;
        gpu_cloud[i].y = pcl_cloud.points[i].y;
        gpu_cloud[i].z = pcl_cloud.points[i].z;
      }
    }
  };

  template<>
  struct transformToGPUCloudFormat<pcl::PointNormal>
  {
    inline void
    operator () (const typename pcl::PointCloud<pcl::PointNormal> & pcl_cloud,
                  std::vector<faat_pcl::xyz_rgb> & gpu_cloud)
    {
      gpu_cloud.resize(pcl_cloud.points.size());
      for(size_t i=0; i < gpu_cloud.size(); i++)
      {
        gpu_cloud[i].x = pcl_cloud.points[i].x;
        gpu_cloud[i].y = pcl_cloud.points[i].y;
        gpu_cloud[i].z = pcl_cloud.points[i].z;
        gpu_cloud[i].rgb_set = false;
      }
    }
  };

  template<>
  struct transformToGPUCloudFormat<pcl::PointXYZRGBNormal>
  {
    inline void
    operator () (const typename pcl::PointCloud<pcl::PointXYZRGBNormal> & pcl_cloud,
                  std::vector<faat_pcl::xyz_rgb> & gpu_cloud)
    {
      gpu_cloud.resize(pcl_cloud.points.size());
      for(size_t i=0; i < gpu_cloud.size(); i++)
      {
        gpu_cloud[i].x = pcl_cloud.points[i].x;
        gpu_cloud[i].y = pcl_cloud.points[i].y;
        gpu_cloud[i].z = pcl_cloud.points[i].z;
        gpu_cloud[i].rgb_set = true;
        gpu_cloud[i].rgb = pcl_cloud.points[i].rgb;
      }

      std::cout << "called transformToGPUCloudFormat<pcl::PointXYZRGBNormal> -- rgb should be set" << std::endl;
    }
  };

  template <typename PointSource, typename PointTarget, typename Scalar = float>
  class IterativeClosestPointWithGC : public Registration<PointSource, PointTarget, Scalar>
  {

    struct ICPNode
    {
        ICPNode(int l, bool root=false)
        {
          converged_ = false;
          is_root_ = root;
          level_ = l;
          color_weight_ = 0.f;
          reg_error_ = 0.f;
          osv_fraction_ = 0.f;
          fsv_fraction_ = 0.f;
          overlap_ = 0;
          incr_transform_.setIdentity();
          accum_transform_.setIdentity();
          childs_.resize(0);
        }

        void
        addChild(boost::shared_ptr<ICPNode> & c)
        {
          childs_.push_back(c);
        }

        boost::shared_ptr<ICPNode> parent_;
        bool is_root_;
        int level_; //equivalent to the ICP iteration
        Eigen::Matrix4f incr_transform_; //transform from parent to current node
        Eigen::Matrix4f accum_transform_;
        bool converged_; //wether the alignment path converged or not...
        std::vector< boost::shared_ptr<ICPNode> > childs_;
        float reg_error_;
        float color_weight_;
        int overlap_;
        float osv_fraction_;
        float fsv_fraction_;
        CorrespondencesPtr after_rej_correspondences_;
    };

    transformToGPUCloudFormat<PointSource> getFieldValueSource_;
    transformToGPUCloudFormat<PointTarget> getFieldValueTarget_;

    public:
      typedef typename Registration<PointSource, PointTarget, Scalar>::PointCloudSource PointCloudSource;
      typedef typename PointCloudSource::Ptr PointCloudSourcePtr;
      typedef typename PointCloudSource::ConstPtr PointCloudSourceConstPtr;

      typedef typename Registration<PointSource, PointTarget, Scalar>::PointCloudTarget PointCloudTarget;
      typedef typename PointCloudTarget::Ptr PointCloudTargetPtr;
      typedef typename PointCloudTarget::ConstPtr PointCloudTargetConstPtr;

      typedef PointIndices::Ptr PointIndicesPtr;
      typedef PointIndices::ConstPtr PointIndicesConstPtr;

      typedef boost::shared_ptr<faat_pcl::IterativeClosestPointWithGC<PointSource, PointTarget, Scalar> > Ptr;
      typedef boost::shared_ptr<const faat_pcl::IterativeClosestPointWithGC<PointSource, PointTarget, Scalar> > ConstPtr;

      using Registration<PointSource, PointTarget, Scalar>::reg_name_;
      using Registration<PointSource, PointTarget, Scalar>::getClassName;
      using Registration<PointSource, PointTarget, Scalar>::setInputSource;
      using Registration<PointSource, PointTarget, Scalar>::input_;
      using Registration<PointSource, PointTarget, Scalar>::indices_;
      using Registration<PointSource, PointTarget, Scalar>::target_;
      using Registration<PointSource, PointTarget, Scalar>::nr_iterations_;
      using Registration<PointSource, PointTarget, Scalar>::max_iterations_;
      using Registration<PointSource, PointTarget, Scalar>::previous_transformation_;
      using Registration<PointSource, PointTarget, Scalar>::final_transformation_;
      using Registration<PointSource, PointTarget, Scalar>::transformation_;
      using Registration<PointSource, PointTarget, Scalar>::transformation_epsilon_;
      using Registration<PointSource, PointTarget, Scalar>::converged_;
      using Registration<PointSource, PointTarget, Scalar>::corr_dist_threshold_;
      using Registration<PointSource, PointTarget, Scalar>::inlier_threshold_;
      using Registration<PointSource, PointTarget, Scalar>::min_number_correspondences_;
      using Registration<PointSource, PointTarget, Scalar>::update_visualizer_;
      using Registration<PointSource, PointTarget, Scalar>::euclidean_fitness_epsilon_;
      using Registration<PointSource, PointTarget, Scalar>::correspondences_;
      using Registration<PointSource, PointTarget, Scalar>::transformation_estimation_;
      using Registration<PointSource, PointTarget, Scalar>::correspondence_estimation_;
      using Registration<PointSource, PointTarget, Scalar>::correspondence_rejectors_;

      typename pcl::registration::DefaultConvergenceCriteria<Scalar>::Ptr convergence_criteria_;
      typedef typename Registration<PointSource, PointTarget, Scalar>::Matrix4 Matrix4;
      bool VIS_FINAL_;

      /** \brief Empty constructor. */
      IterativeClosestPointWithGC ()
        : x_idx_offset_ (0)
        , y_idx_offset_ (0)
        , z_idx_offset_ (0)
        , nx_idx_offset_ (0)
        , ny_idx_offset_ (0)
        , nz_idx_offset_ (0)
        , use_reciprocal_correspondence_ (false)
        , source_has_normals_ (false)
      {
        reg_name_ = "IterativeClosestPointWithGC";
        transformation_estimation_.reset (new pcl::registration::TransformationEstimationSVD<PointSource, PointTarget, Scalar> ());
        correspondence_estimation_.reset (new pcl::registration::CorrespondenceEstimation<PointSource, PointTarget, Scalar>);
        convergence_criteria_.reset(new pcl::registration::DefaultConvergenceCriteria<Scalar> (nr_iterations_, transformation_, *correspondences_));
        use_cg_ = false;
        use_shot_ = true;
        survival_of_the_fittest_ = true;
        ov_percentage_ = 0.5f;
        VIS_FINAL_ = false;
        dt_vx_size_ = 0.003f;
        trans_to_centroid_ = true;
        range_images_provided_ = false;
        use_range_images_ = true;
        inliers_threshold_ = 0.01f;
        use_color_ = false;
      };

      void setuseColor(bool b)
      {
          use_color_ = b;
      }

      void setInitialPoses(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & poses)
      {
        initial_poses_ = poses;
      }

      void setUseRangeImages(bool b)
      {
          use_range_images_ = b;
      }

      void getResults(std::vector<std::pair<float, Eigen::Matrix4f> > & r)
      {
        r = result_;
      }

      inline void
      setTransToCentroid(bool b)
      {
        trans_to_centroid_ = b;
      }

      inline void
      setDtVxSize(float f)
      {
        dt_vx_size_ = f;
      }

      inline void
      setOverlapPercentage(float f)
      {
        ov_percentage_ = f;
      }

      inline void
      setMinNumCorrespondences(int t)
      {
        min_number_correspondences_ = t;
      }

      inline void
      setSourceAndTargetIndices(pcl::IndicesPtr & src, pcl::IndicesPtr & tgt)
      {
            ind_src_ = src;
            ind_tgt_ = tgt;
      }

      /** \brief Returns a pointer to the DefaultConvergenceCriteria used by the IterativeClosestPoint class.
        * This allows to check the convergence state after the align() method as well as to configure
        * DefaultConvergenceCriteria's parameters not available through the ICP API before the align()
        * method is called. Please note that the align method sets max_iterations_,
        * euclidean_fitness_epsilon_ and transformation_epsilon_ and therefore overrides the default / set
        * values of the DefaultConvergenceCriteria instance.
        * \param[out] Pointer to the IterativeClosestPoint's DefaultConvergenceCriteria.
        */
      inline typename pcl::registration::DefaultConvergenceCriteria<Scalar>::Ptr
      getConvergeCriteria ()
      {
        return convergence_criteria_;
      }

      inline void
      setSurvivalOfTheFittest(bool b)
      {
        survival_of_the_fittest_ = b;
      }

      /** \brief Provide a pointer to the input source 
        * (e.g., the point cloud that we want to align to the target)
        *
        * \param[in] cloud the input point cloud source
        */
      virtual void
      setInputSource (const PointCloudSourceConstPtr &cloud)
      {
        Registration<PointSource, PointTarget, Scalar>::setInputSource (cloud);
        std::vector<pcl::PCLPointField> fields;
        pcl::getFields (*cloud, fields);
        source_has_normals_ = false;

        for (size_t i = 0; i < fields.size (); ++i)
        {
          if      (fields[i].name == "x") x_idx_offset_ = fields[i].offset;
          else if (fields[i].name == "y") y_idx_offset_ = fields[i].offset;
          else if (fields[i].name == "z") z_idx_offset_ = fields[i].offset;
          else if (fields[i].name == "normal_x") 
          {
            source_has_normals_ = true;
            nx_idx_offset_ = fields[i].offset;
          }
          else if (fields[i].name == "normal_y") 
          {
            source_has_normals_ = true;
            ny_idx_offset_ = fields[i].offset;
          }
          else if (fields[i].name == "normal_z") 
          {
            source_has_normals_ = true;
            nz_idx_offset_ = fields[i].offset;
          }
        }
      }

      /** \brief Set whether to use reciprocal correspondence or not
        *
        * \param[in] use_reciprocal_correspondence whether to use reciprocal correspondence or not
        */
      inline void
      setUseReciprocalCorrespondences (bool use_reciprocal_correspondence)
      {
        use_reciprocal_correspondence_ = use_reciprocal_correspondence;
      }

      /** \brief Obtain whether reciprocal correspondence are used or not */
      inline bool
      getUseReciprocalCorrespondences () const
      {
        return (use_reciprocal_correspondence_);
      }

      inline void
      setUseCG(bool b)
      {
        use_cg_ = b;
      }

      inline void
      setUseSHOT(bool b)
      {
        use_shot_ = b;
      }

      void
      setVisFinal(bool b)
      {
        VIS_FINAL_ = b;
      }

      template<typename T>
      void
      setRangeImages(typename pcl::PointCloud<T>::Ptr & ri_source,
                       typename pcl::PointCloud<T>::Ptr & ri_target,
                       float fl, float cx, float cy)
      {
        range_image_source_.reset(new PointCloudSource);
        range_image_target_.reset(new PointCloudTarget);
        pcl::copyPointCloud(*ri_source, *range_image_source_);
        pcl::copyPointCloud(*ri_target, *range_image_target_);
        range_images_provided_ = true;
        fl_ = fl;
        cx_ = cx;
        cy_ = cy;
      }

      void setInliersThreshold(float t)
      {
          inliers_threshold_ = t;
      }

    protected:

      /** \brief Apply a rigid transform to a given dataset. Here we check whether whether
        * the dataset has surface normals in addition to XYZ, and rotate normals as well.
        * \param[in] input the input point cloud
        * \param[out] output the resultant output point cloud
        * \param[in] transform a 4x4 rigid transformation
        * \note Can be used with cloud_in equal to cloud_out
        */
      virtual void 
      transformCloud (const PointCloudSource &input, 
                      PointCloudSource &output, 
                      const Matrix4 &transform);

      /** \brief Rigid transformation computation method  with initial guess.
        * \param output the transformed input point cloud dataset using the rigid transformation found
        * \param guess the initial guess of the transformation to compute
        */
      virtual void 
      computeTransformation (PointCloudSource &output, const Matrix4 &guess);

      void
      drawCorrespondences (PointCloudTargetConstPtr scene_cloud, PointCloudSourcePtr model_cloud,
                              PointCloudTargetConstPtr keypoints_pointcloud, PointCloudSourcePtr keypoints_model, pcl::Correspondences & correspondences,
                              pcl::visualization::PCLVisualizer & vis, int viewport);

      void
      visualizeICPNodes(std::vector<boost::shared_ptr<ICPNode> > & nodes,
                          pcl::visualization::PCLVisualizer & vis,
                          std::string wname="icp nodes");

      bool
      filterHypothesesByPose(boost::shared_ptr<ICPNode> & current,
                             std::vector<boost::shared_ptr<ICPNode> > & nodes,
                             float trans_threshold);

      bool
      filterHypothesesByPose(Eigen::Matrix4f & current,
                             std::vector< Eigen::Matrix4f> & accepted_so_far,
                             float trans_threshold,
                             float angle=15.f);

      void
      visualizeOrigins(std::vector<boost::shared_ptr<ICPNode> > & nodes)
      {
        pcl::visualization::PCLVisualizer vis("visualizeOrigins");
        Eigen::Vector4f origin;
        pcl::compute3DCentroid(*input_, origin);
        origin[3] = 1.f;
        std::cout << "Centroid original input" << std::endl;
        std::cout << origin << std::endl;

        std::stringstream cen_;
        cen_ << "centroid_" << origin;
        pcl::PointXYZ cen_orig;
        cen_orig.getVector4fMap() = origin;
        vis.addSphere(cen_orig, 0.015, 0.0, 1.0, 1.0, cen_.str());

        {
          pcl::visualization::PointCloudColorHandlerCustom<PointTarget> handler (target_, 255, 255, 255);
          vis.addPointCloud<PointTarget> (target_, handler, "target");
        }

        {
          pcl::visualization::PointCloudColorHandlerCustom<PointSource> handler (input_, 0, 0, 255);
          vis.addPointCloud<PointSource> (input_, handler, "input_");
        }
        for(size_t i=0; i < nodes.size(); i++)
        {
          Eigen::Vector4f origin_node = nodes[i]->accum_transform_ * origin;
          std::stringstream cloud_name;
          cloud_name << "input_" << i;
          pcl::PointXYZ p;

          p.getVector4fMap() =  origin_node; //.block<3,1>(0, 0);
          if(i == 0)
            vis.addSphere(p, 0.01, 1.0, 0, 0, cloud_name.str());
          else
            vis.addSphere(p, 0.01, cloud_name.str());

          /*Eigen::Quaternionf quat(static_cast<Eigen::Matrix4f>(nodes[i]->accum_transform_).block<3,3>(0,0));
          quat.normalize();
          Eigen::Quaternionf quat_conj = quat.conjugate();*/

          PointCloudSourcePtr input_transformed_intern (new PointCloudSource(*input_));
          pcl::transformPointCloud (*input_, *input_transformed_intern, nodes[i]->accum_transform_);

          Eigen::Vector4f centroid_i;
          pcl::compute3DCentroid(*input_transformed_intern, centroid_i);

          pcl::PointXYZ cen_i;
          cen_i.getVector4fMap() = centroid_i;
          std::stringstream cen_;
          cen_ << "centroid_" << i;
          vis.addSphere(cen_i, 0.005, 0.0, 1.0, 0.0, cen_.str());

          std::cout << "trasnform:" << std::endl;
          std::cout << nodes[i]->accum_transform_ << std::endl;
          std::cout << "centroid transformed:" << std::endl;
          std::cout << origin_node << std::endl;
          std::cout << "centroid computed:" << std::endl;
          std::cout << centroid_i << std::endl;

          for(size_t j=(i+1); j < nodes.size(); j++)
          {
            Eigen::Vector4f origin_node_j = nodes[j]->accum_transform_ * origin;
            pcl::PointXYZ p2;
            //p2.getVector3fMap() = static_cast<Eigen::Matrix4f>(nodes[j]->accum_transform_).block<3,1>(0,3);
            p2.getVector4fMap() = origin_node_j;
            std::stringstream line_name;
            line_name << "line_" << i << "_" << j;
            vis.addLine<pcl::PointXYZ, pcl::PointXYZ> (p, p2, line_name.str ());

            pcl::PointXYZ mid_point;
            mid_point.getVector3fMap() = (p.getVector3fMap() + p2.getVector3fMap()) / 2.f;

            std::stringstream camera_name;
            camera_name << "camera_" << i << "_" << j;

            char cVal[32];
            sprintf(cVal,"%.2f", (p.getVector3fMap() - p2.getVector3fMap()).norm());

//            std::cout << static_cast<Eigen::Matrix4f>(nodes[i]->accum_transform_).block<3,1>(0,3) << std::endl;
//            std::cout << static_cast<Eigen::Matrix4f>(nodes[j]->accum_transform_).block<3,1>(0,3) << std::endl;
//            std::cout << p.getVector3fMap() - p2.getVector3fMap() << std::endl;
//            std::cout << static_cast<Eigen::Matrix4f>(nodes[i]->accum_transform_).block<3,1>(0,3) - static_cast<Eigen::Matrix4f>(nodes[j]->accum_transform_).block<3,1>(0,3) << std::endl;
//            std::cout << "norm:" << (static_cast<Eigen::Matrix4f>(nodes[i]->accum_transform_).block<3,1>(0,3) - static_cast<Eigen::Matrix4f>(nodes[j]->accum_transform_).block<3,1>(0,3)).norm() << std::endl;
//            Eigen::Quaternionf quat_found(static_cast<Eigen::Matrix4f>(nodes[j]->accum_transform_).block<3,3>(0,0));
//            quat_found.normalize();
//            Eigen::Quaternionf quat_prod = quat_found * quat_conj;
//            float angle = static_cast<float>(acos(quat_prod.z()));
//            angle = std::abs(90.f - pcl::rad2deg(angle));
//            std::cout << angle << " " << i << " " << j << std::endl;

            vis.addText3D (cVal, mid_point, 0.005, 0.0, 1.0, 0.0, camera_name.str ());

            PointCloudSourcePtr input_transformed_intern (new PointCloudSource(*input_));
            transformCloud (*input_, *input_transformed_intern, nodes[j]->accum_transform_);

            Eigen::Vector4f centroid_j;
            pcl::compute3DCentroid(*input_transformed_intern, centroid_j);

            std::stringstream cen_;
            cen_ << "centroid_" << j;
            pcl::PointXYZ cen_j;
            cen_j.getVector4fMap() = centroid_j;
            vis.addSphere(cen_j, 0.005, 0.0, 1.0, 0.0, cen_.str());

            //vis.spin();

            vis.removeShape(cen_.str());
            //vis.removePointCloud("input j");
          }

          vis.removeShape(cen_.str());
          //vis.removePointCloud("input i");

          vis.spin();
        }

        vis.spin();
      }

      /** \brief XYZ fields offset. */
      size_t x_idx_offset_, y_idx_offset_, z_idx_offset_;

      /** \brief Normal fields offset. */
      size_t nx_idx_offset_, ny_idx_offset_, nz_idx_offset_;

      /** \brief The correspondence type used for correspondence estimation. */
      bool use_reciprocal_correspondence_;

      /** \brief Internal check whether dataset has normals or not. */
      bool source_has_normals_;

      bool use_cg_;

      bool use_shot_;

      bool survival_of_the_fittest_;

      float ov_percentage_;

      float dt_vx_size_;

      bool trans_to_centroid_;

      bool use_color_;

      float cx_, cy_, fl_;
      PointCloudTargetPtr range_image_target_;
      PointCloudSourcePtr range_image_source_;
      bool range_images_provided_;
      bool use_range_images_;
      std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > initial_poses_;
      std::vector<std::pair<float, Eigen::Matrix4f> > result_;
      float inliers_threshold_;

      pcl::IndicesPtr ind_src_, ind_tgt_;
  };

  /** \brief @b IterativeClosestPointWithNormals is a special case of
    * IterativeClosestPoint, that uses a transformation estimated based on
    * Point to Plane distances by default.
    *
    * \author Radu B. Rusu
    * \ingroup registration
    */
  template <typename PointSource, typename PointTarget, typename Scalar = float>
  class IterativeClosestPointWithGCWithNormals : public IterativeClosestPointWithGC<PointSource, PointTarget, Scalar>
  {
    public:
      typedef typename IterativeClosestPointWithGC<PointSource, PointTarget, Scalar>::PointCloudSource PointCloudSource;
      typedef typename IterativeClosestPointWithGC<PointSource, PointTarget, Scalar>::PointCloudTarget PointCloudTarget;
      typedef typename IterativeClosestPointWithGC<PointSource, PointTarget, Scalar>::Matrix4 Matrix4;

      using IterativeClosestPointWithGC<PointSource, PointTarget, Scalar>::reg_name_;
      using IterativeClosestPointWithGC<PointSource, PointTarget, Scalar>::transformation_estimation_;
      using IterativeClosestPointWithGC<PointSource, PointTarget, Scalar>::correspondence_rejectors_;

      typedef boost::shared_ptr<IterativeClosestPointWithGC<PointSource, PointTarget, Scalar> > Ptr;
      typedef boost::shared_ptr<const IterativeClosestPointWithGC<PointSource, PointTarget, Scalar> > ConstPtr;

      /** \brief Empty constructor. */
      IterativeClosestPointWithGCWithNormals ()
      {
        reg_name_ = "IterativeClosestPointWithGCWithNormals";
        transformation_estimation_.reset (new pcl::registration::TransformationEstimationPointToPlaneLLS<PointSource, PointTarget, Scalar> ());
        //correspondence_rejectors_.add
      };

    protected:

      /** \brief Apply a rigid transform to a given dataset
        * \param[in] input the input point cloud
        * \param[out] output the resultant output point cloud
        * \param[in] transform a 4x4 rigid transformation
        * \note Can be used with cloud_in equal to cloud_out
        */
      virtual void 
      transformCloud (const PointCloudSource &input, 
                      PointCloudSource &output, 
                      const Matrix4 &transform);
  };
}

#include <faat_pcl/registration/impl/icp_with_gc.hpp>

#endif  //#ifndef PCL_ICP_H_
