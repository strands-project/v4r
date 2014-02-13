/*
 * hv_go_1.h
 *
 *  Created on: Feb 27, 2013
 *      Author: aitor
 */

#ifndef FAAT_PCL_GO_1_H_
#define FAAT_PCL_GO_1_H_

#include <pcl/common/common.h>
#include <pcl/pcl_macros.h>
#include <faat_pcl/recognition/hv/hypotheses_verification.h>
#include "pcl/recognition/3rdparty/metslib/mets.hh"
#include <pcl/features/normal_3d.h>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <map>
#include <iostream>
#include <fstream>
#include <faat_pcl/recognition/hv/hv_go_opt.h>
#include <faat_pcl/utils/common_data_structures.h>

#ifdef _MSC_VER
#ifdef FAAT_REC_EXPORTS
#define FAAT_REC_API __declspec(dllexport)
#else
#define FAAT_REC_API __declspec(dllimport)
#endif
#else
#define FAAT_REC_API
#endif

//#define FAAT_PCL_RECOGNITION_USE_GPU 1

namespace faat_pcl
{

  /** \brief A hypothesis verification method proposed in
   * "A Global Hypotheses Verification Method for 3D Object Recognition", A. Aldoma and F. Tombari and L. Di Stefano and Markus Vincze, ECCV 2012
   * \author Aitor Aldoma
   * Extended with physical constraints and color information (see ICRA paper)
   */

  template<typename ModelT, typename SceneT>
    class FAAT_REC_API GlobalHypothesesVerification_1 : public faat_pcl::HypothesisVerification<ModelT, SceneT>
    {
      friend class HVGOBinaryOptimizer<ModelT, SceneT>;
      friend class move_manager<ModelT, SceneT>;
      friend class SAModel<ModelT, SceneT>;

      static float sRGB_LUT[256];
      static float sXYZ_LUT[4000];

      //////////////////////////////////////////////////////////////////////////////////////////////
      //float sRGB_LUT[256] = {- 1};

      //////////////////////////////////////////////////////////////////////////////////////////////
      //float sXYZ_LUT[4000] = {- 1};

      //////////////////////////////////////////////////////////////////////////////////////////////
      void
      RGB2CIELAB (unsigned char R, unsigned char G, unsigned char B, float &L, float &A,float &B2)
      {
        if (sRGB_LUT[0] < 0)
        {
          for (int i = 0; i < 256; i++)
          {
            float f = static_cast<float> (i) / 255.0f;
            if (f > 0.04045)
              sRGB_LUT[i] = powf ((f + 0.055f) / 1.055f, 2.4f);
            else
              sRGB_LUT[i] = f / 12.92f;
          }

          for (int i = 0; i < 4000; i++)
          {
            float f = static_cast<float> (i) / 4000.0f;
            if (f > 0.008856)
              sXYZ_LUT[i] = static_cast<float> (powf (f, 0.3333f));
            else
              sXYZ_LUT[i] = static_cast<float>((7.787 * f) + (16.0 / 116.0));
          }
        }

        float fr = sRGB_LUT[R];
        float fg = sRGB_LUT[G];
        float fb = sRGB_LUT[B];

        // Use white = D65
        const float x = fr * 0.412453f + fg * 0.357580f + fb * 0.180423f;
        const float y = fr * 0.212671f + fg * 0.715160f + fb * 0.072169f;
        const float z = fr * 0.019334f + fg * 0.119193f + fb * 0.950227f;

        float vx = x / 0.95047f;
        float vy = y;
        float vz = z / 1.08883f;

        vx = sXYZ_LUT[int(vx*4000)];
        vy = sXYZ_LUT[int(vy*4000)];
        vz = sXYZ_LUT[int(vz*4000)];

        L = 116.0f * vy - 16.0f;
        if (L > 100)
          L = 100.0f;

        A = 500.0f * (vx - vy);
        if (A > 120)
          A = 120.0f;
        else if (A <- 120)
          A = -120.0f;

        B2 = 200.0f * (vy - vz);
        if (B2 > 120)
          B2 = 120.0f;
        else if (B2<- 120)
          B2 = -120.0f;
      }

    protected:
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::mask_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::scene_cloud_downsampled_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::scene_downsampled_tree_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::visible_models_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::visible_normal_models_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::visible_indices_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::complete_models_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::resolution_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::inliers_threshold_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::normals_set_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::requires_normals_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::occlusion_thres_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::occlusion_cloud_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::zbuffer_self_occlusion_resolution_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::scene_cloud_;
      using faat_pcl::HypothesisVerification<ModelT, SceneT>::scene_sampled_indices_;

      template<typename PointT, typename NormalT>
        inline void
        extractEuclideanClustersSmooth (const typename pcl::PointCloud<PointT> &cloud, const typename pcl::PointCloud<NormalT> &normals, float tolerance,
                                        const typename pcl::search::Search<PointT>::Ptr &tree, std::vector<pcl::PointIndices> &clusters, double eps_angle,
                                        float curvature_threshold, unsigned int min_pts_per_cluster,
                                        unsigned int max_pts_per_cluster = (std::numeric_limits<int>::max) ())
        {

          if (tree->getInputCloud ()->points.size () != cloud.points.size ())
          {
            PCL_ERROR("[pcl::extractEuclideanClusters] Tree built for a different point cloud dataset\n");
            return;
          }
          if (cloud.points.size () != normals.points.size ())
          {
            PCL_ERROR("[pcl::extractEuclideanClusters] Number of points in the input point cloud different than normals!\n");
            return;
          }

          // Create a bool vector of processed point indices, and initialize it to false
          std::vector<bool> processed (cloud.points.size (), false);

          std::vector<int> nn_indices;
          std::vector<float> nn_distances;
          // Process all points in the indices vector
          int size = static_cast<int> (cloud.points.size ());
          for (int i = 0; i < size; ++i)
          {
            if (processed[i])
              continue;

            std::vector<unsigned int> seed_queue;
            int sq_idx = 0;
            seed_queue.push_back (i);

            processed[i] = true;

            while (sq_idx < static_cast<int> (seed_queue.size ()))
            {

              if (normals.points[seed_queue[sq_idx]].curvature > curvature_threshold)
              {
                sq_idx++;
                continue;
              }

              // Search for sq_idx
              if (!tree->radiusSearch (seed_queue[sq_idx], tolerance, nn_indices, nn_distances))
              {
                sq_idx++;
                continue;
              }

              for (size_t j = 1; j < nn_indices.size (); ++j) // nn_indices[0] should be sq_idx
              {
                if (processed[nn_indices[j]]) // Has this point been processed before ?
                  continue;

                if (normals.points[nn_indices[j]].curvature > curvature_threshold)
                {
                  continue;
                }

                //processed[nn_indices[j]] = true;
                // [-1;1]

                double dot_p = normals.points[seed_queue[sq_idx]].normal[0] * normals.points[nn_indices[j]].normal[0]
                    + normals.points[seed_queue[sq_idx]].normal[1] * normals.points[nn_indices[j]].normal[1] + normals.points[seed_queue[sq_idx]].normal[2]
                    * normals.points[nn_indices[j]].normal[2];

                if (fabs (acos (dot_p)) < eps_angle)
                {
                  processed[nn_indices[j]] = true;
                  seed_queue.push_back (nn_indices[j]);
                }
              }

              sq_idx++;
            }

            // If this queue is satisfactory, add to the clusters
            if (seed_queue.size () >= min_pts_per_cluster && seed_queue.size () <= max_pts_per_cluster)
            {
              pcl::PointIndices r;
              r.indices.resize (seed_queue.size ());
              for (size_t j = 0; j < seed_queue.size (); ++j)
                r.indices[j] = seed_queue[j];

              std::sort (r.indices.begin (), r.indices.end ());
              r.indices.erase (std::unique (r.indices.begin (), r.indices.end ()), r.indices.end ());
              clusters.push_back (r); // We could avoid a copy by working directly in the vector
            }
          }
        }

      virtual bool
      handlingNormals (boost::shared_ptr<RecognitionModel<ModelT> > & recog_model, int i, bool is_planar_model, int object_models_size);

      virtual bool
      addModel (int i, boost::shared_ptr<RecognitionModel<ModelT> > & recog_model);

      //Performs smooth segmentation of the scene cloud and compute the model cues
      virtual void
      initialize ();

      float regularizer_;
      pcl::PointCloud<pcl::Normal>::Ptr scene_normals_;
      bool ignore_color_even_if_exists_;
      std::vector<std::string> object_ids_;
      float color_sigma_;
      std::vector<float> extra_weights_;

      float
      getModelConstraintsValue (typename pcl::PointCloud<ModelT>::Ptr & cloud)
      {
        float under = 0;
        for (int i = 0; i < model_constraints_.size (); i++)
        {
          under += model_constraints_[i] (cloud) * model_constraints_weights_[i];
        }

        return under;
      }

      //class attributes
      bool use_super_voxels_;
      typedef typename pcl::NormalEstimation<SceneT, pcl::Normal> NormalEstimator_;
      pcl::PointCloud<pcl::PointXYZL>::Ptr clusters_cloud_;
      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr clusters_cloud_rgb_;
      pcl::PointCloud<pcl::Normal>::Ptr scene_normals_for_clutter_term_;

      std::vector<int> complete_cloud_occupancy_by_RM_;
      float res_occupancy_grid_;
      float w_occupied_multiple_cm_;

      std::vector<int> explained_by_RM_; //represents the points of scene_cloud_ that are explained by the recognition models
      std::vector<double> explained_by_RM_distance_weighted; //represents the points of scene_cloud_ that are explained by the recognition models
      std::vector<double> unexplained_by_RM_neighboorhods; //represents the points of scene_cloud_ that are not explained by the active hypotheses in the neighboorhod of the recognition models
      std::vector<boost::shared_ptr<RecognitionModel<ModelT> > > recognition_models_;
      //std::vector<size_t> indices_;
      std::vector<bool> valid_model_;

      float clutter_regularizer_;
      bool detect_clutter_;
      float radius_neighborhood_GO_;
      float radius_normals_;

      float previous_explained_value;
      float previous_duplicity_;
      int previous_duplicity_complete_models_;
      float previous_bad_info_;
      float previous_unexplained_;

      int max_iterations_; //max iterations without improvement
      SAModel<ModelT, SceneT> best_seen_;
      float initial_temp_;
      bool use_replace_moves_;

      //conflict graph stuff
      bool use_conflict_graph_;
      int n_cc_;
      std::vector<std::vector<int> > cc_;

      std::map<int, int> graph_id_model_map_;
      typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::shared_ptr<RecognitionModel<ModelT> > > Graph;
      Graph conflict_graph_;
      std::vector<std::vector<boost::shared_ptr<RecognitionModel<ModelT> > > > points_explained_by_rm_; //if inner size > 1, conflict

      //general model constraints, they get a model cloud and return the number of points that do not fulfill the condition
      std::vector<boost::function<int
      (const typename pcl::PointCloud<ModelT>::Ptr)> > model_constraints_;
      std::vector<float> model_constraints_weights_;
      int opt_type_;
      float active_hyp_penalty_;

      std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr> normals_for_visibility_;
      double eps_angle_threshold_;
      int min_points_;
      float curvature_threshold_;
      float cluster_tolerance_;

      float
      getOccupiedMultipleW ()
      {
        return w_occupied_multiple_cm_;
      }

      void
      setPreviousBadInfo (float f)
      {
        previous_bad_info_ = f;
      }

      float
      getPreviousBadInfo ()
      {
        return previous_bad_info_;
      }

      void
      setPreviousExplainedValue (float v)
      {
        previous_explained_value = v;
      }

      void
      setPreviousDuplicity (float v)
      {
        previous_duplicity_ = v;
      }

      void
      setPreviousDuplicityCM (int v)
      {
        previous_duplicity_complete_models_ = v;
      }

      void
      setPreviousUnexplainedValue (float v)
      {
        previous_unexplained_ = v;
      }

      float
      getPreviousUnexplainedValue ()
      {
        return previous_unexplained_;
      }

      float
      getExplainedValue ()
      {
        return previous_explained_value;
      }

      float
      getDuplicity ()
      {
        return previous_duplicity_;
      }

      int
      getDuplicityCM ()
      {
        return previous_duplicity_complete_models_;
      }

      float
      getHypPenalty ()
      {
        return active_hyp_penalty_;
      }

      float
      getExplainedByIndices (std::vector<int> & indices, std::vector<float> & explained_values, std::vector<double> & explained_by_RM,
                             std::vector<int> & indices_to_update_in_RM_local)
      {
        float v = 0;
        int indices_to_update_count = 0;
        for (size_t k = 0; k < indices.size (); k++)
        {
          if (explained_by_RM_[indices[k]] == 0)
          { //in X1, the point is not explained
            if (explained_by_RM[indices[k]] == 0)
            { //in X2, this is the single hypothesis explaining the point so far
              v += explained_values[k];
              indices_to_update_in_RM_local[indices_to_update_count] = k;
              indices_to_update_count++;
            }
            else
            {
              //in X2, there was a previous hypotheses explaining the point
              //if the previous hypothesis was better, then reject this hypothesis for this point
              if (explained_by_RM[indices[k]] >= explained_values[k])
              {

              }
              else
              {
                //add the difference
                v += explained_values[k] - explained_by_RM[indices[k]];
                indices_to_update_in_RM_local[indices_to_update_count] = k;
                indices_to_update_count++;
              }
            }
          }
        }

        indices_to_update_in_RM_local.resize (indices_to_update_count);
        return v;
      }

      void
      getExplainedByRM (std::vector<double> & explained_by_rm)
      {
        explained_by_rm = explained_by_RM_distance_weighted;
      }

      void
      updateUnexplainedVector (std::vector<int> & unexplained_, std::vector<float> & unexplained_distances, std::vector<double> & unexplained_by_RM,
                               std::vector<int> & explained, std::vector<int> & explained_by_RM, float val)
      {
        {

          float add_to_unexplained = 0.f;

          for (size_t i = 0; i < unexplained_.size (); i++)
          {

            bool prev_unexplained = (unexplained_by_RM[unexplained_[i]] > 0) && (explained_by_RM[unexplained_[i]] == 0);
            unexplained_by_RM[unexplained_[i]] += val * unexplained_distances[i];

            if (val < 0) //the hypothesis is being removed
            {
              if (prev_unexplained)
              {
                //decrease by 1
                add_to_unexplained -= unexplained_distances[i];
              }
            }
            else //the hypothesis is being added and unexplains unexplained_[i], so increase by 1 unless its explained by another hypothesis
            {
              if (explained_by_RM[unexplained_[i]] == 0)
                add_to_unexplained += unexplained_distances[i];
            }
          }

          for (size_t i = 0; i < explained.size (); i++)
          {
            if (val < 0)
            {
              //the hypothesis is being removed, check that there are no points that become unexplained and have clutter unexplained hypotheses
              if ((explained_by_RM[explained[i]] == 0) && (unexplained_by_RM[explained[i]] > 0))
              {
                add_to_unexplained += unexplained_by_RM[explained[i]]; //the points become unexplained
              }
            }
            else
            {
              //std::cout << "being added..." << add_to_unexplained << " " << unexplained_by_RM[explained[i]] << std::endl;
              if ((explained_by_RM[explained[i]] == 1) && (unexplained_by_RM[explained[i]] > 0))
              { //the only hypothesis explaining that point
                add_to_unexplained -= unexplained_by_RM[explained[i]]; //the points are not unexplained any longer because this hypothesis explains them
              }
            }
          }

          //std::cout << add_to_unexplained << std::endl;
          previous_unexplained_ += add_to_unexplained;
        }
      }

      void
      updateExplainedVector (std::vector<int> & vec, std::vector<float> & vec_float, std::vector<int> & explained_,
                             std::vector<double> & explained_by_RM_distance_weighted, float sign);
      /*{
        float add_to_explained = 0.f;
        float add_to_duplicity_ = 0;

        for (size_t i = 0; i < vec.size (); i++)
        {
          bool prev_dup = explained_[vec[i]] > 1;
          bool prev_explained = explained_[vec[i]] == 1;
          float prev_explained_value = explained_by_RM_distance_weighted[vec[i]];

          explained_[vec[i]] += static_cast<int> (sign);
          explained_by_RM_distance_weighted[vec[i]] += vec_float[i] * sign;

          //add_to_explained += vec_float[i] * sign;
          if (explained_[vec[i]] == 1 && !prev_explained)
          {
            if (sign > 0)
            {
              add_to_explained += vec_float[i];
            }
            else
            {
              add_to_explained += explained_by_RM_distance_weighted[vec[i]];
            }
          }

          //hypotheses being removed, now the point is not explained anymore and was explained before by this hypothesis
          if ((sign < 0) && (explained_[vec[i]] == 0) && prev_explained)
          {
            //assert(prev_explained_value == vec_float[i]);
            add_to_explained -= prev_explained_value;
          }

          //this hypothesis was added and now the point is not explained anymore, remove previous value
          if ((sign > 0) && (explained_[vec[i]] == 2) && prev_explained)
            add_to_explained -= prev_explained_value;

          if ((explained_[vec[i]] > 1) && prev_dup)
          { //its still a duplicate
            add_to_duplicity_ += vec_float[i] * static_cast<int> (sign) / 2.f; //so, just add or remove one
          }
          else if ((explained_[vec[i]] == 1) && prev_dup)
          { //if was duplicate before, now its not, remove 2, we are removing the hypothesis
            add_to_duplicity_ -= prev_explained_value / 2.f; //explained_by_RM_distance_weighted[vec[i]];
          }
          else if ((explained_[vec[i]] > 1) && !prev_dup)
          { //it was not a duplicate but it is now, add 2, we are adding a conflicting hypothesis for the point
            add_to_duplicity_ += explained_by_RM_distance_weighted[vec[i]]  / 2.f;
          }
        }

        //update explained and duplicity values...
        previous_explained_value += add_to_explained;
        previous_duplicity_ += add_to_duplicity_;
      }*/

      void
      updateCMDuplicity (std::vector<int> & vec, std::vector<int> & occupancy_vec, float sign);
      /*{
        int add_to_duplicity_ = 0;
        for (size_t i = 0; i < vec.size (); i++)
        {
          bool prev_dup = occupancy_vec[vec[i]] > 1;
          occupancy_vec[vec[i]] += static_cast<int> (sign);
          if ((occupancy_vec[vec[i]] > 1) && prev_dup)
          { //its still a duplicate, we are adding
            add_to_duplicity_ += static_cast<int> (sign); //so, just add or remove one
          }
          else if ((occupancy_vec[vec[i]] == 1) && prev_dup)
          { //if was duplicate before, now its not, remove 2, we are removing the hypothesis
            add_to_duplicity_ -= 2;
          }
          else if ((occupancy_vec[vec[i]] > 1) && !prev_dup)
          { //it was not a duplicate but it is now, add 2, we are adding a conflicting hypothesis for the point
            add_to_duplicity_ += 2;
          }
        }

        previous_duplicity_complete_models_ += add_to_duplicity_;
      }*/

      float
      getTotalExplainedInformation (std::vector<int> & explained_, std::vector<double> & explained_by_RM_distance_weighted, float * duplicity_);
      /*{
        float explained_info = 0;
        float duplicity = 0;

        for (size_t i = 0; i < explained_.size (); i++)
        {
          //if (explained_[i] > 0)
          if (explained_[i] == 1) //only counts points that are explained once
          {
            //explained_info += explained_by_RM_distance_weighted[i] / 2.f; //what is the magic division by 2?
            explained_info += explained_by_RM_distance_weighted[i];
          }
          if (explained_[i] > 1)
          {
            //duplicity += explained_by_RM_distance_weighted[i];
            duplicity += explained_by_RM_distance_weighted[i] / 2.f;
          }
        }

        *duplicity_ = duplicity;

        return explained_info;
      }*/

      float
      getTotalBadInformation (std::vector<boost::shared_ptr<RecognitionModel<ModelT> > > & recog_models)
      {
        float bad_info = 0;
        for (size_t i = 0; i < recog_models.size (); i++)
          bad_info += recog_models[i]->outliers_weight_ * static_cast<float> (recog_models[i]->bad_information_);

        return bad_info;
      }

      float
      getUnexplainedInformationInNeighborhood (std::vector<double> & unexplained, std::vector<int> & explained)
      {
        float unexplained_sum = 0.f;
        for (size_t i = 0; i < unexplained.size (); i++)
        {
          if (unexplained[i] > 0 && explained[i] == 0)
            unexplained_sum += unexplained[i];
        }

        return unexplained_sum;
      }

      float
      getModelConstraintsValueForActiveSolution (const std::vector<bool> & active)
      {
        float bad_info = 0;
        for (size_t i = 0; i < recognition_models_.size (); i++)
        {
          if (active[i])
            bad_info += recognition_models_[i]->model_constraints_value_;
        }

        return bad_info;
      }

      mets::gol_type
      evaluateSolution (const std::vector<bool> & active, int changed);

      /*bool
       addModel(typename pcl::PointCloud<ModelT>::ConstPtr & model,
       typename pcl::PointCloud<ModelT>::ConstPtr & complete_model,
       pcl::PointCloud<pcl::Normal>::ConstPtr & model_normals,
       boost::shared_ptr<RecognitionModel> & recog_model,
       std::vector<int> & visible_indices,
       float extra_weight = 1.f);*/

      void
      computeClutterCue (boost::shared_ptr<RecognitionModel<ModelT> > & recog_model);

      void
      SAOptimize (std::vector<int> & cc_indices, std::vector<bool> & sub_solution);

      void
      fill_structures (std::vector<int> & cc_indices, std::vector<bool> & sub_solution, SAModel<ModelT, SceneT> & model);

      void
      clear_structures ();

      float
      countActiveHypotheses (const std::vector<bool> & sol);

      boost::shared_ptr<CostFunctionLogger<ModelT,SceneT> > cost_logger_;
      bool initial_status_;

      void
      computeYUVHistogram (std::vector<Eigen::Vector3f> & yuv_values, Eigen::VectorXf & histogram);

      void
      computeRGBHistograms (std::vector<Eigen::Vector3f> & rgb_values, Eigen::MatrixXf & rgb,
                               int dim = 3, float min = 0.f, float max = 255.f, bool soft = false);

      void
      specifyRGBHistograms (Eigen::MatrixXf & src, Eigen::MatrixXf & dst, Eigen::MatrixXf & lookup, int dim = 3);

      void
      computeHueHistogram (std::vector<Eigen::Vector3f> & hsv_values, Eigen::VectorXf & histogram);

      void
      computeGSHistogram (std::vector<float> & hsv_values, Eigen::MatrixXf & histogram);

      void
      convertToHSV (int ri, int gi, int bi, Eigen::Vector3f & hsv)
      {
        float r = ri / 255.f;
        float g = gi / 255.f;
        float b = bi / 255.f;
        //std::cout << "rgb:" << r << " " << g << " " << b << std::endl;
        float max_color = std::max (r, std::max (g, b));
        float min_color = std::min (r, std::min (g, b));
        float h, s, v;
        h = 0;
        if (min_color == max_color)
        {
          h = 0;
        }
        else
        {
          if (max_color == r)
            h = 60.f * (0 + (g - b) / (max_color - min_color));
          else if (max_color == g)
            h = 60.f * (2 + (b - r) / (max_color - min_color));
          else if (max_color == b)
            h = 60.f * (4 + (r - g) / (max_color - min_color));

          if (h < 0)
            h += 360.f;
        }

        hsv[0] = h / 360.f;
        if (max_color == 0.f)
          hsv[1] = 0.f;
        else
          hsv[1] = (max_color - min_color) / max_color;

        hsv[2] = (max_color - min_color) / 2.f;
      }

      void computeClutterCueGPU();

      std::vector<faat_pcl::PlaneModel<ModelT> > planar_models_;
      std::map<int, int> model_to_planar_model_;
      pcl::PointCloud<pcl::PointXYZ>::Ptr occ_edges_;
      bool occ_edges_available_;
      bool use_histogram_specification_;
      typename boost::shared_ptr<pcl::octree::OctreePointCloudSearch<SceneT> > octree_scene_downsampled_;
      boost::shared_ptr<pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> > octree_occ_edges_;
      int min_contribution_;
    public:
      GlobalHypothesesVerification_1 () :
        faat_pcl::HypothesisVerification<ModelT, SceneT> ()
      {
        resolution_ = 0.005f;
        max_iterations_ = 5000;
        regularizer_ = 1.f;
        radius_normals_ = 0.01f;
        initial_temp_ = 1000;
        detect_clutter_ = true;
        radius_neighborhood_GO_ = 0.03f;
        clutter_regularizer_ = 5.f;
        res_occupancy_grid_ = 0.005f;
        w_occupied_multiple_cm_ = 2.f;
        use_conflict_graph_ = false;
        ignore_color_even_if_exists_ = true;
        color_sigma_ = 50.f;
        opt_type_ = 2;
        use_replace_moves_ = true;
        active_hyp_penalty_ = 0.f;
        requires_normals_ = false;
        initial_status_ = false;

        eps_angle_threshold_ = 0.25;
        min_points_ = 20;
        curvature_threshold_ = 0.04f;
        cluster_tolerance_ = 0.015f;
        use_super_voxels_ = false;
        occ_edges_available_ = false;
        use_histogram_specification_ = false;
        min_contribution_ = 0;
      }

      void setDuplicityCMWeight(float w)
      {
          w_occupied_multiple_cm_ = w;
      }

      void setHistogramSpecification(bool b)
      {
          use_histogram_specification_ = b;
      }

      void setOcclusionEdges(pcl::PointCloud<pcl::PointXYZ>::Ptr & occ_edges)
      {
          occ_edges_ = occ_edges;
          occ_edges_available_ = true;
      }

      void setNormalsForClutterTerm(pcl::PointCloud<pcl::Normal>::Ptr & normals)
      {
          scene_normals_for_clutter_term_ = normals;
      }

      void setUseSuperVoxels(bool use)
      {
        use_super_voxels_ = use;
      }
      void addPlanarModels(std::vector<faat_pcl::PlaneModel<ModelT> > & models);

      void
      setSmoothSegParameters (float t_eps, float curv_t, float dist_t, int min_points = 20)
      {
        eps_angle_threshold_ = t_eps;
        min_points_ = min_points;
        curvature_threshold_ = curv_t;
        cluster_tolerance_ = dist_t;
      }

      void
      setObjectIds (std::vector<std::string> & ids)
      {
        object_ids_ = ids;
      }

      void
      writeToLog (std::ofstream & of, bool all_costs_ = false)
      {
        cost_logger_->writeToLog (of);
        if (all_costs_)
        {
          cost_logger_->writeEachCostToLog (of);
        }
      }

      void
      setHypPenalty (float p)
      {
        active_hyp_penalty_ = p;
      }

      void setMinContribution(int min)
      {
          min_contribution_ = min;
      }

      void
      setInitialStatus (bool b)
      {
        initial_status_ = b;
      }

      /*void logCosts() {
       cost_logger_.reset(new CostFunctionLogger());
       }*/

      pcl::PointCloud<pcl::PointXYZL>::Ptr
      getSmoothClusters ()
      {
        return clusters_cloud_;
      }

      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr
      getSmoothClustersRGBCloud ()
      {
        return clusters_cloud_rgb_;
      }

      float
      getResolution ()
      {
        return resolution_;
      }

      void
      setRequiresNormals (bool b)
      {
        requires_normals_ = b;
      }

      void
      setUseReplaceMoves (bool u)
      {
        use_replace_moves_ = u;
      }

      void
      setOptimizerType (int t)
      {
        opt_type_ = t;
      }

      void
      verify ();

      void
      addModelConstraint (boost::function<int
      (const typename pcl::PointCloud<ModelT>::Ptr)> & f, float weight = 1.f)
      {
        model_constraints_.push_back (f);
        model_constraints_weights_.push_back (weight);
      }

      void
      clearModelConstraints ()
      {
        model_constraints_.clear ();
        model_constraints_weights_.clear ();
      }

      void
      setIgnoreColor (bool i)
      {
        ignore_color_even_if_exists_ = i;
      }

      void
      setColorSigma (float s)
      {
        color_sigma_ = s;
      }

      void
      setUseConflictGraph (bool u)
      {
        use_conflict_graph_ = u;
      }

      void
      setRadiusNormals (float r)
      {
        radius_normals_ = r;
      }

      void
      setMaxIterations (int i)
      {
        max_iterations_ = i;
      }

      void
      setInitialTemp (float t)
      {
        initial_temp_ = t;
      }

      void
      setRegularizer (float r)
      {
        regularizer_ = r;
        //w_occupied_multiple_cm_ = regularizer_;
      }

      void
      setRadiusClutter (float r)
      {
        radius_neighborhood_GO_ = r;
      }

      void
      setClutterRegularizer (float cr)
      {
        clutter_regularizer_ = cr;
      }

      void
      setDetectClutter (bool d)
      {
        detect_clutter_ = d;
      }

      //Same length as the recognition models
      void
      setExtraWeightVectorForInliers (std::vector<float> & weights)
      {
        extra_weights_.clear ();
        extra_weights_ = weights;
      }

      void
      addNormalsForVisibility (std::vector<pcl::PointCloud<pcl::Normal>::ConstPtr> & complete_normals_for_visibility)
      {
        normals_for_visibility_ = complete_normals_for_visibility;
      }

      void
      getOutliersForAcceptedModels(std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > & outliers_cloud);

      void
      getOutliersForAcceptedModels(std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > & outliers_cloud_color,
                                   std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > & outliers_cloud_3d);

    };
}

#endif //FAAT_PCL_GO_1_H_
