/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2012, Willow Garage, Inc.
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
 *   * Neither the name of Willow Garage, Inc. nor the names of its
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
 * $Id$
 *
 */

#ifndef FAAT_PCL_RECOGNITION_SI_GEOMETRIC_CONSISTENCY_IMPL_H_
#define FAAT_PCL_RECOGNITION_SI_GEOMETRIC_CONSISTENCY_IMPL_H_

#include <faat_pcl/recognition/cg/si_geometric_consistency.h>
#include <pcl/registration/correspondence_types.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/common/io.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
gcsiCorrespSorter (pcl::Correspondence i, pcl::Correspondence j) 
{ 
  return (i.distance < j.distance); 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointModelT, typename PointSceneT> void
pcl::SIGeometricConsistencyGrouping<PointModelT, PointSceneT>::clusterCorrespondences (std::vector<Correspondences> &model_instances)
{
  model_instances.clear ();
  found_transformations_.clear ();
  corr_group_scale_.clear();

  if (!model_scene_corrs_)
  {
    PCL_ERROR(
      "[pcl::SIGeometricConsistencyGrouping::clusterCorrespondences()] Error! Correspondences not set, please set them before calling again this function.\n");
    return;
  }

  CorrespondencesPtr sorted_corrs (new Correspondences (*model_scene_corrs_));

  std::sort (sorted_corrs->begin (), sorted_corrs->end (), gcsiCorrespSorter);

  model_scene_corrs_ = sorted_corrs;

  std::vector<int> consensus_set;
  std::vector<bool> taken_corresps (model_scene_corrs_->size (), false);

  Eigen::Vector3f dist_ref, dist_trg;

  //temp copy of scene cloud with the type cast to ModelT in order to use Ransac
  PointCloudPtr temp_scene_cloud_ptr (new PointCloud ());
  pcl::copyPointCloud<PointSceneT, PointModelT> (*scene_, *temp_scene_cloud_ptr);

  pcl::registration::CorrespondenceRejectorSampleConsensus<PointModelT> corr_rejector;
  corr_rejector.setMaximumIterations (10000);
  corr_rejector.setInlierThreshold (gc_size_);
  corr_rejector.setInputSource(input_);
  corr_rejector.setInputTarget (temp_scene_cloud_ptr);

  for (size_t i = 0; i < model_scene_corrs_->size (); ++i)
  {
    if (taken_corresps[i])
      continue;

	for (size_t j = 0; j < model_scene_corrs_->size (); ++j)
    { 

		if ( j == i || taken_corresps[j])
			continue;

		consensus_set.clear ();
		consensus_set.push_back (static_cast<int> (i));
		consensus_set.push_back (static_cast<int> (j));

		//assumption: scale is computed as the ratio of the model over the scene
		int scene_index_i = model_scene_corrs_->at (i).index_match;
		int model_index_i = model_scene_corrs_->at (i).index_query;
		int scene_index_j = model_scene_corrs_->at (j).index_match;
		int model_index_j = model_scene_corrs_->at (j).index_query;

		const Eigen::Vector3f& scene_point_i = scene_->at (scene_index_i).getVector3fMap ();
		const Eigen::Vector3f& model_point_i = input_->at (model_index_i).getVector3fMap ();
		const Eigen::Vector3f& scene_point_j = scene_->at (scene_index_j).getVector3fMap ();
		const Eigen::Vector3f& model_point_j = input_->at (model_index_j).getVector3fMap ();

		double scale = (model_point_i - model_point_j) / (scene_point_i - scene_point_j);

		for (size_t k = 0; k < model_scene_corrs_->size (); ++k)
		{

			if ( k==i || k==j || takenCorresps[k])
				continue;
			
			//Let's check if k fits into the current consensus set
			bool is_a_good_candidate = true;
			
			for (size_t c = 0; c < consensus_set.size (); ++c)
			{
			
				int scene_index_c = model_scene_corrs_->at (consensus_set[c]).index_match;
				int model_index_c = model_scene_corrs_->at (consensus_set[c]).index_query;
				int scene_index_k = model_scene_corrs_->at (k).index_match;
				int model_index_k = model_scene_corrs_->at (k).index_query;
			
				const Eigen::Vector3f& scene_point_c = scene_->at (scene_index_c).getVector3fMap ();
				const Eigen::Vector3f& model_point_c = input_->at (model_index_c).getVector3fMap ();
				const Eigen::Vector3f& scene_point_k = scene_->at (scene_index_k).getVector3fMap ();
				const Eigen::Vector3f& model_point_k = input_->at (model_index_k).getVector3fMap ();

				double distance = fabs( (model_point_c - model_point_k) - scale * (scene_point_c - scene_point_k));

				if (distance > gc_size_)
				{
					is_a_good_candidate = false;
					break;
				}
			} //consensus_set loop

			if (is_a_good_candidate)
				consensus_set.push_back (static_cast<int> (k));
		} // k loop

		if (static_cast<int> (consensus_set.size ()) > gc_threshold_)
		{
			Correspondences temp_corrs, filtered_corrs;
			for (size_t k = 0; k < consensus_set.size (); k++)
			{
				temp_corrs.push_back (model_scene_corrs_->at (consensus_set[k]));
				taken_corresps[ consensus_set[k] ] = true;
			}
      
			//ransac filtering
			corr_rejector.getRemainingCorrespondences (temp_corrs, filtered_corrs);
			//save transformations for recognize
			found_transformations_.push_back (corr_rejector.getBestTransformation ());

			//save the scale associated to current cluster 
			corr_group_scale_.push_back (scale);

			model_instances.push_back (filtered_corrs);
			break; //skip the current j-th loop and the current i-th element as the current i-th element has been just taken
		}
	} // j loop
  }//i loop
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointModelT, typename PointSceneT> bool
pcl::SIGeometricConsistencyGrouping<PointModelT, PointSceneT>::recognize (
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transformations)
{
  std::vector<pcl::Correspondences> model_instances;
  return (this->recognize (transformations, model_instances));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointModelT, typename PointSceneT> bool
pcl::GeometricConsistencyGrouping<PointModelT, PointSceneT>::recognize (
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transformations, std::vector<pcl::Correspondences> &clustered_corrs)
{
  transformations.clear ();
  if (!this->initCompute ())
  {
    PCL_ERROR(
      "[pcl::GeometricConsistencyGrouping::recognize()] Error! Model cloud or Scene cloud not set, please set them before calling again this function.\n");
    return (false);
  }

  clusterCorrespondences (clustered_corrs);

  transformations = found_transformations_;

  this->deinitCompute ();
  return (true);
}


#endif // FAAT_PCL_RECOGNITION_SI_GEOMETRIC_CONSISTENCY_IMPL_H_
