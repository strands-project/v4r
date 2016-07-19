#ifndef _V4R_CHANGE_DETECTION_H_
#define _V4R_CHANGE_DETECTION_H_

#include <v4r/core/macros.h>

#include <pcl/common/eigen.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/segment_differences.h>

namespace v4r {

template<class PointType>
class V4R_EXPORTS ChangeDetector {

public:
	typedef pcl::PointCloud<PointType> Cloud;
	typedef typename Cloud::Ptr CloudPtr;
    typedef pcl::search::KdTree<PointType> Tree;

    ChangeDetector() :
    	added(new Cloud()), removed(new Cloud) {
    }

	void detect(const CloudPtr source, const CloudPtr target, const Eigen::Affine3f sensor_pose,
			float diff_tolerance = DEFAULT_PARAMETERS.cloud_difference_tolerance);

	bool isObjectRemoved(CloudPtr object_cloud) const;

	static float computePlanarity(const CloudPtr input_cloud);

	std::vector<typename pcl::PointCloud<PointType>::Ptr>
	static clusterPointCloud(CloudPtr input_cloud, double tolerance,
			int min_cluster_size, int max_cluster_size);

	static bool hasPointInRadius(const PointType &pt, typename Tree::Ptr tree, float distance) {
		// We're interested in a single neighbor only
		std::vector<int> nn_indices(1);
		std::vector<float> nn_distances(1);

		if (!pcl::isFinite(pt)) {
			return false;
		}
		if (tree->radiusSearch(pt, distance, nn_indices, nn_distances, 1) < 1) {
			return false;
		} else {
			return true;
		}
	}

	/**
	 * diff = A \ B
	 * indices = indexes of preserved points from A
	 */
	static void difference(const CloudPtr A, const CloudPtr B, CloudPtr &diff,
			std::vector<int> &indices, float tolerance = DEFAULT_PARAMETERS.cloud_difference_tolerance) {
		if(A->empty()) {
			return;
		}
		if(B->empty()) {
			pcl::copyPointCloud(*A, *diff);
			for(size_t i = 0; i < A->size(); i++) {
				indices.push_back(i);
			}
			return;
		}

		typename Tree::Ptr tree(new pcl::search::KdTree<PointType>);
		tree->setInputCloud(B);

		// Iterate through the source data set
		for (int i = 0; i < static_cast<int>(A->points.size()); ++i) {
			if (pcl::isFinite(A->points[i]) &&
					!hasPointInRadius(A->points[i], tree, tolerance)) {
				indices.push_back(i);
			}
		}

		diff->points.resize(indices.size());
		diff->header = A->header;
		diff->width = static_cast<uint32_t>(indices.size());
		diff->height = 1;
		diff->is_dense = true;
		copyPointCloud(*A, indices, *diff);
	}

	static void difference(const CloudPtr A, const CloudPtr B, CloudPtr diff) {
		std::vector<int> indices;
		difference(A, B, diff, indices);
	}

	static void removePointsFrom(const CloudPtr cloud, const CloudPtr toBeRemoved);

	static int overlapingPoints(const CloudPtr train, const CloudPtr query,
			float tolerance = DEFAULT_PARAMETERS.cloud_difference_tolerance);

	const CloudPtr getAdded() const {
		return added;
	}

	const CloudPtr getRemoved() const {
		return removed;
	}

	static CloudPtr getNonplanarClusters(CloudPtr removed_points) {
		std::vector<CloudPtr> clusters = clusterPointCloud(removed_points,
				DEFAULT_PARAMETERS.maximal_intra_cluster_dist, DEFAULT_PARAMETERS.min_cluster_points,
				DEFAULT_PARAMETERS.max_cluster_points);
		CloudPtr nonplanarClusters(new Cloud());
		for(typename  std::vector<CloudPtr>::iterator c = clusters.begin();
				c < clusters.end(); c++) {
			if(computePlanarity(*c) < DEFAULT_PARAMETERS.planarity_threshold) {
				*nonplanarClusters += **c;
			}
		}
		return nonplanarClusters;
	}

	static CloudPtr removalSupport(CloudPtr removed_points, CloudPtr &object_cloud,
			float tolerance = DEFAULT_PARAMETERS.cloud_difference_tolerance,
			float planarity_threshold = DEFAULT_PARAMETERS.planarity_threshold) {
		CloudPtr support(new Cloud());
		std::vector<CloudPtr> removed_clusters = clusterPointCloud(removed_points,
				DEFAULT_PARAMETERS.maximal_intra_cluster_dist, DEFAULT_PARAMETERS.min_cluster_points,
				DEFAULT_PARAMETERS.max_cluster_points);
		for(typename  std::vector<CloudPtr>::iterator c = removed_clusters.begin();
				c < removed_clusters.end(); c++) {

			// only non planar cluster considered:
			if(planarity_threshold > 1.0 || computePlanarity(*c) < planarity_threshold) {
				if(overlapingPoints(object_cloud, *c, tolerance) > (DEFAULT_PARAMETERS.min_removal_overlap * (*c)->size())) {
					*support += **c;
				}
			}
		}
		return support;
	}

protected:

	CloudPtr removalSupport(CloudPtr &object_cloud) const;

private:

	CloudPtr added, removed;

public:

	class Parameters {
	public:
		Parameters(int occlusion_checker_bins_ = 180,
				int min_cluster_points_ = 50, int max_cluster_points_ = 1000000,
				float maximal_intra_cluster_dist_ = 0.03,
				float planarity_threshold_ = 0.9,
				float min_removal_overlap_ = 0.8,
				float cloud_difference_tolerance_ = 0.01) :

				occlusion_checker_bins(occlusion_checker_bins_),
				min_cluster_points(min_cluster_points_),
				max_cluster_points(max_cluster_points_),
				maximal_intra_cluster_dist(maximal_intra_cluster_dist_),
				planarity_threshold(planarity_threshold_),
				min_removal_overlap(min_removal_overlap_),
				cloud_difference_tolerance(cloud_difference_tolerance_) {
		}

		int occlusion_checker_bins;
		int min_cluster_points;
		int max_cluster_points;
		float maximal_intra_cluster_dist;
		float planarity_threshold;
		float min_removal_overlap;
		float cloud_difference_tolerance;
	};

	static const Parameters DEFAULT_PARAMETERS;
	Parameters params;
};

}

#endif
