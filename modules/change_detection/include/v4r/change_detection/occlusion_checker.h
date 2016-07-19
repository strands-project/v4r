#ifndef __OCCLUSION_CHECKER__H
#define __OCCLUSION_CHECKER__H

#include <stdio.h>
#include <iosfwd>
#include <stdlib.h>
#include <string>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/centroid.h>
#include <pcl/registration/transforms.h>
#include <pcl/segmentation/segment_differences.h>

#include <boost/date_time/posix_time/posix_time.hpp>

namespace v4r {

template <class PointType>
class OcclusionChecker {
public:

    typedef pcl::PointCloud<PointType> Cloud;
    typedef typename Cloud::Ptr CloudPtr;

    struct occluded_points
    {
        CloudPtr toBeAdded;
        CloudPtr toBeRemoved;
    };

    struct occlusion_results {
        CloudPtr occluded;
        CloudPtr nonOccluded;
    };

    OcclusionChecker() :
    	viewpoint(Eigen::Vector3f(0.0,0.0,0.0)),
    	numberOfBins(360),
    	tolerance(0.001) {
    }

    void setViewpoint(Eigen::Vector3f origin)
    {
        viewpoint = origin;
    }

	/**
	 * Assuming the both points of scene and points of obstacles are already registered.
	 */
	occlusion_results checkOcclusions(CloudPtr scene, CloudPtr obstacles) {
		occlusion_results result;
		result.occluded = CloudPtr(new Cloud());
		result.nonOccluded = CloudPtr(new Cloud());

		// Transformation to origin -> needed for spherical projection
		Eigen::Affine3f transformToOrigin = Eigen::Affine3f(Eigen::Translation3f(viewpoint)).inverse();

		// init spherical map
		double thetaphi[numberOfBins][numberOfBins];
		for (size_t j = 0; j < numberOfBins; j++) {
			for (size_t k = 0; k < numberOfBins; k++) {
				thetaphi[j][k] = INFINITY;
			}
		}

		// fill spherical map by obstacles
		CloudPtr obstacles_transformed(new Cloud);
		pcl::transformPointCloud(*obstacles, *obstacles_transformed, transformToOrigin);
		for (size_t j = 0; j < obstacles_transformed->size(); j++) {
			if(!isPointValid(obstacles_transformed->at(j))) {
				continue;
			}
			// convert to spherical coordinates
			int thetabin, phibin;
			double r;
			rPhiThetaBins(obstacles_transformed->at(j), r, phibin, thetabin);
			thetaphi[thetabin][phibin] = std::min(r, thetaphi[thetabin][phibin]);
		}

		// transform cluster to Origin
		CloudPtr scene_transformed(new Cloud);
		pcl::transformPointCloud(*scene, *scene_transformed, transformToOrigin);
		for (size_t k = 0; k < scene->size(); k++) {
			if(!isPointValid(scene_transformed->at(k))) {
				continue;
			}
			// take spherical projection
			int thetabin, phibin;
			double r;
			rPhiThetaBins(scene_transformed->at(k), r, phibin, thetabin);

			if (thetaphi[thetabin][phibin] < (r + tolerance)) {
				result.occluded->push_back(scene->at(k));
			} else {
				result.nonOccluded->push_back(scene->at(k));
			}
		}

		return result;
	}

    void rPhiThetaBins(const PointType &pt, double &r, int &phi_bin, int &theta_bin) {
        r = sqrt(pow(pt.x,2) + pow(pt.y,2) + pow(pt.z,2));
        double theta = M_PI + acos(pt.z/r);
        double phi = M_PI + atan2(pt.y,pt.x);
        theta_bin = (int)((theta/(2*M_PI)) * numberOfBins);
        phi_bin = (int)((phi/(2*M_PI)) * numberOfBins);

        theta_bin = std::min(std::max(0, theta_bin), numberOfBins-1);
        phi_bin = std::min(std::max(0, phi_bin), numberOfBins-1);
    }

	void setNumberOfBins(int bins) {
		this->numberOfBins = bins;
	}

	bool isPointValid(const PointType &pt) {
		return pcl::isFinite(pt) && !std::isnan(pt.x) && !std::isnan(pt.y) && !std::isnan(pt.z);
	}

private:
    Eigen::Vector3f         viewpoint;
    int 					numberOfBins;
    float					tolerance;
};

}

#endif
