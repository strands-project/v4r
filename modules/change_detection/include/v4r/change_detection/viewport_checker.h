/*
 * viewport_checker.hpp
 *
 *  Created on: 10 Aug 2015
 *      Author: martin
 */

#ifndef VIEWPORT_CHECKER_HPP_
#define VIEWPORT_CHECKER_HPP_

#include <vector>

#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/common/distances.h>

#include <v4r/core/macros.h>

namespace v4r {

template <class PointType>
class V4R_EXPORTS ViewVolume {
public:
	/**
	 * angles in rad
	 */
	ViewVolume(double min_dist_, double max_dist_, double h_angle_, double v_angle_,
			const Eigen::Affine3f &sensor_pose_, double tolerance_) :
		min_dist(min_dist_), max_dist(max_dist_),
		max_sin_h_angle(sin(h_angle_/2 - tolerance_)),
		max_sin_v_angle(sin(v_angle_/2 - tolerance_)),
		sensor_pose(sensor_pose_) {
	}

	int computeVisible(const typename pcl::PointCloud<PointType>::Ptr input,
			std::vector<bool> &mask) {
		pcl::PointCloud<PointType> input_transformed;
		pcl::transformPointCloud(*input, input_transformed, sensor_pose.inverse());

		int visible_count = 0;
		assert(input->size() == mask.size());
		for(int i = 0; i < input_transformed.size(); i++) {
			bool isIn = in(input_transformed[i]);
			if(isIn) {
				visible_count++;
			}
			if(!mask[i]) {
				mask[i] = isIn;
			}
		}
		return visible_count;
	}

	static ViewVolume<PointType> ofXtion(const Eigen::Affine3f &sensor_pose, double tolerance = 5.0/*deg*/) {
		static double degToRad = M_PI / 180.0;
		return ViewVolume<PointType>(0.5, 3.5, 58*degToRad, 45*degToRad,
				sensor_pose, tolerance*degToRad);
	}

	pcl::PointCloud<pcl::PointXYZ> getBorders() const {
		double x_near = max_sin_h_angle * min_dist;
		double y_near = max_sin_v_angle * min_dist;
		double x_far = max_sin_h_angle * max_dist;
		double y_far = max_sin_v_angle * max_dist;

		pcl::PointCloud<pcl::PointXYZ> borders;
		borders.push_back(pcl::PointXYZ(x_near, y_near, min_dist));
		borders.push_back(pcl::PointXYZ(-x_near, y_near, min_dist));
		borders.push_back(pcl::PointXYZ(-x_near, -y_near, min_dist));
		borders.push_back(pcl::PointXYZ(x_near, -y_near, min_dist));

		borders.push_back(pcl::PointXYZ(x_far, y_far, max_dist));
		borders.push_back(pcl::PointXYZ(-x_far, y_far, max_dist));
		borders.push_back(pcl::PointXYZ(-x_far, -y_far, max_dist));
		borders.push_back(pcl::PointXYZ(x_far, -y_far, max_dist));

		pcl::transformPointCloud(borders, borders, sensor_pose);
		return borders;
	}

protected:
	bool in(const PointType &pt) {
		double sin_h_angle = fabs(pt.x) / sqrt(pt.x*pt.x + pt.z*pt.z);
		double sin_v_angle = fabs(pt.y) / sqrt(pt.y*pt.y + pt.z*pt.z);

		return (pt.z > min_dist) && (pt.z < max_dist) &&
				(sin_h_angle < max_sin_h_angle) && (sin_v_angle < max_sin_v_angle);
	}

private:
	double min_dist;
	double max_dist;
	double max_sin_h_angle;
	double max_sin_v_angle;
	Eigen::Affine3f sensor_pose;
};

template <class PointType>
class ViewportChecker {
public:
	void add(const ViewVolume<PointType> volume) {
		volumes.push_back(volume);
	}

	void getVisibles(const typename pcl::PointCloud<PointType>::Ptr input,
			typename pcl::PointCloud<PointType>::Ptr visible,
			typename pcl::PointCloud<PointType>::Ptr nonVisible) {
		std::vector<bool> mask(input->size(), false);
		for(typename std::vector< ViewVolume<PointType> >::iterator vol = volumes.begin();
				vol < volumes.end(); vol++) {
			vol->computeVisible(input, mask);
		}
		for(int i = 0; i < mask.size(); i++) {
			if(mask[i]) {
				visible->push_back(input->at(i));
			} else {
				nonVisible->push_back(input->at(i));
			}
		}
	}

private:
	std::vector< ViewVolume<PointType> > volumes;
};

}

#endif /* VIEWPORT_CHECKER_HPP_ */
