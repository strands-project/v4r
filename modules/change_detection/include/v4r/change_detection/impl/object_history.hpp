/*
 * ObjectHistory.hpp
 *
 *  Created on: 13 Aug 2015
 *      Author: martin
 */

#ifndef OBJECTHISTORY_HPP_
#define OBJECTHISTORY_HPP_

#include <v4r/change_detection/object_history.h>

namespace v4r {

template<class PointType>
int ObjectsHistory<PointType>::time = 0;

template<class PointType>
void ObjectsHistory<PointType>::add(const std::vector< ObjectDetection<PointType> > &detections) {
	for(typename std::vector< ObjectDetection<PointType> >::const_iterator d = detections.begin();
			d < detections.end(); d++) {
		ObjectLabel label = d->getClass();
		typename Db::iterator entry = db.find(label);
		if(entry == db.end()) {
			Db_entry new_entry(Db_entry(label, ObjectHistory<PointType>(d->getCloud(false))));
			entry = db.insert(new_entry).first;
		}
		entry->second.addObservation(time, d->getId(), d->getPose());
	}

	for(typename Db::iterator entry = db.begin(); entry != db.end(); entry++) {
		/**
		 * Removal by detection (disabled)
		entry->second.markRemovedIfNotDetected(time);
		 */
		entry->second.markPreservedIfNotDetectedNorRemoved(time);
	}
}

template<class PointType>
std::vector<ObjectIdLabeled> ObjectsHistory<PointType>::markRemovedObjects(
		const ChangeDetector<PointType> &change_detector) {
	std::vector<ObjectIdLabeled> removed;

	for(typename Db::iterator entry = db.begin(); entry != db.end(); entry++) {
		typename pcl::PointCloud<PointType>::Ptr cloud_posed(new pcl::PointCloud<PointType>());
		pcl::transformPointCloud(*(entry->second.getCloud()), *cloud_posed,
				entry->second.getLastPose());
		if(change_detector.isObjectRemoved(cloud_posed)) {
			int id = entry->second.markRemoved(time);
			removed.push_back(ObjectIdLabeled(id, entry->first));
		}
	}

	return removed;
}

template<class PointType>
std::vector< ObjectChangeForVisual<PointType> > ObjectsHistory<PointType>::getChanges(
		ObjectState::EventT changeType) {
	std::vector< ObjectChangeForVisual<PointType> > changes;

	for(typename Db::iterator entry = db.begin(); entry != db.end(); entry++) {
		if(entry->second.getLastEvent() == changeType &&
				entry->second.getTimeStamp() == time) {
			ObjectChangeForVisual<PointType> change;
			change.id = entry->second.getLastId();
			change.label = entry->first;
			change.cloud = entry->second.getCloud();
			change.pose = entry->second.getLastPose();

			if(changeType == ObjectState::MOVED) {
				change.pose_previous = entry->second.getLastPose(1);
			}

			changes.push_back(change);
		}
	}

	return changes;
}

}

#endif /* OBJECTHISTORY_HPP_ */
