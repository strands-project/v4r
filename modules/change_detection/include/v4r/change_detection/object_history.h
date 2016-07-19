/*
 * ObjectHistory.h
 *
 *  Created on: 13 Aug 2015
 *      Author: martin
 */

#ifndef OBJECTHISTORY_H_
#define OBJECTHISTORY_H_

#include <pcl/common/eigen.h>
#include <map>

#include <v4r/core/macros.h>
#include <v4r/change_detection/change_detection.h>
#include <v4r/change_detection/object_detection.h>

namespace v4r {

class ObjectState {
public:
	typedef enum Event {
		ADDED,
		REMOVED,
		PRESERVED,
		MOVED
	} EventT;

	std::string string() const {
		return string(event);
	}

	static std::string string(const EventT &event_) {
		switch(event_) {
		case ADDED:
			return "added";
		case REMOVED:
			return "removed";
		case PRESERVED:
			return "preserved";
		default:
			return "unknown";
		}
	}

	static ObjectState Added(int time, int id, const Eigen::Affine3f &pose) {
		return ObjectState(time, ADDED, id, pose);
	}

	static ObjectState Moved(int time, int id, const Eigen::Affine3f &new_pose) {
		return ObjectState(time, MOVED, id, new_pose);
	}

	static ObjectState Preserved(int time, int id, const Eigen::Affine3f &pose) {
		return ObjectState(time, PRESERVED, id, pose);
	}

	static ObjectState Removed(int time, int id) {
		return ObjectState(time, REMOVED, id, Eigen::Affine3f::Identity());
	}

	const Eigen::Affine3f& getPose() const {
		if(event == REMOVED) {
			throw std::logic_error("Unable to get position of removed object.");
		}
		return pose;
	}

	EventT getEvent() const {
		return event;
	}

	int getId() const {
		return id;
	}

	int getTimestamp() const {
		return timestamp;
	}

private:
	int timestamp;
	EventT event;
	int id;
	Eigen::Affine3f pose;

	ObjectState(int time_, Event event_, int id_, const Eigen::Affine3f &pose_) :
		timestamp(time_), event(event_), id(id_), pose(pose_) {
	}
};

template<class PointType>
class ObjectHistory {
public:
	typedef boost::shared_ptr< ObjectHistory<PointType> > Ptr;

	ObjectHistory(typename pcl::PointCloud<PointType>::Ptr object_cloud) :
		cloud(object_cloud) {
	}

	void addObservation(int timestamp, int id, const Eigen::Affine3f &pose) {
		if(historyStates.empty()) {
			historyStates.push_back(ObjectState::Added(timestamp, id, pose));
		} else if(historyStates.back().getEvent() == ObjectState::REMOVED) {
			if(historyStates.back().getTimestamp() == timestamp) {
				if(historyStates.back().getId() == id) {
					// this exact object has been removed => this is false observation
					return;
				}
				historyStates.pop_back();
			}
			historyStates.push_back(ObjectState::Moved(timestamp, id, pose));
		} else {
			historyStates.push_back(ObjectState::Preserved(timestamp, id, pose));
		}
	}

	void markRemovedIfNotDetected(int timestamp) {
		if(historyStates.back().getTimestamp() < timestamp &&
				historyStates.back().getEvent() != ObjectState::REMOVED) {
			historyStates.push_back(ObjectState::Removed(timestamp, historyStates.back().getId()));
		}
	}

	void markPreservedIfNotDetectedNorRemoved(int timestamp) {
		if(historyStates.back().getTimestamp() < timestamp &&
				historyStates.back().getEvent() != ObjectState::REMOVED) {
			historyStates.push_back(ObjectState::Preserved(timestamp,
					historyStates.back().getId(),
					historyStates.back().getPose()));
		}
	}

	int markRemoved(int timestamp) {
		int id = historyStates.back().getId();
		if(historyStates.back().getTimestamp() == timestamp) {
			historyStates.pop_back();
		} else {
			historyStates.push_back(ObjectState::Removed(timestamp, id));
		}
		return id;
	}

	ObjectState::Event getLastEvent() {
		return historyStates.back().getEvent();
	}

	int getTimeStamp() {
		return historyStates.back().getTimestamp();
	}

	int getLastId() {
		return historyStates.back().getId();
	}

	typename pcl::PointCloud<PointType>::Ptr getCloud() {
		return cloud;
	}

	Eigen::Affine3f getLastPose(int recordsToIgnore = 0) {
		for(int i = historyStates.size() - 1 - recordsToIgnore; i >= 0; i--) {
			if(historyStates[i].getEvent() != ObjectState::REMOVED) {
				return historyStates[i].getPose();
			}
		}
		return Eigen::Affine3f::Identity();
	}

private:
	typename pcl::PointCloud<PointType>::Ptr cloud;
	std::vector<ObjectState> historyStates;
};

typedef std::string ObjectLabel;

class ObjectIdLabeled {
public:
	int id;
	ObjectLabel label;

	ObjectIdLabeled(const int id_, const ObjectLabel &label_) :
		id(id_), label(label_) {
	}
};

template<class PointType>
class ObjectChangeForVisual {
public:
	int id;
	ObjectLabel label;
	typename pcl::PointCloud<PointType>::Ptr cloud;
	Eigen::Affine3f pose;
	Eigen::Affine3f pose_previous;	// only for move
};

template<class PointType>
class V4R_EXPORTS ObjectsHistory {
public:
	typedef typename boost::shared_ptr< ObjectsHistory<PointType> > Ptr;

	void add(const std::vector< ObjectDetection<PointType> > &detections);
	std::vector<ObjectIdLabeled> markRemovedObjects(const ChangeDetector<PointType> &change_detector);

	typename pcl::PointCloud<PointType>::Ptr getLastCloudOf(const ObjectLabel &label) {
		typename Db::iterator entry = db.find(label);
		typename pcl::PointCloud<PointType>::Ptr cloud_posed(new pcl::PointCloud<PointType>());
		if(entry == db.end()) {
			PCL_WARN("Request for last point cloud of %s, but it has not been observed yet! Empty cloud returned\n",
					label.c_str());
			return cloud_posed;
		} else {
			pcl::transformPointCloud(*(entry->second.getCloud()), *cloud_posed, entry->second.getLastPose());
			return cloud_posed;
		}
	}

	static int now() {
		return time;
	}

	static void incrementTime() {
		time++;
	}

	std::vector< ObjectChangeForVisual<PointType> > getChanges(ObjectState::EventT change);

private:
	typedef ObjectHistory<PointType> ObjectHistoryT;
	typedef std::pair< ObjectLabel, ObjectHistoryT > Db_entry;
	typedef std::map<ObjectLabel, ObjectHistoryT> Db;
	Db db;
	static int time;
};

}

#endif /* OBJECTHISTORY_H_ */
