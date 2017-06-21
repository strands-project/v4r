/**
 * $Id$
 * 
 * Software License Agreement (GNU General Public License)
 *
 *  Copyright (C) 2015:
 *
 *    Johann Prankl, prankl@acin.tuwien.ac.at
 *    Aitor Aldoma, aldoma@acin.tuwien.ac.at
 *
 *      Automation and Control Institute
 *      Vienna University of Technology
 *      Gusshausstraße 25-29
 *      1170 Vienn, Austria
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @author Johann Prankl
 *
 */

/**
 * TODO:
 * - if the loop is closed return the updated pose to the tracker and data integration
 * - test tracking confidence
 */


#include <v4r/camera_tracking_and_mapping/TSFVisualSLAM.h>
#include <v4r/common/convertImage.h>
#include <pcl/common/transforms.h>

#include "opencv2/highgui/highgui.hpp"


namespace v4r
{


using namespace std;



/************************************************************************************
 * Constructor/Destructor
 */
TSFVisualSLAM::TSFVisualSLAM(const Parameter &p)
{ 
  setParameter(p);
  tsfDataIntegration.setData(&data);
  tsfPoseTracker.setData(&data);
  tsfMapping.setData(&data);
}

TSFVisualSLAM::~TSFVisualSLAM()
{
}




/***************************************************************************************/

/**
 * @brief TSFVisualSLAM::stop
 */
void TSFVisualSLAM::stop()
{
  tsfDataIntegration.stop();
  tsfPoseTracker.stop();
  tsfMapping.stop();
}

/**
 * reset
 */
void TSFVisualSLAM::reset()
{
//  data.lock();   // i have to think about it....
  tsfPoseTracker.reset();
  tsfDataIntegration.reset();
  tsfMapping.reset();
  data.reset();
//  data.unlock();
}


/**
 * @brief TSFVisualSLAM::track
 * @param cloud
 * @param pose
 */
bool TSFVisualSLAM::track(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const uint64_t &timestamp, Eigen::Matrix4f &pose, double &conf_ransac_iter, double &conf_tracked_points)
{
  if (!tsfDataIntegration.isStarted() && param.tsf_filtering)
    tsfDataIntegration.start();

  if (!tsfPoseTracker.isStarted())
    tsfPoseTracker.start();

  if (!tsfMapping.isStarted() && param.tsf_filtering && param.tsf_mapping)
    tsfMapping.start();

  // should be save ;-)
  convertImage(cloud, data.image);
  cv::cvtColor( data.image, data.gray, CV_BGR2GRAY );

  if (!dbg.empty()) tsfPoseTracker.dbg = dbg;

  // the tracker does not copy data, we need to lock the shared memory
  data.lock();

  data.have_pose = (data.cloud.points.size()==0?true:false);
  data.cloud = cloud;
  data.timestamp = timestamp;
  tsfPoseTracker.track(conf_ransac_iter, conf_tracked_points);
  pose = data.pose;
  bool have_pose = data.have_pose;
  if (!have_pose) data.cnt_pose_lost_map++;

  data.unlock();

  return have_pose;
}

/**
 * @brief TSFVisualSLAM::getFilteredCloudNormals
 * @param cloud
 * @param timestamp
 */
void TSFVisualSLAM::getFilteredCloudNormals(pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud, Eigen::Matrix4f &pose, uint64_t &timestamp)
{
  data.lock();
  v4r::DataMatrix2D<Surfel> &cfilt = *data.filt_cloud;
  cloud.resize(cfilt.data.size());
  cloud.width = cfilt.cols;
  cloud.height = cfilt.rows;
  cloud.is_dense = false;
  TSFDataIntegration::computeNormals(cfilt);
  for (unsigned i=0; i<cfilt.data.size(); i++)
  {
    const Surfel &s = cfilt.data[i];
    pcl::PointXYZRGBNormal &o = cloud.points[i];
    o.getVector3fMap() = s.pt;
    o.r = s.r;
    o.g = s.g;
    o.b = s.b;
    o.getNormalVector3fMap() = s.n;
  }
  timestamp = data.filt_timestamp;
  pose = data.filt_pose;
  data.unlock();
}

/**
 * @brief TSFVisualSLAM::getFilteredCloudNormals
 * @param cloud
 * @param radius
 * @param pose
 * @param timestamp
 */
void TSFVisualSLAM::getFilteredCloudNormals(pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud, std::vector<float> &radius, Eigen::Matrix4f &pose, uint64_t &timestamp)
{
  data.lock();
  v4r::DataMatrix2D<Surfel> &cfilt = *data.filt_cloud;
  tsfDataIntegration.computeRadius(cfilt, intrinsic);
  cloud.resize(cfilt.data.size());
  cloud.width = cfilt.cols;
  cloud.height = cfilt.rows;
  cloud.is_dense = false;
  radius.resize(cloud.points.size());
  TSFDataIntegration::computeNormals(cfilt);
  for (unsigned i=0; i<cfilt.data.size(); i++)
  {
    const Surfel &s = cfilt.data[i];
    pcl::PointXYZRGBNormal &o = cloud.points[i];
    o.getVector3fMap() = s.pt;
    o.r = s.r;
    o.g = s.g;
    o.b = s.b;
    o.getNormalVector3fMap() = s.n;
    radius[i] = s.radius;
  }
  timestamp = data.filt_timestamp;
  pose = data.filt_pose;
  data.unlock();
}

/**
 * @brief TSFVisualSLAM::getFilteredCloudNormals
 * @param cloud
 * @param normals
 * @param timestamp
 */
void TSFVisualSLAM::getFilteredCloudNormals(pcl::PointCloud<pcl::PointXYZRGB> &cloud, pcl::PointCloud<pcl::Normal> &normals, Eigen::Matrix4f &pose, uint64_t &timestamp)
{
  data.lock();
  v4r::DataMatrix2D<Surfel> &cfilt = *data.filt_cloud;
  cloud.resize(cfilt.data.size());
  cloud.width = cfilt.cols;
  cloud.height = cfilt.rows;
  cloud.is_dense = false;
  normals.resize(cfilt.data.size());
  normals.width = cfilt.cols;
  normals.height = cfilt.rows;
  normals.is_dense = false;
  TSFDataIntegration::computeNormals(cfilt);
  for (unsigned i=0; i<cfilt.data.size(); i++)
  {
    const Surfel &s = cfilt.data[i];
    pcl::PointXYZRGB &o = cloud.points[i];
    o.getVector3fMap() = s.pt;
    o.r = s.r;
    o.g = s.g;
    o.b = s.b;
    normals.points[i].getNormalVector3fMap() = s.n;
  }
  timestamp = data.filt_timestamp;
  pose = data.filt_pose;
  data.unlock();
}

/**
 * @brief TSFVisualSLAM::getFilteredCloud
 * @param cloud
 * @param timestamp
 */
void TSFVisualSLAM::getFilteredCloud(pcl::PointCloud<pcl::PointXYZRGB> &cloud, Eigen::Matrix4f &pose, uint64_t &timestamp)
{
  data.lock();
  const v4r::DataMatrix2D<Surfel> &cfilt = *data.filt_cloud;
  cloud.resize(cfilt.data.size());
  cloud.width = cfilt.cols;
  cloud.height = cfilt.rows;
  cloud.is_dense = false;
  for (unsigned i=0; i<cfilt.data.size(); i++)
  {
    const Surfel &s = cfilt.data[i];
    pcl::PointXYZRGB &o = cloud.points[i];
    o.getVector3fMap() = s.pt;
    o.r = s.r;
    o.g = s.g;
    o.b = s.b;
  }
  timestamp = data.filt_timestamp;
  pose = data.filt_pose;
  data.unlock();
}

/**
 * @brief TSFVisualSLAM::getSurfelCloud
 * @param cloud
 * @param pose
 * @param timestamp
 */
void TSFVisualSLAM::getSurfelCloud(v4r::DataMatrix2D<Surfel> &cloud, Eigen::Matrix4f &pose, uint64_t &timestamp, bool need_normals)
{
  data.lock();
  cloud = *data.filt_cloud;
  timestamp = data.filt_timestamp;
  pose = data.filt_pose;
  data.unlock();
  if (need_normals) TSFDataIntegration::computeNormals(cloud);
}


/**
 * @brief TSFVisualSLAM::setParameter
 * @param p
 */
void TSFVisualSLAM::setParameter(const Parameter &p)
{
  param = p;
  tsfDataIntegration.setParameter(p.di_param);
  tsfPoseTracker.setParameter(p.pt_param);
  tsfMapping.setParameter(p.map_param);
}

/**
 * setCameraParameter
 */
void TSFVisualSLAM::setCameraParameter(const cv::Mat &_intrinsic)
{
  if (_intrinsic.type() != CV_64F)
    _intrinsic.convertTo(intrinsic, CV_64F);
  else intrinsic = _intrinsic;

  tsfPoseTracker.setCameraParameter(_intrinsic);
  tsfDataIntegration.setCameraParameter(_intrinsic);
  tsfMapping.setCameraParameter(_intrinsic);

  reset();
}

/**
 * @brief TSFVisualSLAM::setDetectors
 * @param _detector
 * @param _descEstimator
 */
void TSFVisualSLAM::setDetectors(const FeatureDetector::Ptr &_detector, const FeatureDetector::Ptr &_descEstimator)
{
  tsfMapping.setDetectors(_detector, _descEstimator);
}


/**
 * @brief TSFVisualSLAM::optimizeMap
 */
void TSFVisualSLAM::optimizeMap()
{
  tsfMapping.optimizeMap();
}

}












