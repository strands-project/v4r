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
 * @author Johann Prankl, Aitor Aldoma
 *
 */

#ifndef Q_MOC_RUN
#include "BundleAdjustment.h"
#include <v4r/keypoints/impl/toString.hpp>
#endif

using namespace std;


/**
 * @brief BundleAdjustment::BundleAdjustment
 */
BundleAdjustment::BundleAdjustment() :
  cmd(UNDEF), m_run(false)
{
}

/**
 * @brief BundleAdjustment::~BundleAdjustment
 */
BundleAdjustment::~BundleAdjustment()
{
  stop();
}



/******************************** public *******************************/

/**
 * @brief BundleAdjustment::start
 * @param cam_id
 */
void BundleAdjustment::start()
{
  QThread::start();
}

/**
 * @brief BundleAdjustment::stop
 */
void BundleAdjustment::stop()
{
  if(m_run)
  {
    m_run = false;
    this->wait();
  }
}

/**
 * @brief BundleAdjustment::isRunning
 * @return
 */
bool BundleAdjustment::isRunning()
{
  return m_run;
}

/**
 * @brief BundleAdjustment::cam_tracker_params_changed
 * @param _cam_tracker_params
 */
void BundleAdjustment::cam_tracker_params_changed(const CamaraTrackerParameter &_cam_tracker_params)
{
  cam_tracker_params = _cam_tracker_params;
}

/**
 * @brief BundleAdjustment::optimizeCamStructProj
 * @param _model
 */
void BundleAdjustment::optimizeCamStructProj(v4r::Object::Ptr &_model, boost::shared_ptr< std::vector<Sensor::CameraLocation> > &_cam_trajectory, boost::shared_ptr< std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > > &_log_clouds, boost::shared_ptr< Sensor::AlignedPointXYZRGBVector > &_oc_cloud)
{
  model = _model;
  cam_trajectory = _cam_trajectory;
  log_clouds = _log_clouds;
  oc_cloud = _oc_cloud;

  cmd = PROJ_BA_CAM_STRUCT;
  start();
}

/**
 * @brief BundleAdjustment::restoreCameras
 * @return
 */
bool BundleAdjustment::restoreCameras()
{
  if (model.get()==0)
    return false;

  Eigen::Matrix4f inv_pose;
  v4r::Object &ref_model = *model;

  if (stored_cameras.size()==ref_model.cameras.size() && stored_camera_parameter.size()==ref_model.camera_parameter.size() && stored_points.size()==ref_model.points.size())
  {
    ref_model.cameras = stored_cameras;
    ref_model.camera_parameter = stored_camera_parameter;
    ref_model.points = stored_points;

    for (unsigned i=0; i<cam_trajectory->size(); i++)
    {
      Sensor::CameraLocation &cam = (*cam_trajectory)[i];
      v4r::invPose(ref_model.cameras[cam.idx], inv_pose);
      cam.pt = inv_pose.block<3,1>(0,3);
      cam.vr = inv_pose.block<3,1>(0,2);
    }

    renewPrevCloud(ref_model.cameras, *log_clouds);

    emit update_cam_trajectory(cam_trajectory);
    emit update_model_cloud(oc_cloud);
    emit update_visualization();

    return true;
  }

  return false;
}


/********************************** private ****************************************/

/**
 * @brief BundleAdjustment::run
 * main loop
 */
void BundleAdjustment::run()
{
  m_run=true;

  switch (cmd)
  {
  case PROJ_BA_CAM_STRUCT:
  {
    // create tracking model
    emit printStatus("Status: Optimizing camera locations ... Please be patient...");

    optimizeCamStructProj();

    break;
  }
  default:
    break;
  }

  cmd = UNDEF;
  m_run=false;


  //while(m_run)
  //{
  //}
}

/**
 * @brief BundleAdjustment::optimizeCamStructProj
 */
void BundleAdjustment::optimizeCamStructProj()
{
  if (model.get()==0 || cam_trajectory.get()==0 || oc_cloud.get()==0)
    return;

  unsigned z=0;
  Eigen::Matrix4f inv_pose;

  v4r::ProjBundleAdjuster ba(v4r::ProjBundleAdjuster::Parameter(false,false,true, 100.,0.02, 1.2));

  v4r::Object &ref_model = *model;
  stored_cameras = ref_model.cameras;
  stored_camera_parameter = ref_model.camera_parameter;
  stored_points = ref_model.points;

  ba.optimize(ref_model);

  for (unsigned i=0; i<cam_trajectory->size(); i++)
  {
    Sensor::CameraLocation &cam = (*cam_trajectory)[i];

    if (cam.idx>=0)
    {
      v4r::invPose(ref_model.cameras[cam.idx], inv_pose);
      cam.pt = inv_pose.block<3,1>(0,3);
      cam.vr = inv_pose.block<3,1>(0,2);
      (*cam_trajectory)[z] = cam;
      z++;
    }
  }
  cam_trajectory->resize(z);

  renewPrevCloud(ref_model.cameras, *log_clouds);
  int num_cameras = ref_model.cameras.size();

  emit update_cam_trajectory(cam_trajectory);
  emit update_model_cloud(oc_cloud);
  emit update_visualization();
  emit finishedOptimizeCameras(num_cameras);
  std::string txt = std::string("Status: Optimized ")+v4r::toString(num_cameras,0)+std::string(" cameras");
  emit printStatus(txt);
}

/**
 * @brief BundleAdjustment::renewPrevCloud
 * @param poses
 * @param clouds
 */
void BundleAdjustment::renewPrevCloud(const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &poses, const std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > &clouds)
{
  if (clouds.size()>0)
  {
    Eigen::Matrix4f inv_pose;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_cloud2(new pcl::PointCloud<pcl::PointXYZRGB>());
    std::vector<int> indices;

    pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZRGB,pcl::octree::OctreeVoxelCentroidContainerXYZRGB<pcl::PointXYZRGB> >::Ptr octree;
    octree.reset(new pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZRGB,pcl::octree::OctreeVoxelCentroidContainerXYZRGB<pcl::PointXYZRGB> >(cam_tracker_params.prev_voxegrid_size));
    pcl::PassThrough<pcl::PointXYZRGB> pass;

    for (unsigned i=0; i<clouds.size(); i++)
    {
      v4r::invPose(poses[clouds[i].first], inv_pose);
      pcl::removeNaNFromPointCloud(*clouds[i].second,*tmp_cloud,indices);
      pass.setInputCloud (tmp_cloud);
      pass.setFilterFieldName ("z");
      pass.setFilterLimits (0.0, cam_tracker_params.prev_z_cutoff);
      pass.filter (*tmp_cloud2);
      pcl::transformPointCloud(*tmp_cloud2, *tmp_cloud, inv_pose);
      octree->setInputCloud(tmp_cloud);
      octree->addPointsFromInputCloud();
    }

    oc_cloud->clear();
    octree->getVoxelCentroids(*oc_cloud);
  }
}


