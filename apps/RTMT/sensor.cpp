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
#include "sensor.h"
#include <pcl/common/io.h>

#include <v4r/keypoints/impl/invPose.hpp>
#include <v4r/keypoints/impl/PoseIO.hpp>
#include <v4r/keypoints/RigidTransformationRANSAC.h>
#include <v4r/common/convertImage.h>
#include <v4r/keypoints/impl/toString.hpp>
//#include "v4r/KeypointTools/ScopeTime.hpp"
#include <pcl/io/pcd_io.h>
#endif

using namespace std;


/**
 * @brief Sensor::Sensor
 */
Sensor::Sensor() :
  m_run(false),
  m_run_tracker(false),
  m_cam_id(0),
  m_draw_mask(true),
  m_select_roi(false),
  m_activate_roi(false),
  roi_seed_x(320), roi_seed_y(280),
  u_idle_time(35*1000),   // 35 ms
  max_queue_size(3),       // allow a queue size of 3 clouds
  pose(Eigen::Matrix4f::Identity()),
  conf(0),
  cam_id(-1),
  bbox_scale_xy(1.),
  bbox_scale_height(1.),
  seg_offs(0.01),
  bb_min(Eigen::Vector3f(-2.,-2.,-0.01)),
  bb_max(Eigen::Vector3f(2.,2.,.5))
{
  /**
     * kd_param: ransac inlier distance for interest point based reinitialization [m] (e.g. 0.01)
     * lk_param: ransac inlier distance for  lk based tracking [m] (e.g. 0.01)
     * kt_param: ransac inlier distance for projective lk based refinement
     *           (can be a factor 2 to 5 more than dist_reinit and dist_track) [m] (e.g. 0.03)
     */
  ct_param.kd_param.rt_param.inl_dist = 0.01; //e.g. 0.01 .. table top, 0.03 ..rooms
  ct_param.lk_param.rt_param.inl_dist = 0.01; //e.g. 0.01 .. table top, 0.03 ..rooms
  ct_param.kt_param.rt_param.inl_dist = 0.03;  //e.g. 0.04 .. table top, 0.1 ..room
  ct_param.om_param.kd_param.rt_param.inl_dist = 0.01; //e.g. 0.01 .. table top, 0.03 ..rooms
  ct_param.om_param.kt_param.rt_param.inl_dist = 0.03;  //e.g. 0.04 .. table top, 0.1 ..room
  ct_param.kt_param.plk_param.use_ncc = true;
  ct_param.kt_param.plk_param.ncc_residual = .4;    // default 0.3
  ct_param.det_param.nfeatures = 200;             // default 250


  // set cameras
  cv::Mat_<double> cam = cv::Mat_<double>::eye(3,3);
  cv::Mat_<double> dist_coeffs;

  cam(0,0) = cam_params.f[0]; cam(1,1) = cam_params.f[1];
  cam(0,2) = cam_params.c[0]; cam(1,2) = cam_params.c[1];

  // init tracker
  camtracker.reset( new v4r::KeypointSlamRGBD2(ct_param) );
  camtracker->setCameraParameter(cam, dist_coeffs);

  tmp_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
  tmp_cloud2.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
  cam_trajectory.reset(new std::vector<CameraLocation>() );
  log_clouds.reset( new std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> >() );

  cos_min_delta_angle = cos(20*M_PI/180.);
  sqr_min_cam_distance = 1.*1.;
  octree.reset(new pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZRGB,pcl::octree::OctreeVoxelCentroidContainerXYZRGB<pcl::PointXYZRGB> >(cam_tracker_params.prev_voxegrid_size));
  oc_cloud.reset(new AlignedPointXYZRGBVector () );

  v4r::ClusterNormalsToPlanes::Parameter p_param;
  p_param.thrAngle=45;
  p_param.inlDist=0.01;
  p_param.minPoints=5000;
  p_param.least_squares_refinement=true;
  p_param.smooth_clustering=false;
  p_param.thrAngleSmooth=30;
  p_param.inlDistSmooth=0.01;
  p_param.minPointsSmooth=3;
  pest.reset(new v4r::ClusterNormalsToPlanes(p_param));

  v4r::ZAdaptiveNormals::Parameter n_param;
  n_param.adaptive = true;
  nest.reset(new v4r::ZAdaptiveNormals(n_param));
}

/**
 * @brief Sensor::~Sensor
 */
Sensor::~Sensor()
{
  stop();
}



/******************************** public *******************************/

/**
 * @brief Sensor::getClouds
 * @return
 */
//const boost::shared_ptr< std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > >& Sensor::getClouds()
//{
//  stopTracker();
//  return log_clouds;
//}

/**
 * @brief Sensor::getCameras
 * @return
 */
const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >& Sensor::getCameras()
{
  stopTracker();
  return camtracker->getModel().cameras;
}

/**
 * @brief Sensor::start
 * @param cam_id
 */
void Sensor::start(int cam_id)
{
  m_cam_id = cam_id;
  QThread::start();
}

/**
 * @brief Sensor::stop
 */
void Sensor::stop()
{
  if(m_run)
  {
    m_run = false;
    stopTracker();
    this->wait();
  }
}

/**
 * @brief Sensor::startTracker
 * @param cam_id
 */
void Sensor::startTracker(int cam_id)
{
  if (!m_run) start(cam_id);

  m_run_tracker = true;
}

/**
 * @brief Sensor::stopTracker
 */
void Sensor::stopTracker()
{
  if(m_run_tracker)
  {
    m_run_tracker = false;
    camtracker->stopObjectManagement();
  }
}

/**
 * @brief Sensor::isRunning
 * @return
 */
bool Sensor::isRunning()
{
  return m_run;
}

/**
 * @brief Sensor::set_roi_params
 * @param _bbox_scale_xy
 * @param _bbox_scale_height
 * @param _seg_offs
 */
void Sensor::set_roi_params(const double &_bbox_scale_xy, const double &_bbox_scale_height, const double &_seg_offs)
{
  bbox_scale_xy = _bbox_scale_xy;
  bbox_scale_height = _bbox_scale_height;
  seg_offs = _seg_offs;
}

/**
 * @brief Sensor::cam_params_changed
 * @param _cam_params
 */
void Sensor::cam_params_changed(const RGBDCameraParameter &_cam_params)
{
  bool need_start = m_run;
  bool need_start_tracker = m_run_tracker;

  stop();

  cam_params = _cam_params;

  cv::Mat_<double> cam = cv::Mat_<double>::eye(3,3);
  cv::Mat_<double> dist_coeffs;

  cam(0,0) = cam_params.f[0]; cam(1,1) = cam_params.f[1];
  cam(0,2) = cam_params.c[0]; cam(1,2) = cam_params.c[1];

  camtracker->setCameraParameter(cam, dist_coeffs);

  ct_param.kd_param.rt_param.inl_dist = 0.01;          //e.g. 0.01 .. table top, 0.03 ..rooms
  ct_param.lk_param.rt_param.inl_dist = 0.01;          //e.g. 0.01 .. table top, 0.03 ..rooms
  ct_param.kt_param.rt_param.inl_dist = 0.03;          //e.g. 0.04 .. table top, 0.1 ..room
  ct_param.om_param.kd_param.rt_param.inl_dist = 0.01; //e.g. 0.01 .. table top, 0.03 ..rooms
  ct_param.om_param.kt_param.rt_param.inl_dist = 0.03; //e.g. 0.04 .. table top, 0.1 ..room

  //ct_param.kt_param.plk_param.use_ncc = true;
  //ct_param.kt_param.plk_param.ncc_residual = .9;

  // init tracker
  camtracker.reset( new v4r::KeypointSlamRGBD2(ct_param) );
  camtracker->setCameraParameter(cam, dist_coeffs);
  camtracker->setMinDistAddProjections(ba_params.dist_cam_add_projections);


  tmp_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
  tmp_cloud2.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
  cam_trajectory.reset(new std::vector<CameraLocation>() );
  log_clouds.reset( new std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> >() );

  cos_min_delta_angle = cos(20*M_PI/180.);
  sqr_min_cam_distance = 1.*1.;
  octree.reset(new pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZRGB,pcl::octree::OctreeVoxelCentroidContainerXYZRGB<pcl::PointXYZRGB> >(cam_tracker_params.prev_voxegrid_size));
  oc_cloud.reset(new AlignedPointXYZRGBVector () );
  pose = Eigen::Matrix4f::Identity();
  conf = 0.;

  if (need_start) start(cam_id);
  if (need_start_tracker) startTracker(cam_id);
}

/**
 * @brief Sensor::bundle_adjustment_parameter_changed
 * @param param
 */
void Sensor::bundle_adjustment_parameter_changed(const BundleAdjustmentParameter& param)
{
  ba_params = param;

  bool need_start=false;

  if (m_run_tracker)
  {
    stopTracker();
    need_start = true;
  }

  camtracker->setMinDistAddProjections(ba_params.dist_cam_add_projections);

  if (need_start)
    startTracker(cam_id);
}

/**
 * @brief Sensor::cam_tracker_params_changed
 * @param _cam_tracker_params
 */
void Sensor::cam_tracker_params_changed(const CamaraTrackerParameter &_cam_tracker_params)
{
  cam_tracker_params = _cam_tracker_params;

  cos_min_delta_angle = cos(cam_tracker_params.min_delta_angle*M_PI/180.);
  sqr_min_cam_distance = cam_tracker_params.min_delta_distance*cam_tracker_params.min_delta_distance;
  octree->setResolution(cam_tracker_params.prev_voxegrid_size);
}

/**
 * @brief Sensor::select_roi
 * @param x
 * @param y
 */
void Sensor::select_roi(int x, int y)
{
  roi_seed_x = x;
  roi_seed_y = y;
  m_select_roi = true;
}

/**
 * @brief Sensor::reset
 */
void Sensor::reset()
{
  bool is_run_tracker = m_run_tracker;

  stopTracker();

  cam_trajectory.reset(new std::vector<CameraLocation>());
  log_clouds.reset(new std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> >());
  cameras.clear();

  octree.reset(new pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZRGB,pcl::octree::OctreeVoxelCentroidContainerXYZRGB<pcl::PointXYZRGB> >(cam_tracker_params.prev_voxegrid_size));
  oc_cloud.reset(new AlignedPointXYZRGBVector () );

  camtracker->reset();

  pose.setIdentity();
  conf = 0;

  if (is_run_tracker) startTracker(m_cam_id);
}


/**
 * @brief Sensor::storeKeyframes
 * @param _folder
 */
void Sensor::storeKeyframes(const std::string &_folder)
{
  char filename[PATH_MAX];
  cv::Mat_<cv::Vec3b> image;

  std::string cloud_names = _folder+"/cloud_%04d.pcd";
  std::string image_names = _folder+"/image_%04d.jpg";
  std::string pose_names = _folder+"/pose_%04d.txt";

  v4r::Object &model = camtracker->getModel();

  for (unsigned i=0; i<log_clouds->size(); i++)
  {
    snprintf(filename,PATH_MAX, cloud_names.c_str(), i);
    pcl::io::savePCDFileBinary(filename, *log_clouds->at(i).second);
    v4r::convertImage(*log_clouds->at(i).second, image);
    snprintf(filename,PATH_MAX, image_names.c_str(), i);
    cv::imwrite(filename, image);
    snprintf(filename,PATH_MAX, pose_names.c_str(), i);
    v4r::writePose(filename, _folder, model.cameras[log_clouds->at(i).first]);
  }
}

/**
 * @brief Sensor::storePointCloudModel
 * @param _folder
 */
void Sensor::storePointCloudModel(const std::string &_folder)
{
  pcl::PointCloud<pcl::PointXYZRGB> &cloud = *tmp_cloud;
  const AlignedPointXYZRGBVector &oc = *oc_cloud;

  cloud.resize(oc.size());
  cloud.width = oc.size();
  cloud.height = 1;
  cloud.is_dense = true;

  for (unsigned i=0; i<oc.size(); i++)
    cloud.points[i] = oc[i];

  if (cloud.points.size()>0)
    pcl::io::savePCDFileBinary(_folder+"/model.pcd", *tmp_cloud);
}

/**
 * @brief Sensor::showDepthMask
 * @param _draw_depth_mask
 */
void Sensor::showDepthMask(bool _draw_depth_mask)
{
  m_draw_mask = _draw_depth_mask;
}

/**
 * @brief Sensor::selectROI
 * @param _seed_x
 * @param _seed_y
 */
void Sensor::selectROI(int _seed_x, int _seed_y)
{
  m_select_roi = true;
  roi_seed_x = _seed_x;
  roi_seed_y = _seed_y;
}

/**
 * @brief Sensor::activateROI
 * @param enable
 */
void Sensor::activateROI(bool enable)
{
  m_activate_roi = true;
}




/*************************************** private **************************************************/

void Sensor::CallbackCloud (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &_cloud)
{
  shm_mutex.lock();

  if (shm_clouds.size() < max_queue_size)
  {
    shm_clouds.push( pcl::PointCloud<pcl::PointXYZRGB>::Ptr() );
    shm_clouds.back().reset (new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::copyPointCloud(*_cloud,*shm_clouds.back());
  }

  shm_mutex.unlock();

  usleep(u_idle_time);
}

/**
 * @brief Sensor::run
 * main loop
 */
void Sensor::run()
{
  try
  {
    interface.reset( new pcl::OpenNIGrabber() );
  }
  catch (pcl::IOException e)
  {
    m_run = false;
    m_run_tracker = false;
    emit printStatus(std::string("Status: No OpenNI device connected!"));
    return;
  }
  boost::function<void (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> f =
      boost::bind (&Sensor::CallbackCloud, this, _1);
  interface->registerCallback (f);
  interface->start ();

  m_run=true;
  bool is_conf;
  int type=0;

  while(m_run)
  {
    // -------------------- do tracking --------------------------

    shm_mutex.lock();

    if (!shm_clouds.empty()){
      cloud = shm_clouds.front();
      shm_clouds.pop();
    } else cloud.reset();

    shm_mutex.unlock();

    if (cloud.get()!=0)
    {
      //v4r::ScopeTime t("tracking");
      v4r::convertCloud(*cloud, kp_cloud, image);

      // select a roi
      if (m_select_roi) detectROI(kp_cloud);
      if (m_activate_roi) maskCloud(pose*bbox_base_transform, bb_min, bb_max, kp_cloud);

      // track camera
      if (m_run_tracker)
      {
        is_conf = camtracker->track(image, kp_cloud, pose, conf, cam_id);

        if (is_conf)
        {
          type = selectFrames(*cloud, cam_id, pose, *cam_trajectory);

          if (type >= 0) emit update_cam_trajectory(cam_trajectory);
          if (type == 2 && cam_tracker_params.create_prev_cloud) emit update_model_cloud(oc_cloud);
          if (m_activate_roi) emit update_boundingbox(edges, pose*bbox_base_transform);
        }
      }


      if (m_draw_mask) drawDepthMask(*cloud,image);
      drawConfidenceBar(image,conf);

      emit new_image(cloud, image);
      emit update_visualization();

    } else usleep(u_idle_time/2);
  }

  usleep(50000);
  interface->stop ();
  usleep(50000);
}

/**
 * @brief Sensor::selectFrames
 * @param cloud
 * @param cam_id
 * @param pose
 * @param traj
 * @return type
 */
int Sensor::selectFrames(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, int cam_id, const Eigen::Matrix4f &pose, std::vector<CameraLocation> &traj)
{
  int type = 0;


  if (cam_tracker_params.log_point_clouds && cam_id>=0)
  {
    type = 1;

    Eigen::Matrix4f inv_pose;
    v4r::invPose(pose, inv_pose);

    unsigned z;
    for (z=0; z<cameras.size(); z++)
    {
      if ( (inv_pose.block<3,1>(0,2).dot(cameras[z].block<3,1>(0,2)) > cos_min_delta_angle) &&
           (inv_pose.block<3,1>(0,3)-cameras[z].block<3,1>(0,3)).squaredNorm() < sqr_min_cam_distance )
      {
        break;
      }
    }

    if (z>=cameras.size())
    {
      type = 2;
      cameras.push_back(inv_pose);
      log_clouds->push_back(make_pair(cam_id, pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>())));
      pcl::copyPointCloud(cloud,*log_clouds->back().second);

      //create preview modelclouds[i].first
      if (cam_tracker_params.create_prev_cloud)
      {
        pcl::removeNaNFromPointCloud(cloud,*tmp_cloud,indices);
        pass.setInputCloud (tmp_cloud);
        pass.setFilterFieldName ("z");
        pass.setFilterLimits (0.0, cam_tracker_params.prev_z_cutoff);
        pass.filter (*tmp_cloud2);
        pcl::transformPointCloud(*tmp_cloud2, *tmp_cloud, inv_pose);
        octree->setInputCloud(tmp_cloud);
        octree->addPointsFromInputCloud();
        oc_cloud->clear();
        octree->getVoxelCentroids(*oc_cloud);
      }

      emit printStatus(std::string("Status: Selected ")+v4r::toString(log_clouds->size(),0)+std::string(" keyframes"));
    }

    traj.push_back( CameraLocation(cam_id, type,inv_pose.block<3,1>(0,3),inv_pose.block<3,1>(0,2)) );
  }


  return type;
}

/**
 * @brief Sensor::renewPrevCloud
 * @param clouds
 */
void Sensor::renewPrevCloud(const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &poses, const std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > &clouds)
{
  if (clouds.size()>0)
  {
    Eigen::Matrix4f inv_pose;
    octree.reset(new pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZRGB,pcl::octree::OctreeVoxelCentroidContainerXYZRGB<pcl::PointXYZRGB> >(cam_tracker_params.prev_voxegrid_size));

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

/**
 * drawConfidenceBar
 */
void Sensor::drawConfidenceBar(cv::Mat &im, const double &conf)
{
  int bar_start = 50, bar_end = 200;
  int diff = bar_end-bar_start;
  int draw_end = diff*conf;
  double col_scale = 255./(double)diff;
  cv::Point2f pt1(0,30);
  cv::Point2f pt2(0,30);
  cv::Vec3b col(0,0,0);

  if (draw_end<=0) draw_end = 1;

  for (int i=0; i<draw_end; i++)
  {
    col = cv::Vec3b(255-(i*col_scale), i*col_scale, 0);
    pt1.x = bar_start+i;
    pt2.x = bar_start+i+1;
    cv::line(im, pt1, pt2, CV_RGB(col[0],col[1],col[2]), 8);
  }
}

/**
 * @brief Sensor::drawDepthMask
 * @param cloud
 * @param im
 */
void Sensor::drawDepthMask(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, cv::Mat &im)
{
  if ((int)cloud.width!=im.cols || (int)cloud.height!=im.rows)
    return;

  for (unsigned i=0; i<cloud.width*cloud.height; i++)
    if (isnan(cloud.points[i].x)) im.at<cv::Vec3b>(i) = cv::Vec3b(255,0,0);

//  for (unsigned i=0; i<plane.indices.size(); i++)
//  {
//    im.at<cv::Vec3b>(plane.indices[i]) = cv::Vec3b(255,0,0);
//  }
}

/**
 * @brief Sensor::getInplaneTransform
 * @param pt
 * @param normal
 * @param pose
 */
void Sensor::getInplaneTransform(const Eigen::Vector3f &pt, const Eigen::Vector3f &normal, Eigen::Matrix4f &pose)
{
  pose.setIdentity();

  Eigen::Vector3f px, py;
  Eigen::Vector3f pz = normal;

  if (pt.dot(pz) > 0) pz *= -1;
  px = (Eigen::Vector3f(1,0,0).cross(pz)).normalized();
  py = (pz.cross(px)).normalized();

  pose.block<3,1>(0,0) = px;
  pose.block<3,1>(0,1) = py;
  pose.block<3,1>(0,2) = pz;
  pose.block<3,1>(0,3) = pt;

//  std::vector<Eigen::Vector3f> pts0(4), pts1(4);
//  std::vector<int> indices(4,0);
//  indices[1] = 1, indices[2] = 2, indices[3] = 3;
//  pts0[0] = Eigen::Vector3f(0,0,0), pts0[1] = Eigen::Vector3f(1,0,0);
//  pts0[2] = Eigen::Vector3f(0,1,0), pts0[3] = Eigen::Vector3f(0,0,1);

//  pts1[0] = pt;
//  pts1[3] = normal.normalized();

//  if (pts1[0].dot(pts1[3]) > 0)
//    pts1[3] *= -1;

//  pts1[1] = (pts0[1].cross(pts1[3])).normalized();
//  pts1[2] = (pts1[3].cross(pts1[1])).normalized();

//  pts1[1]+=pts1[0];
//  pts1[2]+=pts1[0];
//  pts1[3]+=pts1[0];

//  v4r::RigidTransformationRANSAC rt;
//  rt.estimateRigidTransformationSVD(pts0, indices, pts1, indices, pose);
}

/**
 * @brief Sensor::maskCloud
 * @param cloud
 * @param pose
 * @param bb_min
 * @param bb_max
 */
void Sensor::maskCloud(const Eigen::Matrix4f &pose, const Eigen::Vector3f &bb_min, const Eigen::Vector3f &bb_max, v4r::DataMatrix2D<Eigen::Vector3f> &cloud)
{
  Eigen::Vector3f pt_glob;
  Eigen::Matrix4f inv_pose;
  v4r::invPose(pose,inv_pose);
  Eigen::Matrix3f R = inv_pose.topLeftCorner<3,3>();
  Eigen::Vector3f t = inv_pose.block<3,1>(0,3);

  for (unsigned i=0; i<cloud.data.size(); i++)
  {
    Eigen::Vector3f &pt = cloud[i];

    if (!isNaN(pt))
    {
      pt_glob = R*pt + t;

      if (pt_glob[0]<bb_min[0] || pt_glob[0]>bb_max[0] || pt_glob[1]<bb_min[1] || pt_glob[1]>bb_max[1] || pt_glob[2]<bb_min[2] || pt_glob[2]>bb_max[2])
      {
        pt = Eigen::Vector3f(std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN());
      }
    }
  }

}

/**
 * @brief Sensor::getBoundingBox
 * @param cloud
 * @param pose
 * @param bbox
 * @param xmin
 * @param xmax
 * @param ymin
 * @param ymax
 */
void Sensor::getBoundingBox(const v4r::DataMatrix2D<Eigen::Vector3f> &cloud, const std::vector<int> &indices, const Eigen::Matrix4f &pose, std::vector<Eigen::Vector3f> &bbox, Eigen::Vector3f &bb_min, Eigen::Vector3f &bb_max)
{
  Eigen::Vector3f pt, bbox_center_xy;
  double xmin, xmax, ymin, ymax, bbox_height, h_bbox_length, h_bbox_width;

  xmin = ymin = DBL_MAX;
  xmax = ymax = -DBL_MAX;

  Eigen::Matrix4f inv_pose;
  v4r::invPose(pose,inv_pose);
  Eigen::Matrix3f R = inv_pose.topLeftCorner<3,3>();
  Eigen::Vector3f t = inv_pose.block<3,1>(0,3);

  for (unsigned i=0; i<indices.size(); i++)
  {
    pt = R*cloud[indices[i]]+t;
    if (pt[0]>xmax) xmax = pt[0];
    if (pt[0]<xmin) xmin = pt[0];
    if (pt[1]>ymax) ymax = pt[1];
    if (pt[1]<ymin) ymin = pt[1];
  }

  h_bbox_length = bbox_scale_xy*(xmax-xmin)/2.;
  h_bbox_width = bbox_scale_xy*(ymax-ymin)/2.;
  bbox_height = bbox_scale_height*(xmax-xmin+ymax-ymin)/2.;
  bbox_center_xy = Eigen::Vector3f((xmin+xmax)/2.,(ymin+ymax)/2.,0.);


  bbox.clear();
  bbox.push_back(bbox_center_xy+Eigen::Vector3f(-h_bbox_length,-h_bbox_width,0.));
  bbox.push_back(bbox_center_xy+Eigen::Vector3f(h_bbox_length,-h_bbox_width,0.));
  bbox.push_back(bbox_center_xy+Eigen::Vector3f(h_bbox_length,-h_bbox_width,0.));
  bbox.push_back(bbox_center_xy+Eigen::Vector3f(h_bbox_length,h_bbox_width,0.));
  bbox.push_back(bbox_center_xy+Eigen::Vector3f(h_bbox_length,h_bbox_width,0.));
  bbox.push_back(bbox_center_xy+Eigen::Vector3f(-h_bbox_length,h_bbox_width,0.));
  bbox.push_back(bbox_center_xy+Eigen::Vector3f(-h_bbox_length,h_bbox_width,0.));
  bbox.push_back(bbox_center_xy+Eigen::Vector3f(-h_bbox_length,-h_bbox_width,0.));

  bbox.push_back(bbox_center_xy+Eigen::Vector3f(-h_bbox_length,-h_bbox_width,bbox_height));
  bbox.push_back(bbox_center_xy+Eigen::Vector3f(h_bbox_length,-h_bbox_width,bbox_height));
  bbox.push_back(bbox_center_xy+Eigen::Vector3f(h_bbox_length,-h_bbox_width,bbox_height));
  bbox.push_back(bbox_center_xy+Eigen::Vector3f(h_bbox_length,h_bbox_width,bbox_height));
  bbox.push_back(bbox_center_xy+Eigen::Vector3f(h_bbox_length,h_bbox_width,bbox_height));
  bbox.push_back(bbox_center_xy+Eigen::Vector3f(-h_bbox_length,h_bbox_width,bbox_height));
  bbox.push_back(bbox_center_xy+Eigen::Vector3f(-h_bbox_length,h_bbox_width,bbox_height));
  bbox.push_back(bbox_center_xy+Eigen::Vector3f(-h_bbox_length,-h_bbox_width,bbox_height));

  for (unsigned i=0; i<8; i+=2)
  {
    bbox.push_back(bbox[i]);
    bbox.push_back(bbox[i+8]);
  }

  bb_min = bbox_center_xy+Eigen::Vector3f(-h_bbox_length,-h_bbox_width, -seg_offs);
  bb_max = bbox_center_xy+Eigen::Vector3f(h_bbox_length,h_bbox_width, bbox_height);
}


/**
 * @brief Sensor::detectROI
 * @param cloud
 */
void Sensor::detectROI(const v4r::DataMatrix2D<Eigen::Vector3f> &cloud)
{
  v4r::DataMatrix2D<Eigen::Vector3f> normals;
  //v4r::ClusterNormalsToPlanes::Plane plane;

  nest->compute(cloud, normals);
  pest->compute(cloud, normals, roi_seed_x, roi_seed_y, plane);

  if (plane.indices.size()>3)
  {
    getInplaneTransform(plane.pt,plane.normal,bbox_base_transform);
    getBoundingBox(cloud, plane.indices, bbox_base_transform, edges, bb_min, bb_max);
    emit update_boundingbox(edges, bbox_base_transform);
    emit set_roi(bb_min, bb_max, bbox_base_transform);
  }

  //cout<<"[Sensor::detectROI] roi plane nb pts: "<<plane.indices.size()<<endl;

  m_activate_roi = true;
  m_select_roi = false;
}





