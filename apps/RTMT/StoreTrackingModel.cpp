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

#include "StoreTrackingModel.h"
#include <pcl/common/io.h>
#include <v4r/keypoints/impl/invPose.hpp>
#include <v4r/keypoints/impl/toString.hpp>
#include <v4r/common/convertCloud.h>
#include <v4r/io/filesystem.h>
#include <v4r/keypoints/CodebookMatcher.h>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>

using namespace std;


/**
 * @brief StoreTrackingModel::StoreTrackingModel
 */
StoreTrackingModel::StoreTrackingModel() :
  cmd(UNDEF), m_run(false), object_base_transform(Eigen::Matrix4f::Identity()), create_codebook(1), thr_desc_rnn(0.55)
{
}

/**
 * @brief StoreTrackingModel::~StoreTrackingModel
 */
StoreTrackingModel::~StoreTrackingModel()
{
  stop();
}



/******************************** public *******************************/

/**
 * @brief StoreTrackingModel::start
 * @param cam_id
 */
void StoreTrackingModel::start()
{
  QThread::start();
}

/**
 * @brief StoreTrackingModel::stop
 */
void StoreTrackingModel::stop()
{
  if(m_run)
  {
    m_run = false;
    this->wait();
  }
}

/**
 * @brief StoreTrackingModel::isRunning
 * @return
 */
bool StoreTrackingModel::isRunning()
{
  return m_run;
}

/**
 * @brief toreTrackingModel::set_object_base_transform
 * @param object_base_transform
 */
void StoreTrackingModel::set_object_base_transform(const Eigen::Matrix4f &_object_base_transform)
{
  object_base_transform = _object_base_transform;
}

/**
 * @brief StoreTrackingModel::set_cb_param
 * @param create_cb
 * @param rnn_thr
 */
void StoreTrackingModel::set_cb_param(bool create_cb, float rnn_thr)
{
  create_codebook = create_cb;
  thr_desc_rnn = rnn_thr;
}

/**
 * @brief StoreTrackingModel::cam_params_changed
 * @param _cam_params
 */
void StoreTrackingModel::cam_params_changed(const RGBDCameraParameter &_cam_params)
{
  intrinsic = cv::Mat_<double>::eye(3,3);

  intrinsic(0,0) = _cam_params.f[0]; intrinsic(1,1) = _cam_params.f[1];
  intrinsic(0,2) = _cam_params.c[0]; intrinsic(1,2) = _cam_params.c[1];
}

/**
 * @brief StoreTrackingModel::storeTrackingModel
 * @brief This method creates a tracking model and saves it.
 * @brief Attention: This is not thread save! Do not continue till the finished signal is published!
 * @param _folder
 * @param _cameras
 * @param _clouds
 * @param _masks
 */
void StoreTrackingModel::storeTrackingModel(const std::string &_folder, const std::string &_objectname, const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &_cameras, const boost::shared_ptr< std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > > &_clouds, const std::vector< cv::Mat_<unsigned char> > &_masks, const Eigen::Matrix4f &_object_base_transform)
{
  object_base_transform = _object_base_transform;

  if (_clouds->size()>0 && _clouds->size()==_masks.size())
  {
    folder = _folder;
    objectname = _objectname;
    cameras = _cameras;
    clouds = _clouds;
    masks = _masks;

    cmd = STORE_TRACKING_MODEL;
    start();
  }
  else
  {
    emit printStatus("No segmented point clouds available!");
    emit finishedStoreTrackingModel();
  }
}



/********************************** private ****************************************/

/**
 * @brief StoreTrackingModel::run
 * main loop
 */
void StoreTrackingModel::run()
{
  m_run=true;

  switch (cmd)
  {
  case STORE_TRACKING_MODEL:
  {
    // create tracking model
    emit printStatus("Status: Start creating the tracking model... be patient...");
    createTrackingModel();

    emit printStatus("Status: Save the tracking model...");
    // save the model
    saveTrackingModel();

    emit printStatus("Status: Saved the tracking model!");
    emit finishedStoreTrackingModel();
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
 * @brief StoreTrackingModel::createTrackingModel
 */
void StoreTrackingModel::createTrackingModel()
{
  if (clouds->size() == 0)
    return;

  Eigen::Matrix4f pose0, pose;

  cv::Mat_<cv::Vec3b> image;
  cv::Mat_<unsigned char> im_gray;
  v4r::DataMatrix2D<Eigen::Vector3f>::Ptr kp_cloud( new v4r::DataMatrix2D<Eigen::Vector3f>() );
  v4r::DataMatrix2D<Eigen::Vector3f>::Ptr kp_normals( new v4r::DataMatrix2D<Eigen::Vector3f>() );

  v4r::ZAdaptiveNormals::Parameter n_param;
  n_param.adaptive = true;
  v4r::ZAdaptiveNormals::Ptr nest(new v4r::ZAdaptiveNormals(n_param));

  //v4r::FeatureDetector_KD_FAST_IMGD::Parameter param(10000, 1.44, 2, 17);
  v4r::FeatureDetector_KD_FAST_IMGD::Parameter param(10000, 1.2, 6, 13);
  //v4r::FeatureDetector_KD_FAST_IMGD::Parameter param(10000, 1.32, 4, 15);
  param.do_feature_selection = true;
  keyDet.reset(new v4r::FeatureDetector_KD_FAST_IMGD(param));
  keyDesc = keyDet;

  model.reset(new v4r::ArticulatedObject());
  model->addCameraParameter(intrinsic, dist_coeffs);

  //detectCoordinateSystem(pose0);
  pose0 = object_base_transform;

  for (unsigned i=0; i<clouds->size(); i++)
  {
    v4r::convertImage(*clouds->at(i).second, image);

    if( image.type() != CV_8U ) cv::cvtColor( image, im_gray, CV_RGB2GRAY );
    else im_gray = image;

    pose = cameras[clouds->at(i).first];

    v4r::convertCloud(*clouds->at(i).second, *kp_cloud);
    nest->compute(*kp_cloud, *kp_normals);

    addObjectView(*kp_cloud, *kp_normals, im_gray, masks[i], pose*pose0, *model);

  }

  if (create_codebook==1)
  {
    v4r::CodebookMatcher cm = v4r::CodebookMatcher( v4r::CodebookMatcher::Parameter(thr_desc_rnn) );

    for (unsigned i=0; i<model->views.size(); i++)
      cm.addView(model->views[i]->descs,i);

    cm.createCodebook(model->cb_centers, model->cb_entries);
  }

  keyDet = v4r::FeatureDetector::Ptr();
  keyDesc = v4r::FeatureDetector::Ptr();
}

/**
 * @brief StoreTrackingModel::saveTrackingModel
 */
void StoreTrackingModel::saveTrackingModel()
{
  if (model.get()==0)
      return;

  boost::filesystem::create_directories(folder);

  std::cout << "Writing model to " << folder << "/models/" << objectname << "/tracking_model.ao" << std::endl;

  std::string model_name = folder + "/models/" + objectname + "/tracking_model.ao";

  v4r::io::createDirForFileIfNotExist(model_name);
  v4r::io::write(model_name, model);

  std::cout << "Tracking model saved!" << std::endl;

  model = v4r::ArticulatedObject::Ptr();
}


/**
 * @brief StoreTrackingModel::addObjectView
 * @param cloud
 * @param normals
 * @param im
 * @param mask
 * @param pose
 * @param model
 */
void StoreTrackingModel::addObjectView(const v4r::DataMatrix2D<Eigen::Vector3f> &cloud, const v4r::DataMatrix2D<Eigen::Vector3f> &normals, const cv::Mat_<unsigned char> &im, const cv::Mat_<unsigned char> &mask, const Eigen::Matrix4f &pose, v4r::ArticulatedObject &model)
{
  // get and transform 3d points
  Eigen::Matrix4f inv_pose;
  Eigen::Vector3f pt_model, n_model, vr_model;
  static const unsigned MIN_KEYPOINTS = 20;

  v4r::invPose(pose, inv_pose);

  Eigen::Matrix3f R = inv_pose.topLeftCorner<3,3>();
  Eigen::Vector3f t = inv_pose.block<3, 1>(0,3);

  std::vector<cv::KeyPoint> keys;
  cv::Mat descs;
  unsigned cnt=0;

  // detect keypoints
  keyDet->detect(im, keys);
  keyDesc->extract(im, keys, descs);

  for (unsigned i=0; i<keys.size(); i++)
  {
    cv::KeyPoint &key = keys[i];

    int idx = int(key.pt.y+.5)*cloud.cols+int(key.pt.x+.5);

    const Eigen::Vector3f &pt =  cloud[idx];
    const Eigen::Vector3f &n = normals[idx];

    if(!isnan(pt[0]) && !isnan(n[0]) && mask(key.pt.y,key.pt.x)>128)
      cnt++;
  }


  if (cnt<MIN_KEYPOINTS) return;

  v4r::ObjectView &view = model.addObjectView(pose, im);


  for (unsigned i=0; i<keys.size(); i++)
  {
    cv::KeyPoint &key = keys[i];

    int idx = int(key.pt.y+.5)*cloud.cols+int(key.pt.x+.5);

    const Eigen::Vector3f &pt =  cloud[idx];
    const Eigen::Vector3f &n = normals[idx];

    if(!isnan(pt[0]) && !isnan(n[0]) && mask(key.pt.y,key.pt.x)>128)
    {
      pt_model = R*pt + t;
      n_model = (R*n).normalized();
      vr_model = -(R*pt).normalized();

      view.add(keys[i], &descs.at<float>(i,0), descs.cols, pt_model, n_model, vr_model);
    }
  }

  view.descs.copyTo(descs);
  view.descs = descs;
}


/**
 * @brief StoreTrackingModel::detectCoordinateSystem
 * @param pose
 */
void StoreTrackingModel::detectCoordinateSystem(Eigen::Matrix4f &pose)
{
  pose.setIdentity();

  pcl::PointXYZRGB pt;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr gcloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::PointCloud<pcl::PointXYZRGB> &gc = *gcloud;
  Eigen::Vector3f centroid(0.,0.,0.);
  int cnt=0;

  for (unsigned i=0; i<clouds->size(); i++)
  {
    pcl::PointCloud<pcl::PointXYZRGB> &cloud = *clouds->at(i).second;
    cv::Mat_<unsigned char> &mask = masks[i];
    Eigen::Matrix4f &pose = cameras[clouds->at(i).first];
    Eigen::Matrix4f inv_pose;

    v4r::invPose(pose,inv_pose);

    Eigen::Matrix3f R = inv_pose.topLeftCorner<3,3>();
    Eigen::Vector3f t = inv_pose.block<3,1>(0,3);

    int xmin = INT_MAX, ymin = INT_MAX;
    int xmax = 0, ymax = 0;

    for (int v=0; v<mask.rows; v++)
    {
      for (int u=0; u<mask.cols; u++)
      {
        if (mask(v,u)<128) continue;
        if (u<xmin) xmin = u;
        if (u>xmax) xmax = u;
        if (v<ymin) ymin = v;
        if (v>ymax) ymax = v;
      }
    }

    int deltax_h = (xmax-xmin)/2.;
    int deltay_h = (ymax-ymin)/2.;
   
    if (deltax_h<=0 || deltay_h<=0) continue;

    xmin = (xmin-deltax_h>=0?xmin-deltax_h:0);
    ymin = (ymin-deltay_h>=0?ymin-deltay_h:0);
    xmax = (xmax+deltax_h<mask.cols?xmax+deltax_h:mask.cols-1);
    ymax = (ymax+deltay_h<mask.rows?ymax+deltay_h:mask.rows-1);

    for (int v=ymin; v<=ymax; v++)
    {
      for (int u=xmin; u<=xmax; u++)
      {
        const pcl::PointXYZRGB &pt1 = cloud(u,v);

        if (isnan(pt1.x) || isnan(pt1.y) || isnan(pt1.z)) continue;

        if (mask(v,u)<128)
        {
          pt.getVector3fMap() = R*pt1.getVector3fMap() + t;
          gc.points.push_back(pt);
        }
        else
        {
          centroid += R*pt1.getVector3fMap() + t;
          cnt++;
        }
      }
    }
  }

  gc.width = gc.points.size();
  gc.height = 1;
  gc.is_dense = true;

  if (cnt<10 || gc.points.size()<10)
    return;

  centroid /= float(cnt);

  // detect dominat plane
  Eigen::VectorXf plane_coeffs = Eigen::VectorXf(4);
  std::vector<int> inliers;

  pcl::SampleConsensusModelPlane<pcl::PointXYZRGB>::Ptr
    model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZRGB> (gcloud));

  pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac (model_p);

  ransac.setDistanceThreshold (.05);
  ransac.computeModel();
  ransac.getInliers(inliers);

  if (inliers.size()<10)
    return;

  model_p->optimizeModelCoefficients(inliers, plane_coeffs, plane_coeffs);

  // get coordinate system
  Eigen::Vector3f px, py;
  Eigen::Vector3f pz = plane_coeffs.segment<3>(0);

  if (centroid.dot(pz) > 0) pz *= -1;
  px = (Eigen::Vector3f(1,0,0).cross(pz)).normalized();
  py = pz.cross(px);

  pose.block<3,1>(0,0) = px;
  pose.block<3,1>(0,1) = py;
  pose.block<3,1>(0,2) = pz;
  pose.block<3,1>(0,3) = centroid;
}









