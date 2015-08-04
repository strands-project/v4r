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

#include <v4r/reconstruction/KeyframeManagementRGBD2.h>
#include <v4r/keypoints/impl/invPose.hpp>
#include <v4r/common/impl/ScopeTime.hpp>
#include <v4r/features/FeatureDetector_K_HARRIS.h>



namespace v4r
{


using namespace std;

inline bool cmpViewsInc(const pair<double,int> &i, const pair<double,int> &j)
{
  return (i.first<j.first);
}


/************************************************************************************
 * Constructor/Destructor
 */
KeyframeManagementRGBD2::KeyframeManagementRGBD2(const Parameter &p)
 : param(p), run(false), have_thread(false), nb_add(0), cnt_not_reliable_pose(0), inv_last_add_proj_pose(Eigen::Matrix4f::Identity()), last_reliable_pose(Eigen::Matrix4f::Identity()), loop_in_progress(false), have_loop_data(0)
{ 
  sqr_max_dist_tracking_view = p.max_dist_tracking_view*p.max_dist_tracking_view;
  sqr_min_dist_add_proj = p.min_dist_add_proj*p.min_dist_add_proj;
  sqr_dist_err_loop = p.dist_err_loop*p.dist_err_loop;
  model.reset(new Object());
  nest.reset(new ZAdaptiveNormals(param.n_param));
  estDesc.reset(new FeatureDetector_KD_FAST_IMGD(param.det_param));
  det = estDesc;//.reset(new FeatureDetector_K_HARRIS());// = estDesc;
  param.kd_param.compute_global_pose = true;
  param.kt_param.compute_global_pose = true;
  kpDetector.reset(new KeypointPoseDetectorRT(param.kd_param,det,estDesc));
  kpTracker.reset(new ProjLKPoseTrackerRT(param.kt_param));
}

KeyframeManagementRGBD2::~KeyframeManagementRGBD2()
{
  if (have_thread) stop();
}

/**
 * viewPointChange
 */
double KeyframeManagementRGBD2::viewPointChange(const Eigen::Vector3f &pt, const Eigen::Matrix4f &inv_pose1, const Eigen::Matrix4f &inv_pose2)
{
  Eigen::Vector3f v1 = (inv_pose1.block<3,1>(0,3)-pt).normalized();
  Eigen::Vector3f v2 = (inv_pose2.block<3,1>(0,3)-pt).normalized();

  float a = v1.dot(v2);

  if (a>0.9999) a=0;
  else a=acos(a);

  return a;
}

/**
 * cameraRotationZ
 */
double KeyframeManagementRGBD2::cameraRotationZ(const Eigen::Matrix4f &inv_pose1, const Eigen::Matrix4f &inv_pose2)
{
  Eigen::Vector3f r1 = inv_pose1.topLeftCorner<3,3>()*Eigen::Vector3f(0.,0.,1.);
  Eigen::Vector3f r2 = inv_pose2.topLeftCorner<3,3>()*Eigen::Vector3f(0.,0.,1.);

  float a = r1.dot(r2);

  if (a>0.9999) a=0;
  else a=acos(a);

  return a;
}


/**
 * getPoints3D
 */
void KeyframeManagementRGBD2::getPoints3D(const DataMatrix2D<Eigen::Vector3f> &cloud, const std::vector< std::pair<int,cv::Point2f> > &im_pts, std::vector<Eigen::Vector3f> &points)
{
  points.resize(im_pts.size());

  for (unsigned i=0; i<im_pts.size(); i++)
  {
    points[i] = cloud(int(im_pts[i].second.y+.5),int(im_pts[i].second.x+.5));
  }
}

/**
 * getGlobalCorrespondences
 */
int KeyframeManagementRGBD2::getGlobalCorrespondences(const std::vector<unsigned> glob_indices, const std::vector< std::pair<int,cv::Point2f> > &im_pts, std::vector<cv::KeyPoint> &keys, std::vector<unsigned> &points,std::vector<cv::Point2f> &im_points)
{
  int cnt=0, idx=INT_MAX;
  float dist, min_dist;
  im_points.resize(keys.size());
  float sqr_inl_dist = param.inl_dist_px*param.inl_dist_px;

  for (unsigned i=0; i<im_pts.size(); i++)
  {
    min_dist = FLT_MAX;
    const std::pair<int,cv::Point2f> &im_pt = im_pts[i];

    for (unsigned j=0; j<keys.size(); j++)
    {
      if (points[j]==UINT_MAX) 
      {
        dist = ( Eigen::Map<const Eigen::Vector2f>(&im_pt.second.x) - Eigen::Map<const Eigen::Vector2f>(&keys[j].pt.x) ).squaredNorm();
        if (dist < sqr_inl_dist) {
          idx = j;
          min_dist = dist;
        }
      }
    }

    if (min_dist < FLT_MAX)
    {
      points[idx] = glob_indices[im_pt.first];
      im_points[idx] = im_pt.second;
      cnt++;
    }
  }

  return cnt;
}



/**
 * createView
 */
bool KeyframeManagementRGBD2::createView(Shm &data, ObjectView::Ptr &view_ptr)
{
  ObjectView &view = *view_ptr;

  Eigen::Matrix4f inv_pose;
  std::vector<int> indices;
  std::vector<unsigned> points;
  std::vector<cv::Point2f> im_points;
  std::vector<Eigen::Vector3f> normals;

  invPose(data.pose, inv_pose);

  Eigen::Matrix3f R = inv_pose.topLeftCorner<3,3>();
  Eigen::Vector3f t = inv_pose.block<3, 1>(0,3);

  // detect keypoints
  det->detect(data.image, view.keys);

  // compute normals
  for (unsigned i=0; i<view.keys.size(); i++)
  {
    int x = int(view.keys[i].pt.x+.5);
    int y = int(view.keys[i].pt.y+.5);
    indices.push_back(y*data.cloud.cols+x);
  }

  // look for point tracks
  points.clear();
  points.resize(view.keys.size(),UINT_MAX);
  if (data.view_idx!=-1 && model->views.size()>0)
  {
    shm.lock();
    std::vector<unsigned> glob_points = model->views[data.view_idx]->points;
    shm.unlock();
    /*int cnt=*/getGlobalCorrespondences(glob_points, data.im_pts, view.keys, points, im_points);
    //cout<<"pt/kf tracks: "<<cnt<<endl;
  }

  nest->compute(data.cloud, indices, normals);

  // assemble model
  unsigned z=0;
  view.points.clear();
  view.points.reserve(view.keys.size());
  view.viewrays.resize(view.keys.size());
  view.cam_points.resize(view.keys.size());

  // count valid points
  int cnt=0;
  for (unsigned i=0; i<indices.size(); i++)
  {
    const Eigen::Vector3f &pt = data.cloud.data[indices[i]];
    if (!isnan(normals[i][0]) && !isnan(pt[0]))
      cnt++;
  }

  if (cnt<int(param.min_model_points))
    return false;

  // return model
  shm.lock();
  for (unsigned i=0; i<view.keys.size(); i++)
  {
    const Eigen::Vector3f &pt = data.cloud.data[indices[i]];
    if (!isnan(normals[i][0]) && !isnan(pt[0]))
    {
      view.keys[z] = view.keys[i];
      if (points[i] != UINT_MAX) {
        view.keys[z].pt = im_points[i];
        view.addPt(points[i]);
      } else view.addPt(R*pt+t, R*normals[i]);
      view.viewrays[z] = -(R*pt).normalized();
      view.cam_points[z] = pt;
      z++;
    }
  }
  shm.unlock();

  view.keys.resize(z);
  view.viewrays.resize(z);
  view.cam_points.resize(z);
  view.image = data.image;
  data.image = cv::Mat();

  estDesc->extract(view.image, view.keys, view.descs);   

  view.computeCenter();

  return true;
}

/**
 * closeLoops
 */
bool KeyframeManagementRGBD2::closeLoops()
{
  //ScopeTime t("closeLoops");

  double conf;
  Eigen::Matrix4f inv, delta_pose[2];
  std::vector< std::pair<int,cv::Point2f> > im_pts[2];

  // complete new keyframe
  kpDetector->setModel(model->views[new_view]);
  kpTracker->setModel(model->views[new_view], model->cameras[model->views[new_view]->camera_id]);

  conf = kpDetector->detect(loop_image[0], loop_cloud[0], new_pose[0]);
  if (conf>0.001) conf = kpTracker->detect(loop_image[0], loop_cloud[0], new_pose[0]); 
  else return false;

  if (conf < param.min_conf)
    return false; 

  kpTracker->getProjections(im_pts[0]);

  // complete old keyframe
  kpDetector->setModel(model->views[last_view]);
  kpTracker->setModel(model->views[last_view],model->cameras[model->views[last_view]->camera_id]);
  
  conf = kpDetector->detect(loop_image[1], loop_cloud[1], last_pose[1]);
  if (conf>0.001) conf = kpTracker->detect(loop_image[1], loop_cloud[1], last_pose[1]);
  else return false;

  if (conf < param.min_conf)
    return false;

  kpTracker->getProjections(im_pts[1]);

  // test poses
  invPose(last_pose[0], inv);
  delta_pose[0] = inv*new_pose[0];
  invPose(last_pose[1], inv);
  delta_pose[1] = inv*new_pose[1];

  double dist = (delta_pose[0].block<3,1>(0,3)-delta_pose[1].block<3,1>(0,3)).squaredNorm();
  //cout<<"dist="<<sqrt(dist)<<" (thr="<<param.dist_err_loop<<")"<<endl;
  if ( dist < sqr_dist_err_loop )
  {
    std::vector<Eigen::Vector3f> pts30(im_pts[0].size());
    std::vector<Eigen::Vector3f> pts31(im_pts[1].size());

    for (unsigned i=0; i<im_pts[0].size(); i++)
      pts30[i] = loop_cloud[0](int(im_pts[0][i].second.y+.5),int(im_pts[0][i].second.x+.5));
    for (unsigned i=0; i<im_pts[1].size(); i++)
      pts31[i] = loop_cloud[1](int(im_pts[1][i].second.y+.5),int(im_pts[1][i].second.x+.5));

    shm.lock();
    model->addProjections(*model->views[new_view], im_pts[0], pts30, cam_ids[0]);
    model->addProjections(*model->views[last_view], im_pts[1], pts31, cam_ids[1]);
    shm.unlock(); 

    //cout<<"WE HAVE A LOOP !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
    return true;
  }  

  return false;
}

/**
 * operate
 */
void KeyframeManagementRGBD2::operate()
{
  bool do_loops, have_data, have_new_view;
  std::vector<Eigen::Vector3f> pts3;
  Eigen::Matrix4f pose;
  Eigen::Matrix4f inv_pose(Eigen::Matrix4f::Identity());

  while(run)
  {
    // new views
    have_data = false;
    have_new_view = false;
    do_loops = false;

    shm.lock();
    if (nb_add != shm.nb_add)
    {
      if(!dbg.empty()) cout<<"[KeyframeManagementRGBD2::operate] create view!"<<endl;

      view.reset( new ObjectView(model.get()) );
      view->idx = model->views.size();
      shm.process_view = true;
      shm.copyTo(local_data);
      nb_add = shm.nb_add;
      have_data = true;
    }
    shm.unlock();

    if (have_data) have_new_view = createView(local_data, view);

    shm.lock();
    if (have_new_view)
    {
      view->camera_id = model->cameras.size();
      model->cameras.push_back(local_data.pose);
      model->views.push_back( view );
      model->initProjections( *view );
      shm.process_view = false;
    }
    shm.unlock();

    // loops
    shm.lock();
    if (have_loop_data==2) { 
      loop_in_progress = true; 
      do_loops = true; 
    }
    shm.unlock();
  
    if (do_loops) closeLoops();

    shm.lock();
    if (loop_in_progress) {
      loop_in_progress = false;
      have_loop_data = 0;
    }
    shm.unlock();

    if (!have_data) usleep(10000);
  }
}

/**
 * selectGuidedRandom
 */
int KeyframeManagementRGBD2::selectGuidedRandom(const Eigen::Matrix4f &pose)
{
  if (model->views.size()==0) return -1;
  else if (model->views.size()==1) return 0;

  double angle;
  std::vector< std::pair<double, int> > ranked_poses;
  Eigen::Matrix4f inv_pose, inv_pose2;
  invPose(pose,inv_pose);

  for (unsigned i=0; i<model->views.size(); i++)
  {
    const ObjectView &view = *model->views[i];
    invPose(model->cameras[view.camera_id],inv_pose2);
    angle = viewPointChange(view.center, inv_pose2, inv_pose) +
            cameraRotationZ(inv_pose2, inv_pose);

    if ( (inv_pose.block<3,1>(0,3)-inv_pose2.block<3,1>(0,3)).squaredNorm() < sqr_max_dist_tracking_view)
    {
      ranked_poses.push_back(make_pair(angle,i));
    }
  }

  std::sort(ranked_poses.begin(), ranked_poses.end() ,cmpViewsInc);

  if (ranked_poses.size()==0)
    return rand()%model->views.size()-1;

  return ranked_poses[expSelect(ranked_poses.size()-1)].second;
}


/***************************************************************************************/

/**
 * start
 */
void KeyframeManagementRGBD2::start()
{
  if (have_thread) stop();

  run = true;
  th_obectmanagement = boost::thread(&KeyframeManagementRGBD2::operate, this);  
  have_thread = true;
}

/**
 * stop
 */
void KeyframeManagementRGBD2::stop()
{
  run = false;
  th_obectmanagement.join();
  have_thread = false;
}


/**
 * addKeyframe
 */
void KeyframeManagementRGBD2::addKeyframe(const cv::Mat &image, const DataMatrix2D<Eigen::Vector3f> &cloud, const Eigen::Matrix4f &pose, int view_idx, const std::vector< std::pair<int,cv::Point2f> > &im_pts)
{
  shm.lock();
  if (!shm.process_view)
  {
    image.copyTo(shm.image);
    shm.cloud = cloud;
    shm.pose = pose;
    shm.view_idx = view_idx;
    shm.im_pts = im_pts;
    shm.nb_add++;
  }
  shm.unlock();
}

/**
 * addProjections
 */
int KeyframeManagementRGBD2::addProjections(const DataMatrix2D<Eigen::Vector3f> &cloud, const Eigen::Matrix4f &pose, int view_idx, const std::vector< std::pair<int,cv::Point2f> > &im_pts)
{
  int cam_id = -1;
  std::vector<Eigen::Vector3f> pts3(im_pts.size());

  Eigen::Matrix4f inv_pose;
  invPose(pose, inv_pose);

  if ((inv_last_add_proj_pose.block<3,1>(0,3)-inv_pose.block<3,1>(0,3)).squaredNorm() > sqr_min_dist_add_proj) 
  {
    for (unsigned i=0; i<im_pts.size(); i++)
      pts3[i] = cloud(int(im_pts[i].second.y+.5),int(im_pts[i].second.x+.5));
    
    shm.lock();
    cam_id = model->cameras.size();
    model->addProjections(*model->views[view_idx], im_pts, pts3, pose);
    shm.unlock();

    inv_last_add_proj_pose = inv_pose;
  }

  return cam_id;
}

/**
 * addLinkHyp
 */
int KeyframeManagementRGBD2::addLinkHyp1(const cv::Mat &image, const DataMatrix2D<Eigen::Vector3f> &cloud, const Eigen::Matrix4f &pose, int last_view_idx, const std::vector< std::pair<int,cv::Point2f> > &im_pts, int new_view_idx)
{
  int cam_id = -1;

  std::vector<Eigen::Vector3f> pts3(im_pts.size());

  Eigen::Matrix4f inv_pose;
  invPose(pose, inv_pose);

  for (unsigned i=0; i<im_pts.size(); i++)
    pts3[i] = cloud(int(im_pts[i].second.y+.5),int(im_pts[i].second.x+.5));
  
  shm.lock();
  cam_id = model->cameras.size();
  model->addProjections(*model->views[last_view_idx], im_pts, pts3, pose);
  shm.unlock();

  shm.lock();
  if (!loop_in_progress)
  {
    have_loop_data=1;
    last_view = last_view_idx;
    new_view = new_view_idx;
    last_pose[0] = pose;
    image.copyTo(loop_image[0]);
    loop_cloud[0] = cloud;
    cam_ids[0] = cam_id;
  }
  shm.unlock();

  inv_last_add_proj_pose = inv_pose;

  return cam_id;
}

/**
 * addLinkHyp
 */
int KeyframeManagementRGBD2::addLinkHyp2(const cv::Mat &image, const DataMatrix2D<Eigen::Vector3f> &cloud, const Eigen::Matrix4f &pose, int last_view_idx, int new_view_idx, const std::vector< std::pair<int,cv::Point2f> > &im_pts)
{
  int cam_id = -1;
  std::vector<Eigen::Vector3f> pts3(im_pts.size());

  Eigen::Matrix4f inv_pose;
  invPose(pose, inv_pose);

  for (unsigned i=0; i<im_pts.size(); i++)
    pts3[i] = cloud(int(im_pts[i].second.y+.5),int(im_pts[i].second.x+.5));
  
  shm.lock();
  cam_id = model->cameras.size();
  model->addProjections(*model->views[new_view_idx], im_pts, pts3, pose);
  shm.unlock();

  shm.lock();
  if (have_loop_data==1 && !loop_in_progress)
  {
    have_loop_data = 2;
    new_pose[1] = pose;
    image.copyTo(loop_image[1]);
    loop_cloud[1] = cloud;
    cam_ids[1] = cam_id;
  } else have_loop_data = 0;
  shm.unlock();

  inv_last_add_proj_pose = inv_pose;

  return cam_id;
}

/**
 * getTrackingModel
 */
bool KeyframeManagementRGBD2::getTrackingModel(ObjectView &view, Eigen::Matrix4f &view_pose, const Eigen::Matrix4f &current_pose, bool is_reliable_pose)
{
  bool have_update = false;
  double angle, min_angle=FLT_MAX;
  int idx=-1;
  Eigen::Matrix4f inv_pose1, inv_pose2;
  invPose(current_pose, inv_pose1);

  shm.lock();

  // look for a better view
  if (is_reliable_pose)
  {
    for (unsigned i=0; i<model->views.size(); i++)
    {
      const ObjectView &view = *model->views[i];
      invPose(model->cameras[view.camera_id],inv_pose2);
      angle = viewPointChange(view.center, inv_pose2, inv_pose1) +
              cameraRotationZ(inv_pose2, inv_pose1);

      if ( angle < min_angle && (inv_pose1.block<3,1>(0,3)-inv_pose2.block<3,1>(0,3)).squaredNorm() < sqr_max_dist_tracking_view)
      {
        min_angle = angle;
        idx = int(i);
      }
    }

    last_reliable_pose = current_pose;
  }

  if (!is_reliable_pose) cnt_not_reliable_pose++;
  else cnt_not_reliable_pose = 0;

  // random init
  if (view.idx==-1 && model->views.size()==1) 
    idx = 0;
  else if (cnt_not_reliable_pose > param.min_not_reliable_poses && model->views.size()>0) 
    idx = selectGuidedRandom(last_reliable_pose);
    //idx = rand()%model->views.size()-1;

  // return view
  if (idx>=0 && idx != view.idx)
  {
    have_update = true;
    model->views[idx]->copyTo(view);
    view_pose = model->cameras[view.camera_id];
  }

  shm.unlock();

  return have_update;
}

/**
 * reset
 */
void KeyframeManagementRGBD2::reset()
{
  stop();

  shm.nb_add = 0;
  shm.process_view = false;
  local_data.nb_add = 0;
  local_data.process_view = false;

  model.reset(new Object());

  if (!intrinsic.empty()) model->addCameraParameter(intrinsic, dist_coeffs);

  nb_add = 0;
  cnt_not_reliable_pose =0;
  inv_last_add_proj_pose = Eigen::Matrix4f::Identity();
  last_reliable_pose = Eigen::Matrix4f::Identity();
  loop_in_progress = false;
  have_loop_data = 0;
}


/**
 * setCameraParameter
 */
void KeyframeManagementRGBD2::setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs)
{
  dist_coeffs = cv::Mat_<double>();
  if (_intrinsic.type() != CV_64F)
    _intrinsic.convertTo(intrinsic, CV_64F);
  else intrinsic = _intrinsic;
  if (!_dist_coeffs.empty())
  {
    dist_coeffs = cv::Mat_<double>::zeros(1,8);
    for (int i=0; i<_dist_coeffs.cols*_dist_coeffs.rows; i++)
      dist_coeffs(0,i) = _dist_coeffs.at<double>(0,i);
  }

  reset();
 
  //model->addCameraParameter(intrinsic, dist_coeffs);
  kpTracker->setTargetCameraParameter(intrinsic, dist_coeffs);
  kpTracker->setSourceCameraParameter(intrinsic, dist_coeffs);
}

/**
 * setMinDistAddProjections
 */
void KeyframeManagementRGBD2::setMinDistAddProjections(const double &dist)
{
  param.min_dist_add_proj = dist;
  sqr_min_dist_add_proj = param.min_dist_add_proj*param.min_dist_add_proj;
}


}












