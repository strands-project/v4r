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


#include <v4r/camera_tracking_and_mapping/TSFMapping.hh>
#include <v4r/keypoints/impl/invPose.hpp>
#include <v4r/common/convertImage.h>
#include "pcl/common/transforms.h"

#include "opencv2/highgui/highgui.hpp"
#include <v4r/reconstruction/impl/projectPointToImage.hpp>
#include <v4r/keypoints/impl/toString.hpp>
#include <v4r/keypoints/impl/warpPatchHomography.hpp>
#include <v4r/camera_tracking_and_mapping/TSFDataIntegration.hh>

//#define DBG_OUTPUT


namespace v4r
{


using namespace std;





/************************************************************************************
 * Constructor/Destructor
 */
TSFMapping::TSFMapping(const Parameter &p)
 : run(false), have_thread(false), data(NULL)
{ 
  setParameter(p);
}

/**
 * @brief TSFMapping::~TSFMapping
 */
TSFMapping::~TSFMapping()
{
  if (have_thread) stop();
}

/**
 * operate
 */
void TSFMapping::operate()
{
  bool have_todo;

  while(run)
  {
    have_todo = false;
    data->lock();
    if (data->map_frames.size()>0)
    {
      map_frames.push_back(data->map_frames.front());
      data->map_frames.pop();
      have_todo = true;
    }
    data->unlock();

    if (have_todo)
    {
      //v4r::ScopeTime t("[Mapping]");
      map_frames.back()->idx = map_frames.size()-1;
      TSFData::convert(map_frames.back()->sf_cloud, image0);
      cv::cvtColor( image0, im_gray0, CV_BGR2GRAY );
      TSFDataIntegration::computeNormals(map_frames.back()->sf_cloud, 2);
      initKeypoints( im_gray0, *map_frames.back() );

      if (map_frames.size()>=2)
      {
        for (int i=0; i<param.nb_tracked_frames; i++)
        {
          if (i+2<=(int)map_frames.size() && map_frames[map_frames.size()-i-1]->have_track)
          {
            TSFData::convert(map_frames[map_frames.size()-i-2]->sf_cloud, image);
            cv::cvtColor( image, im_gray1, CV_BGR2GRAY );
            addFeatureLinks(*map_frames.back(), *map_frames[map_frames.size()-i-2], im_gray0, im_gray1, map_frames.back()->pose, map_frames[map_frames.size()-i-2]->pose, false);
            cout<<"  Have track ("<<map_frames[map_frames.size()-i-2]->idx<<"-"<<map_frames.back()->idx<<")"<<endl;
          }
        }

        if (param.detect_loops) addLoops();
        cout<<"  Number keyframes: "<<map_frames.size()<<endl;
      }
      
      data->lock();
      // copy back results????
      data->unlock();
    }

    if (!have_todo) usleep(10000);
  }
}



/**
 * @brief TSFMapping::getPoints3D
 * @param cloud
 * @param points
 * @param points3d
 * @param normals
 */
void TSFMapping::getPoints3D(const v4r::DataMatrix2D<Surfel> &cloud, const std::vector<cv::Point2f> &points, std::vector<Eigen::Vector3f> &points3d, std::vector<Eigen::Vector3f> &normals)
{
  points3d.resize(points.size());
  normals.resize(points.size());

  for (unsigned i=0; i<points.size(); i++)
  {
    const cv::Point2f &pt = points[i];
    if (pt.x>=0 && pt.y>=0 && pt.x<cloud.cols && pt.y<cloud.rows)
    {
      const Surfel &s = cloud((int)pt.y,(int)pt.x);
      points3d[i] = s.pt;
      normals[i] = s.n;
    }
    else points3d[i] = Eigen::Vector3f(std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN());
  }
}

/**
 * @brief TSFOptimizeBundle::filterValidPoints3D
 * @param points
 * @param points3d
 * @param normals
 */
void TSFMapping::filterValidPoints3D(std::vector<cv::Point2f> &points, std::vector<Eigen::Vector3f> &points3d, std::vector<Eigen::Vector3f> &normals)
{
  if (points.size()!=points3d.size())
    return;

  int z=0;
  for (unsigned i=0; i<points.size(); i++)
  {
    const Eigen::Vector3f &pt3 = points3d[i];
    const Eigen::Vector3f &n = normals[i];
    if (!isnan(pt3[0]) && !isnan(pt3[1]) && !isnan(pt3[2]) && !isnan(n[0] && !isnan(n[1]) && !isnan(n[2])))
    {
      if (n.dot(-pt3.normalized()) > cos_rad_max_dev_vr_normal)
      {
        points[z] = points[i];
        points3d[z] = points3d[i];
        normals[z] = normals[i];
        z++;
      }
    }
  }
  points.resize(z);
  points3d.resize(z);
  normals.resize(z);
}

/**
 * @brief TSFMapping::getKeys3D
 * @param cloud
 * @param keys
 * @param keys3d
 */
void TSFMapping::getKeys3D(const v4r::DataMatrix2D<Surfel> &cloud, const std::vector<cv::KeyPoint> &keys, std::vector<Eigen::Vector3f> &keys3d)
{
  keys3d.resize(keys.size());

  for (unsigned i=0; i<keys.size(); i++)
  {
    const cv::Point2f &pt = keys[i].pt;
    if (pt.x>=0 && pt.y>=0 && pt.x<cloud.cols && pt.y<cloud.rows)
    {
      const Surfel &s = cloud((int)pt.y,(int)pt.x);
      keys3d[i] = s.pt;
    }
    else keys3d[i] = Eigen::Vector3f(std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN());
  }
}

/**
 * @brief TSFMapping::filterValidKeys3D
 * @param keys
 * @param keys3d
 */
void TSFMapping::filterValidKeys3D(std::vector<cv::KeyPoint> &keys, std::vector<Eigen::Vector3f> &keys3d)
{
  if (keys.size()!=keys3d.size())
    return;

  int z=0;
  for (unsigned i=0; i<keys.size(); i++)
  {
    const Eigen::Vector3f &pt3 = keys3d[i];
    if (!isnan(pt3[0]) && !isnan(pt3[1]) && !isnan(pt3[2]))
    {
        keys[z] = keys[i];
        keys3d[z] = keys3d[i];
        z++;
    }
  }
  keys.resize(z);
  keys3d.resize(z);
}

/**
 * @brief TSFMapping::initKeypoints
 */
void TSFMapping::initKeypoints(const cv::Mat_<unsigned char> &im, TSFFrame &frame0)
{
  cv::goodFeaturesToTrack(im, frame0.points, param.max_count, 0.01, 10, cv::Mat(), 3, 0, 0.04);
  getPoints3D(frame0.sf_cloud, frame0.points, frame0.points3d, frame0.normals);
  filterValidPoints3D(frame0.points, frame0.points3d, frame0.normals);
  frame0.projections.resize(frame0.points.size());
  detector->detect(im, frame0.keys);
  getKeys3D(frame0.sf_cloud, frame0.keys, frame0.keys3d);
  filterValidKeys3D(frame0.keys, frame0.keys3d);
  descEstimator->extract(im, frame0.keys, frame0.descs);
}

/**
 * @brief addProjectionsPLK
 * @param frame_idx
 * @param cv_points
 * @param converged
 * @param projs
 */
int TSFMapping::addProjectionsPLK(int frame_idx, const v4r::DataMatrix2D<Surfel> &sf_cloud, const std::vector<cv::Point2f> &cv_points, const std::vector<int> &converged, std::vector< std::vector< v4r::triple<int, cv::Point2f, Eigen::Vector3f > > > &projs)
{
  int cnt=0;
  Eigen::Vector3f pt3;
  for (unsigned i=0; i<cv_points.size(); i++)
  {
    if (converged[i]==1)
    {
      const cv::Point2f &pt = cv_points[i];
      if (pt.x>=0 && pt.y>=0 && pt.x<sf_cloud.cols && pt.y<sf_cloud.rows)
        pt3 = sf_cloud((int)pt.y,(int)pt.x).pt;
      else pt3 = Eigen::Vector3f(std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN());
      projs[i].push_back( v4r::triple<int,cv::Point2f,Eigen::Vector3f>(frame_idx, pt, pt3) );
      cnt++;
    }
  }
  return cnt;
}

/**
 * @brief TSFMapping::addFeatureLinks
 */
void TSFMapping::addFeatureLinks(TSFFrame &frame0, TSFFrame &frame1, const cv::Mat_<unsigned char> &im0, const cv::Mat_<unsigned char> &im1, const Eigen::Matrix4f &pose0, const Eigen::Matrix4f &pose1, bool is_loop)
{
  Eigen::Matrix4f inv_pose0, inv_pose1, inc_pose;
  std::vector<cv::Point2f> refined_projs;
  std::vector<int> converged;

  v4r::invPose(pose0, inv_pose0);
  inc_pose = pose1*inv_pose0;
  refineLK(frame0, frame1, im0, im1, inc_pose, refined_projs, converged);
  if (addProjectionsPLK(frame1.idx, frame1.sf_cloud, refined_projs, converged, frame0.projections) > 5)
  {
    if (is_loop)
      frame0.loop_links.push_back(frame1.idx);
    else frame0.bw_link = frame1.idx;
  }

  v4r::invPose(pose1, inv_pose1);
  inc_pose = pose0*inv_pose1;
  refineLK(frame1, frame0, im1, im0, inc_pose, refined_projs, converged);
  if (addProjectionsPLK(frame0.idx, frame0.sf_cloud, refined_projs, converged, frame1.projections) > 5)
  {
    if (is_loop)
      frame1.loop_links.push_back(frame0.idx);
    else frame1.fw_link = frame0.idx;
  }
}

/**
 * @brief TSFMapping::ransacPose
 * @param query
 * @param train
 * @param matches
 * @param pose
 * @return
 */
bool TSFMapping::ransacPose(const std::vector<Eigen::Vector3f> &query, const std::vector<cv::KeyPoint> &query_keys, const std::vector<Eigen::Vector3f> &train, const std::vector<std::vector< cv::DMatch > > &matches, Eigen::Matrix4f &pose, int &nb_inls)
{
  std::vector<int> inliers;
  std::vector<Eigen::Vector3f> pts3d0, pts3d1;
  std::vector<cv::Point2f> pts1;
  for (unsigned i=0; i<matches.size(); i++)
  {
    if (matches[i].size()>=2)
    {
      const cv::DMatch &m = matches[i][0];
      if (m.distance/matches[i][1].distance < param.nnr)
      {
        pts3d0.push_back(train[m.trainIdx]);
        pts3d1.push_back(query[m.queryIdx]);
        pts1.push_back(query_keys[m.queryIdx].pt);
      }
    }
  }
  if (pts3d0.size()>=5)
  {
    int nb_iter = pnp->ransac(pts3d0, pts1, pts3d1, pose, inliers);
    nb_inls = inliers.size();
    if (nb_iter < (int)param.pnp.max_rand_trials)
      return true;
  }
  return false;
}

/**
 * @brief TSFMapping::warpImage
 * @param frame1
 * @param im0
 * @param pose
 * @param im_warped
 */
void TSFMapping::warpImage(const TSFFrame &frame1, const cv::Mat_<unsigned char> &im0, const Eigen::Matrix4f &pose, cv::Mat_<unsigned char> &im_warped, Eigen::Matrix3f &H)
{
  Eigen::Vector3d pt3(0.,0.,0.);
  Eigen::Vector3d n(0.,0.,0.);
  for (unsigned i=0; i<frame1.points3d.size(); i++)
  {
    pt3 += frame1.points3d[i].cast<double>();
    n += frame1.normals[i].cast<double>();
  }
  if (frame1.points3d.size()>0)
  {
    pt3/=(double)frame1.points3d.size();
    n.normalize();
  }

  double d = n.transpose()*pt3;
  H = pose.topLeftCorner<3,3>() + 1./d*pose.block<3,1>(0,3)*(n.transpose().cast<float>());
  H = C * H * C.inverse();

  cv::Mat_<float> cv_H(3,3);
  for (int v=0; v<3; v++)
    for (int u=0; u<3; u++)
      cv_H(v,u) = H(v,u);

  cv::warpPerspective(im0, im_warped, cv_H, im0.size());
}

/**
 * @brief TSFMapping::refineLK
 * @param frame0
 * @param frame1
 * @param pose01
 */
#ifdef DBG_OUTPUT
int im_cnt = 0;
#endif
bool TSFMapping::refineLK(const TSFFrame &frame0, const TSFFrame &frame1, const cv::Mat_<unsigned char> &im0, const cv::Mat_<unsigned char> &im1, Eigen::Matrix4f &pose01, std::vector<cv::Point2f> &refined1, std::vector<int> &converged1)
{
  Eigen::Matrix3f pose_R = pose01.topLeftCorner<3, 3>();
  Eigen::Vector3f pose_t = pose01.block<3,1>(0, 3);
  Eigen::Vector3f pt3;
  std::vector<float> depth(frame0.points3d.size(), std::numeric_limits<float>::quiet_NaN());
  std::vector<Eigen::Vector3f> pts3d(frame0.points3d.size());
  std::vector<Eigen::Vector3f> normals(frame0.points3d.size());
  std::vector<unsigned char> status;
  std::vector<float> error;
  std::vector<int> inliers;
  std::vector<int> lt(frame0.points3d.size());
  std::vector<cv::Point2f> cv_points0(frame0.points3d.size());
  std::vector<cv::Point2f> cv_points1(frame0.points3d.size());
  refined1.resize(frame0.points3d.size());
  converged1.assign(frame0.points3d.size(),0);
  Eigen::Matrix3f H;

  warpImage(frame1, im0, pose01, im_warped, H);

  for (unsigned i=0; i<frame0.points3d.size(); i++)
  {
    mapPoint(frame0.points[i], H, cv_points0[i]);
    pt3 = pose_R*frame0.points3d[i] + pose_t;
    v4r::projectPointToImage(&pt3[0], intrinsic.ptr<double>(), &cv_points1[i].x);
    lt[i] = i;
  }

  cv::calcOpticalFlowPyrLK(im_warped, im1, cv_points0, cv_points1, status, error, param.win_size, param.max_level, param.termcrit, cv::OPTFLOW_USE_INITIAL_FLOW, 0.001 );

  int z=0;
  for (unsigned i=0; i<cv_points0.size(); i++)
  {
    if ( status[i] && error[i]<param.max_error && cv_points1[i].x>=0 && cv_points1[i].x<frame1.sf_cloud.cols && cv_points1[i].y>=0 && cv_points1[i].y<frame1.sf_cloud.rows)
    {
      lt[z] = lt[i];
      cv_points0[z] = cv_points0[i];
      cv_points1[z] = cv_points1[i];
      depth[z] = frame1.sf_cloud(cv_points1[i].y,cv_points1[i].x).pt[2];
      pts3d[z] = frame0.points3d[i];
      normals[z] = frame0.normals[i];
      z++;
    }
  }

  cv_points0.resize(z);
  cv_points1.resize(z);
  depth.resize(z);
  pts3d.resize(z);
  lt.resize(z);

  if (cv_points1.size()>=5)
  {
    //std::vector<cv::Point2f> tmp_pt = cv_points1;
    if (param.refine_plk)
    {
      plk.setNumberOfThreads(1);
      plk.useInitialFlow(1);
      plk.setSourceImage(im0, Eigen::Matrix4f::Identity());
      plk.setTargetImage(im1, pose01);
      plk.refineImagePoints(pts3d, normals, cv_points1, converged);
    }

    int cnt = pnp->ransac(pts3d, cv_points1, pose01, inliers, depth);

    for (unsigned i=0; i<inliers.size(); i++)
    {
      refined1[lt[inliers[i]]] = cv_points1[inliers[i]];
      converged1[lt[inliers[i]]] = 1;
    }

    #ifdef DBG_OUTPUT
    cv::Mat_<unsigned char> dbg0, dbg1;
    im0.copyTo(dbg0);
    im1.copyTo(dbg1);
    for (unsigned i=0; i<inliers.size(); i++)
    {
      cv::line(dbg1, cv_points0[inliers[i]], cv_points1[inliers[i]],0);
      cv::line(dbg1, cv_points1[inliers[i]], cv_points1[inliers[i]],255);
      //cv::line(dbg1, tmp_pt[inliers[i]], cv_points1[inliers[i]],0);
      //cv::line(dbg1, cv_points1[inliers[i]], cv_points1[inliers[i]],255);
    }
    for (unsigned i=0; i<frame0.points.size(); i++)
    {
      //cv::circle(dbg0, frame0.points[i], 2, 255);
      cv::line(dbg0, frame0.points[i], frame0.points[i],255);
    }
    cv::imwrite(std::string("log/im0_")+v4r::toString(im_cnt)+std::string(".jpg"), dbg0);
    cv::imwrite(std::string("log/im1_")+v4r::toString(im_cnt)+std::string(".jpg"), dbg1);
    cv::imwrite(std::string("log/imwarped_")+v4r::toString(im_cnt)+std::string(".jpg"), im_warped);
    im_cnt++;
    #endif

    if (cnt<(int)param.pnp.max_rand_trials)
      return true;
    return false;
  }
  return false;
}

/**
 * @brief TSFMapping::addLoops
 */
void TSFMapping::addLoops()
{
  const static double COS_START_ANGLE = cos(45.*M_PI/180.);
  double cos_max_delta_angle_loop = cos(param.max_delta_angle_loop*M_PI/180.);
  double sqr_max_cam_distance = param.max_cam_dist_loop*param.max_cam_dist_loop;
  double cos_max_delta_angle_eq_pose = cos(param.max_delta_angle_eq_pose*M_PI/180.);
  double sqr_max_cam_dist_eq_pose = param.max_cam_dist_eq_pose*param.max_cam_dist_eq_pose;
  int nb_inls;

  std::vector<cv::Point2f> refined0, refined1;
  std::vector<int> converged0, converged1;

  bool have_start=false;
  Eigen::Matrix4f inv_pose0, inv_pose1, pose01, pose20;
  TSFFrame &frame0 = *map_frames.back();
  v4r::invPose(frame0.pose, inv_pose0);

  for (int i=map_frames.size()-2; i>=0; i--)
  {
    v4r::invPose(map_frames[i]->pose, inv_pose1);
    if (!have_start && inv_pose0.block<3,1>(0,2).dot(inv_pose1.block<3,1>(0,2)) < COS_START_ANGLE )
      have_start = true;

    if (have_start)
    {
      double cosa = inv_pose0.block<3,1>(0,2).dot(inv_pose1.block<3,1>(0,2));
      if (cosa>cos_max_delta_angle_loop && (inv_pose0.block<3,1>(0,3)-inv_pose1.block<3,1>(0,3)).squaredNorm() < sqr_max_cam_distance)
      {
        TSFFrame &frame1 = *map_frames[i];
        matches.clear();
        if (frame1.descs.rows>0 && frame0.descs.rows>0)
          matcher.knnMatch(frame1.descs, frame0.descs, matches, 2); //query=1, train=0 -> pose01
        bool ok01 = ransacPose(frame1.keys3d, frame1.keys, frame0.keys3d, matches, pose01, nb_inls);
        if (ok01)
        {
          int link2 = -1;
          if (frame1.fw_link>=0)
          {
            TSFFrame &frame2 = *map_frames[frame1.fw_link];
            matches.clear();
            if (frame0.descs.rows>0 && frame2.descs.rows>0)
                matcher.knnMatch(frame0.descs, frame2.descs, matches, 2); //query=0, train=2 -> pose20
            bool ok20 = ransacPose(frame0.keys3d, frame0.keys, frame2.keys3d, matches, pose20, nb_inls);
            if (ok20) link2 = frame1.fw_link;
          }
          if (link2==-1 && frame1.bw_link>=0)
          {
            TSFFrame &frame2 = *map_frames[frame1.bw_link];
            matches.clear();
            if (frame0.descs.rows>0 && frame2.descs.rows>0)
              matcher.knnMatch(frame0.descs, frame2.descs, matches, 2); //query=0, train=2 -> pose20
            bool ok20 = ransacPose(frame0.keys3d, frame0.keys, frame2.keys3d, matches, pose20, nb_inls);
            if (ok20) link2 = frame1.bw_link;
          }
          if (link2>=0)
          {
            TSFFrame &frame2 = *map_frames[link2];
            TSFData::convert(frame1.sf_cloud, image);
            cv::cvtColor( image, im_gray1, CV_BGR2GRAY );
            TSFData::convert(frame2.sf_cloud, image);
            cv::cvtColor( image, im_gray2, CV_BGR2GRAY );
            if (refineLK(frame0, frame1, im_gray0, im_gray1, pose01, refined0, converged0) && refineLK(frame2, frame0, im_gray2, im_gray0, pose20, refined1, converged1))
            {
              Eigen::Matrix4f pose0_loop = pose20*frame2.pose*inv_pose1*pose01*frame0.pose;
              double cosa = frame0.pose.block<3,1>(0,2).dot(pose0_loop.block<3,1>(0,2));
              cout<< "  Closed loop error: "<<(acos(cosa)*180./M_PI)<<"°, "<<(frame0.pose.block<3,1>(0,3)-pose0_loop.block<3,1>(0,3)).norm()<<"m"<<endl;
              if (cosa>cos_max_delta_angle_eq_pose && (frame0.pose.block<3,1>(0,3)-pose0_loop.block<3,1>(0,3)).squaredNorm() < sqr_max_cam_dist_eq_pose)
              {
                cout<<"  Found loop ("<<frame0.idx<<"-"<<frame1.idx<<"-"<<frame2.idx<<")"<<endl;
                if (addProjectionsPLK(frame1.idx, frame1.sf_cloud, refined0, converged0, frame0.projections) > 5)
                  frame0.loop_links.push_back(frame1.idx);
                if (addProjectionsPLK(frame0.idx, frame0.sf_cloud, refined1, converged1, frame2.projections) > 5)
                  frame2.loop_links.push_back(frame0.idx);
              }
            }
          }
        }
      }
    }
  }
}






/***************************************************************************************/

/**
 * start
 */
void TSFMapping::start()
{
  if (data==NULL)
    throw std::runtime_error("[TSFMapping::start] No data available! Did you call 'setData'?");
  if (detector.get()==0 || descEstimator.get()==0)
    throw std::runtime_error("[TSFMapping::start] No keypoint detectors are available");

  if (have_thread) stop();

  run = true;
  th_obectmanagement = boost::thread(&TSFMapping::operate, this);  
  have_thread = true;
}

/**
 * stop
 */
void TSFMapping::stop()
{
  run = false;
  th_obectmanagement.join();
  have_thread = false;
}




/**
 * reset
 */
void TSFMapping::reset()
{
  stop();
  map_frames.clear();
}

/**
 * @brief TSFMapping::optimizeMap
 */
void TSFMapping::optimizeMap()
{
  stop();
  ba.optimize(map_frames);
}

/**
 * @brief TSFMapping::transformMap
 * @param transform
 */
void TSFMapping::transformMap(const Eigen::Matrix4f &transform)
{
  stop();
  for (unsigned i=0; i<map_frames.size(); i++)
  {
    v4r::TSFFrame &frame = *map_frames[i];
    frame.pose = frame.pose*transform;
  }
}


/**
 * setCameraParameter
 */
void TSFMapping::setCameraParameter(const cv::Mat &_intrinsic)
{
  if (_intrinsic.type() != CV_64F)
    _intrinsic.convertTo(intrinsic, CV_64F);
  else intrinsic = _intrinsic;

  C = Eigen::Matrix3f::Identity();
  C(0,0) = intrinsic(0,0);
  C(1,1) = intrinsic(1,1);
  C(0,2) = intrinsic(0,2);
  C(1,2) = intrinsic(1,2);

  plk.setSourceCameraParameter(intrinsic, cv::Mat_<double>::zeros(1,8));
  plk.setTargetCameraParameter(intrinsic, cv::Mat_<double>::zeros(1,8));
  ba.setCameraParameter(intrinsic, cv::Mat_<double>::zeros(1,8));
  if (!intrinsic.empty()) pnp->setCameraParameter(intrinsic,cv::Mat());

  reset();
}

/**
 * @brief TSFMapping::setParameter
 * @param p
 */
void TSFMapping::setParameter(const Parameter &p)
{
  param = p;
  cos_rad_max_dev_vr_normal = cos(param.max_dev_vr_normal*M_PI/180.);
  pnp.reset(new v4r::RansacSolvePnPdepth(param.pnp));
  plk.setParameter(param.plk_param);
  ba.setParameter(param.ba);
  if(!intrinsic.empty()) pnp->setCameraParameter(intrinsic, cv::Mat());
}

/**
 * @brief TSFMapping::setDetectors
 * @param _detector
 * @param _descEstimator
 */
void TSFMapping::setDetectors(const FeatureDetector::Ptr &_detector, const FeatureDetector::Ptr &_descEstimator)
{
  detector = _detector;
  descEstimator = _descEstimator;
}


}












