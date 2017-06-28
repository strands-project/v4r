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


#include <v4r/camera_tracking_and_mapping/TSFPoseTrackerKLT.hh>
#include <v4r/keypoints/impl/invPose.hpp>
#include <v4r/common/convertImage.h> 
#include "pcl/common/transforms.h"
#include "opencv2/calib3d/calib3d.hpp"

#include "opencv2/highgui/highgui.hpp"


namespace v4r
{


using namespace std;



/************************************************************************************
 * Constructor/Destructor
 */
TSFPoseTrackerKLT::TSFPoseTrackerKLT(const Parameter &p)
 : param(p), run(false), have_thread(false), data(NULL)
{ 
  pnp.reset(new v4r::RansacSolvePnPdepth(param.rt));
}

TSFPoseTrackerKLT::~TSFPoseTrackerKLT()
{
  if (have_thread) stop();
}

/**
 * operate
 */
void TSFPoseTrackerKLT::operate()
{
  bool have_todo;

  cv::Mat im_gray;
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  std::vector<cv::Point2f> points;
  std::vector<Eigen::Vector3f> points3d;
  Eigen::Matrix4f pose;
  uint64_t timestamp = 0;

  while(run)
  {
    have_todo = false;

    data->lock();
    if (data->need_init && data->cloud.points.size() > 0 && data->timestamp!=data->kf_timestamp)
    {
      have_todo = true;
      cloud = data->cloud;
      pose = data->pose;
      timestamp = data->timestamp;
      data->gray.copyTo(im_gray);
    }
    data->unlock();

    if (have_todo)
    {
      cv::goodFeaturesToTrack(im_gray, points, param.max_count, 0.01, 10, cv::Mat(), 3, 0, 0.04);
      getPoints3D(cloud, points, points3d);
      filterValidPoints3D(points, points3d);

      data->lock();
      data->init_points = points.size();
      data->lk_flags = 0;
      im_gray.copyTo(data->prev_gray);
      data->points[0] = points;
      data->points3d[0] = points3d;
      data->kf_pose = pose;
      data->kf_timestamp = timestamp;
      if (data->points[0].size() > param.max_count*param.pcent_reinit)
        data->need_init = false;
      data->unlock();
    }

    if (!have_todo) usleep(10000);
  }
}

/**
 * @brief TSFPoseTrackerKLT::getImage
 * @param cloud
 * @param im
 */
void TSFPoseTrackerKLT::getImage(const v4r::DataMatrix2D<Surfel> &cloud, cv::Mat &im)
{
  im = cv::Mat_<cv::Vec3b>(cloud.rows, cloud.cols);
  for (int v=0; v<cloud.rows; v++)
  {
    for (int u=0; u<cloud.cols; u++)
    {
      cv::Vec3b &c = im.at<cv::Vec3b>(v,u);
      const Surfel &s = cloud(v,u);
      c[0] = s.b;
      c[1] = s.g;
      c[2] = s.r;
    }
  }
}


/**
 * @brief TSFPoseTrackerKLT::getPoints3D
 * @param cloud
 * @param points
 * @param points3d
 */
void TSFPoseTrackerKLT::getPoints3D(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const std::vector<cv::Point2f> &points, std::vector<Eigen::Vector3f> &points3d)
{
  if (intrinsic.empty())
    throw std::runtime_error("[TSFPoseTrackerKLT::getPoints3D] Camera parameter not set!");
  double *C = &intrinsic(0,0);
  double invC0 = 1./C[0];
  double invC4 = 1./C[4];
  float d;

  points3d.resize(points.size());

  for (unsigned i=0; i<points.size(); i++)
  {
    const cv::Point2f &pt = points[i];
    if (pt.x>=0 && pt.y>=0 && pt.x<cloud.width-1 && pt.y<cloud.height-1)
    {
      d = getInterpolated(cloud, pt);
      if (d>=0)
        points3d[i] = Eigen::Vector3f(d*((pt.x-C[2])*invC0), d*((pt.y-C[5])*invC4), d);
      else points3d[i] = Eigen::Vector3f(std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN());
    }
  }
}

/**
 * @brief TSFPoseTrackerKLT::filterValidPoints
 * @param points
 * @param points3d
 */
void TSFPoseTrackerKLT::filterValidPoints3D(std::vector<cv::Point2f> &points, std::vector<Eigen::Vector3f> &points3d)
{
  if (points.size()!=points3d.size())
    return;

  int z=0;
  for (unsigned i=0; i<points.size(); i++)
  {
    const Eigen::Vector3f &pt3 = points3d[i];
    if (!isnan(pt3[0]) && !isnan(pt3[1]) && !isnan(pt3[2]))
    {
        points[z] = points[i];
        points3d[z] = points3d[i];
        z++;
    }
  }
  points.resize(z);
  points3d.resize(z);
}

/**
 * @brief TSFPoseTrackerKLT::filterValidPoints
 * @param points
 * @param points3d
 */
void TSFPoseTrackerKLT::filterValidPoints3D(std::vector<cv::Point2f> &pts1, std::vector<Eigen::Vector3f> &pts3d1, std::vector<cv::Point2f> &pts2, std::vector<Eigen::Vector3f> &pts3d2)
{
  if (pts1.size()!=pts3d1.size() || pts1.size()!=pts3d2.size() ||  pts1.size()!=pts2.size())
    return;

  int z=0;
  for (unsigned i=0; i<pts1.size(); i++)
  {
    const Eigen::Vector3f &pt3 = pts3d1[i];
    const Eigen::Vector3f &pt32 = pts3d2[i];
    if (!isnan(pt3[0]) && !isnan(pt3[1]) && !isnan(pt3[2]) && !isnan(pt32[0]) && !isnan(pt32[1]) && !isnan(pt32[2]))
    {
      pts1[z] = pts1[i];
      pts2[z] = pts2[i];
      pts3d1[z] = pts3d1[i];
      pts3d2[z] = pts3d2[i];
      z++;
    }
  }
  pts1.resize(z);
  pts2.resize(z);
  pts3d1.resize(z);
  pts3d2.resize(z);
}

/**
 * @brief TSFPoseTrackerKLT::filterValidPoints
 * @param points
 * @param points3d
 */
void TSFPoseTrackerKLT::filterInliers(std::vector<cv::Point2f> &pts1, std::vector<Eigen::Vector3f> &pts3d1, std::vector<cv::Point2f> &pts2, std::vector<Eigen::Vector3f> &pts3d2, std::vector<int> &inliers)
{
  if (pts1.size()!=pts3d1.size() || pts1.size()!=pts3d2.size() ||  pts1.size()!=pts2.size())
    return;

  std::vector<cv::Point2f> tmp_pts1;
  std::vector<Eigen::Vector3f> tmp_pts3d1;
  std::vector<cv::Point2f> tmp_pts2;
  std::vector<Eigen::Vector3f> tmp_pts3d2;

  tmp_pts1.reserve(inliers.size());
  tmp_pts2.reserve(inliers.size());
  tmp_pts3d1.reserve(inliers.size());
  tmp_pts3d2.reserve(inliers.size());

  for (unsigned i=0; i<inliers.size(); i++)
  {
    tmp_pts1.push_back(pts1[inliers[i]]);
    tmp_pts2.push_back(pts2[inliers[i]]);
    tmp_pts3d1.push_back(pts3d1[inliers[i]]);
    tmp_pts3d2.push_back(pts3d2[inliers[i]]);
  }

  pts1 = tmp_pts1;
  pts2 = tmp_pts2;
  pts3d1 = tmp_pts3d1;
  pts3d2 = tmp_pts3d2;
}

/**
 * @brief TSFPoseTrackerKLT::filterConverged
 * @param pts1
 * @param pts3d1
 * @param pts2
 * @param pts3d2
 * @param converged
 */
void TSFPoseTrackerKLT::filterConverged(std::vector<cv::Point2f> &pts1, std::vector<Eigen::Vector3f> &pts3d1, std::vector<cv::Point2f> &pts2, std::vector<Eigen::Vector3f> &pts3d2, std::vector<int> &converged)
{
  if (pts1.size()!=pts3d1.size() || pts1.size()!=pts3d2.size() ||  pts1.size()!=pts2.size() || pts1.size()!=converged.size())
    return;

  unsigned z=0;

  for (unsigned i=0; i<pts1.size(); i++)
  {
    if (converged[i]==1)
    {
      pts1[z] = pts1[i];
      pts2[z] = pts2[i];
      pts3d1[z] = pts3d1[i];
      pts3d2[z] = pts3d2[i];
      z++;
    }
  }

  pts1.resize(z);
  pts2.resize(z);
  pts3d1.resize(z);
  pts3d2.resize(z);
}

/**
 * @brief TSFPoseTrackerKLT::needReinit
 * @param points
 * @return
 */
bool TSFPoseTrackerKLT::needReinit(const std::vector<cv::Point2f> &points)
{
  if (points.size() < data->init_points*param.pcent_reinit)
    return true;

  int hw = data->cloud.width/2;
  int hh = data->cloud.height/2;

  int cnt[4] = {0,0,0,0};

  for (unsigned i=0; i<points.size(); i++)
  {
    const cv::Point2f &pt = points[i];
    if (pt.x<hw)
    {
      if (pt.y<hh) cnt[0]++;
      else cnt[1]++;
    }
    else
    {
      if (pt.y<hh) cnt[3]++;
      else cnt[2]++;
    }
  }

  if (cnt[0]==0 || cnt[1]==0 || cnt[2]==0 || cnt[3]==0)
    return true;

  return false;
}




/**
 * @brief TSFPoseTrackerKLT::trackCamera
 * @return
 */
bool TSFPoseTrackerKLT::trackCamera(double &conf_ransac_iter, double &conf_tracked_points)
{
  bool have_pose = false;
  conf_ransac_iter = conf_tracked_points = 0;

  cv::calcOpticalFlowPyrLK(data->prev_gray, data->gray, data->points[0], data->points[1], status, err, param.win_size, 3, param.termcrit, data->lk_flags, 0.001);
  data->lk_flags = cv::OPTFLOW_USE_INITIAL_FLOW;

  // update lk points
  size_t i, k;
  for( i = k = 0; i < data->points[1].size() && i<data->points[0].size(); i++ )
  {
    if( status[i] )
    {
      data->points[0][k] = data->points[0][i];
      data->points[1][k] = data->points[1][i];
      data->points3d[0][k] = data->points3d[0][i];
      k++;
    }
  }
  data->points[0].resize(k);
  data->points[1].resize(k);
  data->points3d[0].resize(k);


  // track pose
  getPoints3D(data->cloud, data->points[1], data->points3d[1]);
  filterValidPoints3D(data->points[0],data->points3d[0], data->points[1], data->points3d[1]);

  if (data->points3d[1].size()>4)
  {
    Eigen::Matrix4f pose;
    depth.assign(data->points3d[1].size(), std::numeric_limits<float>::quiet_NaN());
    for (unsigned i=0; i<data->points3d[1].size(); i++)
      depth[i] = data->points3d[1][i][2];

    int nb_iter = pnp->ransac(data->points3d[0], data->points[1], pose, inliers, depth);

    filterInliers(data->points[0],data->points3d[0], data->points[1], data->points3d[1], inliers);

    conf_ransac_iter = 1. - ((double)nb_iter)/(double)param.rt.max_rand_trials;
    conf_tracked_points = (inliers.size()>=param.conf_tracked_points_norm ? 1. : ((double)inliers.size()) / (double)param.conf_tracked_points_norm);

    //cout<<"nb_iter="<<nb_iter;
    if (nb_iter < (int)param.rt.max_rand_trials)
    {
      //cout<<" ok!!!!"<<endl;
      // refine projective
      if (data->points[0].size()>4)
      {
        data->pose = pose*data->kf_pose;
        have_pose = true;

        // --- debug out ---
        if (!dbg.empty())
        {
          for (unsigned i=0; i<data->points[0].size(); i++)
          {
            cv::line(dbg, data->points[0][i], data->points[1][i], CV_RGB(255,255,255));
            cv::circle(dbg, data->points[1][i],2, CV_RGB(0,0,255));
          }
        }
        // -----------------
      }
    } //else cout<<" lost<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"<<endl;
  }

  return have_pose;
}





/***************************************************************************************/

/**
 * start
 */
void TSFPoseTrackerKLT::start()
{
  if (data==NULL)
    throw std::runtime_error("[TSFPoseTrackerKLT::start] No data available! Did you call 'setData'?");

  if (have_thread) stop();

  run = true;
  th_obectmanagement = boost::thread(&TSFPoseTrackerKLT::operate, this);
  have_thread = true;
}

/**
 * stop
 */
void TSFPoseTrackerKLT::stop()
{
  run = false;
  th_obectmanagement.join();
  have_thread = false;
}




/**
 * reset
 */
void TSFPoseTrackerKLT::reset()
{
  stop();
}


/**
 * @brief TSFPoseTrackerKLT::filter
 * @param cloud
 * @param filtered_cloud
 */
void TSFPoseTrackerKLT::track(double &conf_ransac_iter, double &conf_tracked_points)
{
  if (data==NULL)
    throw std::runtime_error("[TSFPoseTrackerKLT::track] No data available! Did you call 'setData'?");

  if (!isStarted()) start();

  if(data->prev_gray.empty())
  {
    data->need_init=true;

  }
  else if (data->points[0].size()>4)
  {
    data->have_pose = trackCamera(conf_ransac_iter, conf_tracked_points);

    // test stability
    if (needReinit(data->points[1]))
      data->need_init = true;
  }
  else data->need_init = true;
}

/**
 * setCameraParameter
 */
void TSFPoseTrackerKLT::setCameraParameter(const cv::Mat &_intrinsic)
{
  if (_intrinsic.type() != CV_64F)
    _intrinsic.convertTo(intrinsic, CV_64F);
  else intrinsic = _intrinsic;

  pnp->setCameraParameter(_intrinsic, cv::Mat());

  reset();
}

/**
 * @brief TSFPoseTrackerKLT::setParameter
 * @param p
 */
void TSFPoseTrackerKLT::setParameter(const Parameter &p)
{
  param = p;
}




}












