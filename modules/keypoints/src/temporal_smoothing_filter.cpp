/**
 * $Id$
 * 
 * Copyright (C) 2016: Johann Prankl, prankl@acin.tuwien.ac.at
 * @author Johann Prankl
 *
 */

#include <v4r/keypoints/temporal_smoothing_filter.h>
#include <v4r/keypoints/impl/invPose.hpp>
#include <v4r/common/convertImage.h>
#include <pcl/common/transforms.h>



namespace v4r
{


using namespace std;




/************************************************************************************
 * Constructor/Destructor
 */
TemporalSmoothingFilter::TemporalSmoothingFilter(const Parameter &p)
 : param(p), run(false), have_thread(false), sf_pose(Eigen::Matrix4f::Identity())
{ 
  rt.reset(new v4r::RigidTransformationRANSAC(param.rt));
  sf_cloud.reset( new v4r::DataMatrix2D<Surfel>() );
  tmp_cloud.reset( new v4r::DataMatrix2D<Surfel>() );
  global_cloud.reset( new pcl::PointCloud<pcl::PointXYZRGBNormal>() );
  npat.resize(4);
  npat[0] = cv::Vec4i(1,0,0,1);
  npat[1] = cv::Vec4i(0,1,-1,0);
  npat[2] = cv::Vec4i(-1,0,0,-1);
  npat[3] = cv::Vec4i(0,-1,0,1);
}

TemporalSmoothingFilter::~TemporalSmoothingFilter()
{
  if (have_thread) stop();
}

/**
 * operate
 */
void TemporalSmoothingFilter::operate()
{
  bool have_todo;

  cv::Mat im;
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  std::vector<cv::Point2f> points;
  std::vector<Eigen::Vector3f> points3d;
  Eigen::Matrix4f pose;

  while(run)
  {
    have_todo = false;

    shm.lock();
    if (shm.need_init)
    {
      have_todo = true;
      shm.gray.copyTo(im);
      cloud = shm.cloud;
      pose = shm.pose;
    }
    shm.unlock();

    if (have_todo)
    {
      cv::goodFeaturesToTrack(im, points, param.max_count, 0.01, 10, cv::Mat(), 3, 0, 0.04);
      cv::cornerSubPix(im, points, param.subpix_win_size, cv::Size(-1,-1), param.termcrit);
      getPoints3D(cloud, points, points3d);
      filterValidPoints3D(points, points3d);

      shm.lock();
      shm.init_points = points.size();
      shm.lk_flags = 0;
      im.copyTo(shm.prev_gray);
      shm.points[0] = points;
      shm.points3d[0] = points3d;
      shm.kf_pose = pose;
      if (shm.points[0].size() > param.max_count*param.pcent_reinit)
        shm.need_init = false;
      shm.unlock();
    }

    if (!have_todo) usleep(10000);
  }
}

/**
 * @brief TemporalSmoothingFilter::getPoints3D
 * @param cloud
 * @param points
 * @param points3d
 */
void TemporalSmoothingFilter::getPoints3D(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const std::vector<cv::Point2f> &points, std::vector<Eigen::Vector3f> &points3d)
{
  if (!cloud.isOrganized())
    throw std::runtime_error("[TemporalSmoothingFilter::getPoints3D] Need an organized RGBD cloud!");

  points3d.resize(points.size());

  for (unsigned i=0; i<points.size(); i++)
  {
    const cv::Point2f &pt = points[i];
    if (pt.x>=0 && pt.y>=0 && pt.x<cloud.width-1 && pt.y<cloud.height-1)
      points3d[i] = cloud((int)(pt.x+.5),(int)(pt.y+.5)).getVector3fMap();
  }
}

/**
 * @brief TemporalSmoothingFilter::filterValidPoints
 * @param points
 * @param points3d
 */
void TemporalSmoothingFilter::filterValidPoints3D(std::vector<cv::Point2f> &points, std::vector<Eigen::Vector3f> &points3d)
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
 * @brief TemporalSmoothingFilter::filterValidPoints
 * @param points
 * @param points3d
 */
void TemporalSmoothingFilter::filterValidPoints3D(std::vector<cv::Point2f> &pts1, std::vector<Eigen::Vector3f> &pts3d1, std::vector<cv::Point2f> &pts2, std::vector<Eigen::Vector3f> &pts3d2)
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
 * @brief TemporalSmoothingFilter::filterValidPoints
 * @param points
 * @param points3d
 */
void TemporalSmoothingFilter::filterInliers(std::vector<cv::Point2f> &pts1, std::vector<Eigen::Vector3f> &pts3d1, std::vector<cv::Point2f> &pts2, std::vector<Eigen::Vector3f> &pts3d2, std::vector<int> &inliers_)
{
  if (pts1.size()!=pts3d1.size() || pts1.size()!=pts3d2.size() ||  pts1.size()!=pts2.size())
    return;

  std::vector<cv::Point2f> tmp_pts1;
  std::vector<Eigen::Vector3f> tmp_pts3d1;
  std::vector<cv::Point2f> tmp_pts2;
  std::vector<Eigen::Vector3f> tmp_pts3d2;

  tmp_pts1.reserve(inliers_.size());
  tmp_pts2.reserve(inliers_.size());
  tmp_pts3d1.reserve(inliers_.size());
  tmp_pts3d2.reserve(inliers_.size());

  for (unsigned i=0; i<inliers_.size(); i++)
  {
    tmp_pts1.push_back(pts1[inliers_[i]]);
    tmp_pts2.push_back(pts2[inliers_[i]]);
    tmp_pts3d1.push_back(pts3d1[inliers_[i]]);
    tmp_pts3d2.push_back(pts3d2[inliers_[i]]);
  }

  pts1 = tmp_pts1;
  pts2 = tmp_pts2;
  pts3d1 = tmp_pts3d1;
  pts3d2 = tmp_pts3d2;
}

/**
 * @brief TemporalSmoothingFilter::needReinit
 * @param points
 * @return
 */
bool TemporalSmoothingFilter::needReinit(const std::vector<cv::Point2f> &points)
{
  if (points.size() < shm.init_points*param.pcent_reinit)
    return true;

  int hw = im_width/2;
  int hh = im_height/2;

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
 * @brief TemporalSmoothingFilter::addCloud
 * @param cloud
 * @param pose
 * @param sf_cloud
 * @param sf_pose
 */
//void TemporalSmoothingFilter::addCloud(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const Eigen::Matrix4f &pose, v4r::DataMatrix2D<Surfel> &sf_cloud, Eigen::Matrix4f &sf_pose)
//{
//  if (intrinsic.empty())
//    throw std::runtime_error("[TemporalSmoothingFilter::addCloud] Camera parameter not set!");

//  sf_width = cloud.width;
//  sf_height = cloud.height;

//  cv::Point2f im_pt;
//  double *C = &intrinsic(0,0);
//  Eigen::Matrix4f inv_sf, inc_pose;
//  v4r::invPose(sf_pose, inv_sf);
//  inc_pose = pose*inv_sf;

//  // init
//  if (sf_cloud.rows!=sf_height || sf_cloud.cols!=sf_width)
//  {
//    sf_cloud.resize(sf_height, sf_width);

//    for (unsigned v=0; v<cloud.height; v++)
//    {
//      for (unsigned u=0; u<cloud.width; u++)
//      {
//        sf_cloud(v,u) = Surfel(cloud(u,v));
//      }
//    }
//  }
//  else
//  {
//    //transform to current frame
//    tmp_cloud = sf_cloud;
//    Eigen::Vector3f pt,n;
//    Eigen::Matrix3f R = inc_pose.topLeftCorner<3,3>();
//    Eigen::Vector3f t = inc_pose.block<3,1>(0,3);
//    Eigen::Vector3f vr;
//    int x, y;

//    // reset sf_cloud
//    for (unsigned i=0; i<sf_cloud.data.size(); i++)
//      sf_cloud.data[i].cnt = 0;

//    // tranform tmp_cloud to current sf_cloud
//    for (int v=0; v<tmp_cloud.rows; v++)
//    {
//      for (int u=0; u<tmp_cloud.cols; u++)
//      {
//        const Surfel &s = tmp_cloud(v,u);

//        if (std::isnan(s.pt[0])||std::isnan(s.pt[1])||std::isnan(s.pt[2]))
//          continue;

//        pt = R*s.pt+t;
//        n = R*s.n;
//        v4r::projectPointToImage(&pt[0],C,&im_pt.x);
//        x = (int)(im_pt.x+.5);
//        y = (int)(im_pt.y+.5);

//        if (x>=0 && x<sf_width && y>=0 && y<sf_height)
//        {
//          Surfel &sf = sf_cloud(y,x);
//          vr = Eigen::Vector3f( (x-C[2])/C[0], (y-C[5])/C[4], 1. ).normalized();
//          intersectPlaneLine(pt,n,vr,sf.pt);
//          sf.n = n;
//          sf.cnt = s.cnt;
//          sf.r = s.r;
//          sf.g = s.g;
//          sf.b = s.b;
//        }
//      }
//    }

//    // integrate new data
//    for (unsigned v=0; v<cloud.height; v++)
//    {
//      for (unsigned u=0; u<cloud.width; u++)
//      {
//        Surfel &sf = sf_cloud(v,u);

//        if (sf.cnt==0)
//        {
//          sf = Surfel(cloud(u,v));
//        }
//        else
//        {
//          const pcl::PointXYZRGB &pt = cloud(u,v);
//          sf.pt = (sf.pt*(float)sf.cnt + pt.getVector3fMap()) / (float)(sf.cnt+1);
//          sf.cnt++;
//        }
//      }
//    }
//  }

//  sf_pose = pose;
//}

void TemporalSmoothingFilter::addCloud(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const Eigen::Matrix4f &pose, v4r::DataMatrix2D<Surfel>::Ptr &sf_cloud_, Eigen::Matrix4f &sf_pose_)
{
  if (intrinsic.empty())
    throw std::runtime_error("[TemporalSmoothingFilter::addCloud] Camera parameter not set!");

  sf_width = cloud.width;
  sf_height = cloud.height;

  cv::Point2f im_pt;
  double *C = &intrinsic(0,0);
  double invC0 = 1./C[0];
  double invC4 = 1./C[4];
  Eigen::Matrix4f inv_sf, inc_pose;
  v4r::invPose(sf_pose_, inv_sf);
  inc_pose = pose*inv_sf;

  // init
  if (sf_cloud_->rows!=sf_height || sf_cloud_->cols!=sf_width)
  {
    v4r::DataMatrix2D<Surfel> &ref = *sf_cloud_;
    ref.resize(sf_height, sf_width);

    for (unsigned v=0; v<cloud.height; v++)
    {
      for (unsigned u=0; u<cloud.width; u++)
      {
        ref(v,u) = Surfel(cloud(u,v));
      }
    }
  }
  else
  {
    //transform to current frame
    std::swap(sf_cloud_, tmp_cloud);
    v4r::DataMatrix2D<Surfel> &ref_sf = *sf_cloud_;
    v4r::DataMatrix2D<Surfel> &ref_tmp = *tmp_cloud;
    Eigen::Vector3f pt;//,n;
    Eigen::Matrix3f R = inc_pose.topLeftCorner<3,3>();
    Eigen::Vector3f t = inc_pose.block<3,1>(0,3);
    int x, y;
    float ax, ay, inv_z, thr_depth_cutoff, norm;

    // reset sf_cloud
    ref_sf.resize(sf_height, sf_width);
    for (unsigned i=0; i<ref_sf.data.size(); i++)
    {
      ref_sf.data[i].weight = 0.;
      ref_sf.data[i].norm = 0.;
      ref_sf.data[i].pt[2] = 0.;
    }

    // tranform ref_tmp to current ref_sf
    for (int v=0; v<ref_tmp.rows; v++)
    {
      for (int u=0; u<ref_tmp.cols; u++)
      {
        const Surfel &s = ref_tmp(v,u);

        if (std::isnan(s.pt[0])||std::isnan(s.pt[1])||std::isnan(s.pt[2]))
          continue;

        pt = R*s.pt+t;
        inv_z = 1./pt[2];
        im_pt.x = C[0]*pt[0]*inv_z + C[2];
        im_pt.y = C[4]*pt[1]*inv_z + C[5];
        x = (int)(im_pt.x);
        y = (int)(im_pt.y);
        ax = im_pt.x-x;
        ay = im_pt.y-y;
        
        thr_depth_cutoff = param.scale_depth_cutoff*pt[2];

        if (x>=0 && x<sf_width && y>=0 && y<sf_height && fabs(cloud(x,y).z-pt[2]) < thr_depth_cutoff )
        {
          Surfel &sf = ref_sf(y,x);
          norm = (1.-ax) * (1.-ay);
          sf.pt[2] += norm*pt[2];
          sf.weight += norm*s.weight;
          sf.norm += norm;
          if (s.weight < param.max_integration_frames) sf.is_stable = false;
          else sf.is_stable = true;
        }
        if (x+1>=0 && x+1<sf_width && y>=0 && y<sf_height && fabs(cloud(x+1,y).z-pt[2]) < thr_depth_cutoff)
        {
          Surfel &sf = ref_sf(y,x+1);
          norm = ax * (1.-ay);
          sf.pt[2] += norm*pt[2];
          sf.weight += norm*s.weight;
          sf.norm += norm;
        }
        if (x>=0 && x<sf_width && y+1>=0 && y+1<sf_height && fabs(cloud(x,y+1).z-pt[2]) < thr_depth_cutoff)
        {
          Surfel &sf = ref_sf(y+1,x);
          norm = (1.-ax) *  ay;
          sf.pt[2] += norm*pt[2];
          sf.weight += norm*s.weight;
          sf.norm += norm;
        }
        if (x+1>=0 && x+1<sf_width && y+1>=0 && y+1<sf_height && fabs(cloud(x+1,y+1).z-pt[2]) < thr_depth_cutoff)
        {
          Surfel &sf = ref_sf(y+1,x+1);
          norm = ax * ay;
          sf.pt[2] += norm*pt[2];
          sf.weight += norm*s.weight;
          sf.norm += norm;
        }
      }
    }

    // integrate new data
    float inv_norm;

    for (unsigned v=0; v<cloud.height; v++)
    {
      for (unsigned u=0; u<cloud.width; u++)
      {
        Surfel &sf = ref_sf(v,u);

        if (fabs(sf.weight)<=std::numeric_limits<float>::epsilon())
        {
          sf = Surfel(cloud(u,v));
        }
        else
        {
          inv_norm = 1./sf.norm;
          sf.pt[2] *= inv_norm;
          sf.weight *= inv_norm;
          const pcl::PointXYZRGB &pt_ = cloud(u,v);
          sf.pt[2] = (sf.pt[2]*(float)sf.weight + pt_.z) / (float)(sf.weight+1.);
          sf.pt[0] = sf.pt[2]*((u-C[2])*invC0);
          sf.pt[1] = sf.pt[2]*((v-C[5])*invC4);
          sf.r = pt_.r;
          sf.g = pt_.g;
          sf.b = pt_.b;
          if (sf.weight<param.max_integration_frames) sf.weight+=1;
        }
      }
    }
  }

  sf_pose_ = pose;
}



/**
 * @brief TemporalSmoothingFilter::convert
 * @param cfilt
 * @param out
 */
void TemporalSmoothingFilter::convertSurfelMap(const v4r::DataMatrix2D<Surfel> &cfilt, pcl::PointCloud<pcl::PointXYZRGBNormal> &out)
{
  out.resize(cfilt.data.size());
  out.width = cfilt.cols;
  out.height = cfilt.rows;
  out.is_dense = false;

  for (unsigned i=0; i<cfilt.data.size(); i++)
  {
    const Surfel &s = cfilt.data[i];
    pcl::PointXYZRGBNormal &o = out.points[i];
    o.getVector3fMap() = s.pt;
    o.getNormalVector3fMap() = s.n;
    o.r = s.r;
    o.g = s.g;
    o.b = s.b;
  }
}

/**
 * @brief TemporalSmoothingFilter::logGlobalMap
 * @param cfilt
 * @param pose
 * @param out
 */
void TemporalSmoothingFilter::logGlobalMap(const v4r::DataMatrix2D<Surfel> &cfilt, const Eigen::Matrix4f &pose, pcl::PointCloud<pcl::PointXYZRGBNormal> &out)
{
  Eigen::Matrix4f inv_pose;
  v4r::invPose(pose, inv_pose);
  Eigen::Matrix3f R = inv_pose.topLeftCorner<3,3>();
  Eigen::Vector3f t = inv_pose.block<3,1>(0,3);

  for (unsigned i=0; i<cfilt.data.size(); i++)
  {
    const Surfel &s = cfilt.data[i];

    if (!s.is_stable && s.weight>=param.max_integration_frames)
    {
      out.points.push_back( pcl::PointXYZRGBNormal() );
      pcl::PointXYZRGBNormal &o = out.points.back();
      o.getVector3fMap() = R*s.pt+t;
      o.getNormalVector3fMap() = R*s.n;
      o.r = s.r;
      o.g = s.g;
      o.b = s.b;
    }
  }

  if (out.points.size()>param.global_map_size)
  {
    int z=0;
    int _start = out.points.size()-param.global_map_size;
    for (int i=_start; i<(int)out.points.size(); i++,z++)
      out.points[z] = out.points[i];
    out.points.resize(z);
  }

  out.width = out.points.size();
  out.height = 1;
  out.is_dense = true;
}

/**
 * @brief TemporalSmoothingFilter::computeNormal
 * @param sf_cloud
 */
void TemporalSmoothingFilter::computeNormal(v4r::DataMatrix2D<Surfel> &sf_cloud_)
{
  Surfel *s1, *s2, *s3;
  Eigen::Vector3f l1, l2;
  int z;

  for (int v=0; v<sf_cloud_.rows; v++)
  {
    for (int u=0; u<sf_cloud_.cols; u++)
    {
      s2=s3=0;
      s1 = &sf_cloud_(v,u);
      if (std::isnan(s1->pt[0]) || std::isnan(s1->pt[1]) || std::isnan(s1->pt[2]))
        continue;

      for (z=0; z<4; z++)
      {
        const cv::Vec4i &p = npat[z];
        if (u+p[0]>=0 && u+p[0]<sf_cloud_.cols && v+p[1]>=0 && v+p[1]<sf_cloud_.rows &&
            u+p[2]>=0 && u+p[2]<sf_cloud_.cols && v+p[3]>=0 && v+p[3]<sf_cloud_.rows)
        {
          s2 = &sf_cloud_(v+p[1],u+p[0]);
          if (std::isnan(s2->pt[0]) || std::isnan(s2->pt[1]) || std::isnan(s2->pt[2]))
            continue;
          s3 = &sf_cloud_(v+p[3],u+p[2]);
          if (std::isnan(s3->pt[0]) || std::isnan(s3->pt[1]) || std::isnan(s3->pt[2]))
            continue;
          break;
        }
      }

      if (z<4)
      {
        l1 = s2->pt-s1->pt;
        l2 = s3->pt-s1->pt;
        s1->n = l1.cross(l2).normalized();
        if (s1->n.dot(s1->pt) > 0) s1->n *= -1;
      }
    }
  }
}



/***************************************************************************************/

/**
 * start
 */
void TemporalSmoothingFilter::start()
{
  if (have_thread) stop();

  run = true;
  th_obectmanagement = boost::thread(&TemporalSmoothingFilter::operate, this);  
  have_thread = true;
}

/**
 * stop
 */
void TemporalSmoothingFilter::stop()
{
  run = false;
  th_obectmanagement.join();
  have_thread = false;
}




/**
 * reset
 */
void TemporalSmoothingFilter::reset()
{
  stop();
  sf_cloud->clear();
  global_cloud.reset( new pcl::PointCloud<pcl::PointXYZRGBNormal>() );
  shm.reset();
}


/**
 * @brief TemporalSmoothingFilter::filter
 * @param cloud
 * @param filtered_cloud
 */
void TemporalSmoothingFilter::filter(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, pcl::PointCloud<pcl::PointXYZRGBNormal> &filtered_cloud, Eigen::Matrix4f &pose)
{
  //v4r::ScopeTime t("[TemporalSmoothingFilter::filter]");

  if (!isStarted()) start();

  convertImage(cloud, image);

//// -- dbg
//cv::Mat drw;
//image.copyTo(drw);
//// --

  im_width = image.cols;
  im_height = image.rows;

  shm.lock();

  cv::cvtColor( image, shm.gray, CV_RGB2GRAY );
  shm.cloud = cloud;

  if(shm.prev_gray.empty())
  {
    shm.need_init=true;
  }
  else if (shm.points[0].size()>0)
  {
    cv::calcOpticalFlowPyrLK(shm.prev_gray, shm.gray, shm.points[0], shm.points[1], status, err, param.win_size, 3, param.termcrit,shm.lk_flags, 0.001);
    shm.lk_flags = cv::OPTFLOW_USE_INITIAL_FLOW;

    // update lk points
    size_t i, k;
    for( i = k = 0; i < shm.points[1].size() && i<shm.points[0].size(); i++ )
    {
      if( status[i] )
      {
        shm.points[0][k] = shm.points[0][i];
        shm.points[1][k] = shm.points[1][i];
        shm.points3d[0][k] = shm.points3d[0][i];
        k++;
      }
    }
    shm.points[0].resize(k);
    shm.points[1].resize(k);
    shm.points3d[0].resize(k);

    // track pose
    getPoints3D(cloud,shm.points[1], shm.points3d[1]);

    filterValidPoints3D(shm.points[0],shm.points3d[0], shm.points[1], shm.points3d[1]);

    if (shm.points3d[1].size()>=5)
    {
      Eigen::Matrix4f pose_;
      int nb_iter = rt->compute(shm.points3d[0], shm.points3d[1], pose_, inliers);
      //cout<<"Ransac iter: "<<nb_iter<<"("<<inliers.size()<<"/"<<shm.points3d[0].size()<<")"<<endl;
      filterInliers(shm.points[0],shm.points3d[0], shm.points[1], shm.points3d[1], inliers);

      if (nb_iter < (int)param.rt.max_rand_trials)
        shm.pose = pose_*shm.kf_pose;
    }

    // test stability
    if (needReinit(shm.points[1]))
      shm.need_init = true;

//    // -- dbg
//    for (unsigned i=0; i<shm.points[0].size()&&i<shm.points[1].size(); i++)
//    {
//      cv::circle( drw, shm.points[1][i], 3, cv::Scalar(0,255,0), -1, 8);
//      cv::line(drw, shm.points[0][i], shm.points[1][i], cv::Scalar(255,255,255), 1, 8, 0);
//    }
//    cv::imshow("dbg-filt",drw);
//    cv::waitKey(1);
//    cout<<"[TemporalSmoothingFilter::filter] "<<shm.points[1].size()<<"/"<<shm.init_points<<endl;
//    // --

  }
  else shm.need_init = true;

  pose = shm.pose;

  shm.unlock();

  // filter point cloud
  addCloud(cloud, shm.pose, sf_cloud, sf_pose);
  if (param.compute_normals) computeNormal(*sf_cloud);
  convertSurfelMap(*sf_cloud, filtered_cloud);
  if (param.global_map_size > 0) logGlobalMap(*sf_cloud, pose, *global_cloud);
//  cout<<"global_cloud->points.size()="<<global_cloud->points.size()<<endl;
}

/**
 * @brief TemporalSmoothingFilter::getGlobalCloud
 * @param _global_cloud
 */
void TemporalSmoothingFilter::getGlobalCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &_global_cloud)
{
  _global_cloud = global_cloud;
}

/**
 * setCameraParameter
 */
void TemporalSmoothingFilter::setCameraParameter(const cv::Mat &_intrinsic)
{
  if (_intrinsic.type() != CV_64F)
    _intrinsic.convertTo(intrinsic, CV_64F);
  else intrinsic = _intrinsic;

  reset();
}


}












