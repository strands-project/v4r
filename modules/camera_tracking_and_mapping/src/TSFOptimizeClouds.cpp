/**
 * $Id$
 * 
 * @author Johann Prankl
 *
 */


#include <v4r/camera_tracking_and_mapping/TSFOptimizeClouds.h>
#include <v4r/keypoints/impl/invPose.hpp>
#include <v4r/common/convertImage.h>
#include <pcl/common/transforms.h>
#include <v4r/reconstruction/impl/projectPointToImage.hpp>
#include <pcl/common/time.h>



namespace v4r
{


using namespace std;


std::vector<cv::Vec4i> TSFOptimizeClouds::npat = std::vector<cv::Vec4i>();



/************************************************************************************
 * Constructor/Destructor
 */
TSFOptimizeClouds::TSFOptimizeClouds(const Parameter &p)
 : width(0), height(0), sf_timestamp(0), sf_pose(Eigen::Matrix4f::Identity()), run(false), have_thread(false)
{ 
  setParameter(p);
}

TSFOptimizeClouds::~TSFOptimizeClouds()
{
  if (have_thread) stop();
}

/**
 * operate
 */
void TSFOptimizeClouds::operate()
{
  bool have_todo;

  Eigen::Matrix4f inv_pose;
  v4r::DataMatrix2D<Surfel> sf_cloud_local;
  std::list< v4r::triple< pcl::PointCloud<pcl::PointXYZRGB>::Ptr, Eigen::Matrix4f, double > > frames_local;
  std::list< v4r::triple< pcl::PointCloud<pcl::PointXYZRGB>::Ptr, Eigen::Matrix4f, double > >::iterator it0, it;

  while(run)
  {
    have_todo = false;
    mtx_shm.lock();
    if (frames.size()==(unsigned)param.batch_size_clouds)
    {
      frames_local = frames;
      frames.clear();
      have_todo = true;
    }
    mtx_shm.unlock();

    if (have_todo)
    {
      if (intrinsic.empty() || tgt_intrinsic.empty())
        throw std::runtime_error("[TSFOptimizeClouds::addCloud] Camera parameter not available");

      it0 = std::next(frames_local.begin(), frames_local.size()/2);
      initKeyframe(*it0->first);

      for (it=frames_local.begin(); it!=frames_local.end(); it++)
      {
        invPose(it->second, inv_pose);
        //integrateData(*it->first, it0->second*inv_pose);
        integrateDataRGB(*it->first, it0->second*inv_pose);
      }

      project3D(sf_cloud_local, 0.5);
      computeNormals(sf_cloud_local,2);

      mtx_shm.lock();
      sf_timestamp = it0->third;
      sf_pose = it0->second;
      sf_cloud = sf_cloud_local;
      mtx_shm.unlock();
      have_todo = false;
    }

    if (!have_todo) usleep(10000);
  }
}

/**
 * @brief TSFOptimizeClouds::initKeyframe
 */
void TSFOptimizeClouds::initKeyframe(const pcl::PointCloud<pcl::PointXYZRGB> &cloud0)
{
  depth = cv::Mat_<float>::zeros(height, width);
  depth_weight = cv::Mat_<float>::zeros(height, width);
  im_bgr = cv::Mat_<cv::Vec3f>::zeros(height, width);

  if (cloud0.isOrganized())
  {
    float scale_x = ((float)width)/(float)cloud0.width;
    float scale_y = ((float)height)/(float)cloud0.height;

    for (unsigned v=0; v<cloud0.height; v++)
    {
      for (unsigned u=0; u<cloud0.width; u++)
      {
        cv::Vec3f &c = im_bgr(scale_y*v, scale_x*u);
        const pcl::PointXYZRGB &pt = cloud0(u,v);
        c[0] = pt.b;
        c[1] = pt.g;
        c[2] = pt.r;
      }
    }
  }
}


/**
 * @brief TSFOptimizeClouds::integrateData
 * @param cloud in camera coordinates
 * @param pose to transform the cloud from global coordinates to camera coordinates
 */
void TSFOptimizeClouds::integrateData(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const Eigen::Matrix4f &pose)
{
  int x,y;
  cv::Point2f im_pt;

  pcl::transformPointCloud(cloud,tmp_cloud,pose);

  // tranform filt cloud to current frame and update
  for (unsigned i=0; i<tmp_cloud.points.size(); i++)
  {
    const pcl::PointXYZRGB &pt3 = tmp_cloud.points[i];

    if (isnan(pt3.x) || isnan(pt3.y) || isnan(pt3.z))
      continue;

    projectPointToImage(&pt3.x, &tgt_intrinsic(0,0), &im_pt.x);

    x = (int)(im_pt.x);
    y = (int)(im_pt.y);

    if (x<0 || y<0 || x>=width || y>=height)
      continue;

    float &d = depth(y,x);
    float &w = depth_weight(y,x);

    if (w>0)
    {
      if (fabs(1./d - 1./pt3.z) < param.inv_depth_cut_off)
      {
        d = (w*d + pt3.z);
        w += 1.;
        d /= w;
      }
      else
      {
        w -= 1.;
      }
    }
    else
    {
      d = pt3.z;
      w=1.;
    }
  }
}

/**
 * @brief TSFOptimizeClouds::integrateDataRGB
 * @param cloud
 * @param pose
 */
void TSFOptimizeClouds::integrateDataRGB(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const Eigen::Matrix4f &pose)
{
  int x,y;
  cv::Point2f im_pt;

  pcl::transformPointCloud(cloud,tmp_cloud,pose);

  // tranform filt cloud to current frame and update
  for (unsigned i=0; i<tmp_cloud.points.size(); i++)
  {
    const pcl::PointXYZRGB &pt3 = tmp_cloud.points[i];

    if (isnan(pt3.x) || isnan(pt3.y) || isnan(pt3.z))
      continue;

    projectPointToImage(&pt3.x, &tgt_intrinsic(0,0), &im_pt.x);

    x = (int)(im_pt.x);
    y = (int)(im_pt.y);

    if (x<0 || y<0 || x>=width || y>=height)
      continue;

    float &d = depth(y,x);
    cv::Vec3f &col = im_bgr(y,x);
    float &w = depth_weight(y,x);

    if (w>0)
    {
      if (fabs(1./d - 1./pt3.z) < param.inv_depth_cut_off)
      {
        d = (w*d + pt3.z);
        col[0] = (w*col[0]+pt3.b);
        col[1] = (w*col[1]+pt3.g);
        col[2] = (w*col[2]+pt3.r);
        w += 1.;
        d /= w;
        col[0] /= w;
        col[1] /= w;
        col[2] /= w;
      }
      else
      {
        w -= 1.;
      }
    }
    else
    {
      d = pt3.z;
      col[0] = pt3.b;
      col[1] = pt3.g;
      col[2] = pt3.r;
      w=1.;
    }
  }
}


void TSFOptimizeClouds::project3D(v4r::DataMatrix2D<Surfel> &_sf_cloud, const float &px_offs)
{
  double *C = &tgt_intrinsic(0,0);
  double invC0 = 1./C[0];
  double invC4 = 1./C[4];
  _sf_cloud.resize(depth.rows, depth.cols);

  for (int v=0; v<depth.rows; v++)
  {
    for (int u=0; u<depth.cols; u++)
    {
      Surfel &sf = _sf_cloud(v,u);
      sf.weight = depth_weight(v,u);

      if (sf.weight>std::numeric_limits<float>::epsilon())
      {
        sf.pt[2] = depth(v,u);
        sf.pt[0] = sf.pt[2] * (( ((float)u)+px_offs-C[2]) * invC0);
        sf.pt[1] = sf.pt[2] * (( ((float)v)+px_offs-C[5]) * invC4);
      }
      else sf = Surfel();

      const cv::Vec3b &c = im_bgr(v,u);
      sf.r = c[2];
      sf.g = c[1];
      sf.b = c[0];
    }
  }
}



/**
 * @brief TSFOptimizeClouds::selectintegrateDataFrame
 * @param pose0
 * @param pose1
 * @return
 */
bool TSFOptimizeClouds::selectFrame(const Eigen::Matrix4f &pose0, const Eigen::Matrix4f &pose1)
{
  Eigen::Matrix4f inv_pose0, inv_pose1;
  invPose(pose0, inv_pose0);
  invPose(pose1, inv_pose1);
  if ( (inv_pose0.block<3,1>(0,2).dot(inv_pose1.block<3,1>(0,2)) > cos_delta_angle_select_frame) &&
       (inv_pose0.block<3,1>(0,3)-inv_pose1.block<3,1>(0,3)).squaredNorm() < sqr_cam_distance_select_frame )
    return false;
  return true;
}


/***************************************************************************************/

/**
 * start
 */
void TSFOptimizeClouds::start()
{
  if (intrinsic.empty())
    throw std::runtime_error("[TSFOptimizeClouds::start] No camera parameter available!");

  if (have_thread) stop();

  run = true;
  th_obectmanagement = boost::thread(&TSFOptimizeClouds::operate, this);
  have_thread = true;
}

/**
 * stop
 */
void TSFOptimizeClouds::stop()
{
  run = false;
  th_obectmanagement.join();
  have_thread = false;
}




/**
 * reset
 */
void TSFOptimizeClouds::reset()
{
  stop();

  frames.clear();
  sf_cloud.clear();
  sf_timestamp = 0;
  sf_pose.setIdentity();
}


/**
 * @brief TSFOptimizeClouds::addCloud
 * @param cloud
 * @param pose
 * @param have_track
 */
void TSFOptimizeClouds::addCloud(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const Eigen::Matrix4f &pose, const double &_timestamp, bool have_track)
{
  if (!isStarted()) start();

  mtx_shm.lock();

  if (!have_track)
    frames.clear();

  if ( selectFrame(pose, frames.back().second) )
  {
    if (frames.size()==(unsigned)param.batch_size_clouds)
      frames.pop_front();

    frames.push_back( v4r::triple< pcl::PointCloud<pcl::PointXYZRGB>::Ptr, Eigen::Matrix4f, double >( pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>()), pose, _timestamp) );
    pcl::copyPointCloud(cloud, *frames.back().first);
  }

  mtx_shm.unlock();
}


/**
 * @brief TSFOptimizeClouds::computeRadius
 * @param sf_cloud
 */
void TSFOptimizeClouds::computeRadius(v4r::DataMatrix2D<Surfel> &sf_cloud, const cv::Mat_<double> &intrinsic)
{
  const float norm = 1./sqrt(2)*(2./(intrinsic(0,0)+intrinsic(1,1)));
  for (int v=0; v<sf_cloud.rows; v++)
  {
    for (int u=0; u<sf_cloud.cols; u++)
    {
      Surfel &s  = sf_cloud(v,u);
      if (std::isnan(s.pt[0]) || std::isnan(s.pt[1]) || std::isnan(s.pt[2]))
      {
        s.radius = 0.;
        continue;
      }
      s.radius = norm*s.pt[2];
    }
  }
}

/**
 * @brief TSFOptimizeClouds::computeNormals
 * @param sf_cloud
 */
void TSFOptimizeClouds::computeNormals(v4r::DataMatrix2D<Surfel> &sf_cloud, int nb_dist)
{
  {
    npat.resize(4);
    npat[0] = cv::Vec4i(nb_dist,0,0,nb_dist);
    npat[1] = cv::Vec4i(0,nb_dist,-nb_dist,0);
    npat[2] = cv::Vec4i(-nb_dist,0,0,-nb_dist);
    npat[3] = cv::Vec4i(0,-nb_dist,0,nb_dist);
  }

  Surfel *s1, *s2, *s3;
  Eigen::Vector3f l1, l2;
  int z;

  for (int v=0; v<sf_cloud.rows; v++)
  {
    for (int u=0; u<sf_cloud.cols; u++)
    {
      s2=s3=0;
      s1 = &sf_cloud(v,u);
      if (std::isnan(s1->pt[0]) || std::isnan(s1->pt[1]) || std::isnan(s1->pt[2]))
        continue;
      for (z=0; z<4; z++)
      {
        const cv::Vec4i &p = npat[z];
        if (u+p[0]>=0 && u+p[0]<sf_cloud.cols && v+p[1]>=0 && v+p[1]<sf_cloud.rows &&
            u+p[2]>=0 && u+p[2]<sf_cloud.cols && v+p[3]>=0 && v+p[3]<sf_cloud.rows)
        {
          s2 = &sf_cloud(v+p[1],u+p[0]);
          if (std::isnan(s2->pt[0]) || std::isnan(s2->pt[1]) || std::isnan(s2->pt[2]))
            continue;
          s3 = &sf_cloud(v+p[3],u+p[2]);
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
      else s1->n = Eigen::Vector3f(std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN());
    }
  }
}


void TSFOptimizeClouds::getFilteredCloudNormals(pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud, Eigen::Matrix4f &pose, double &timestamp)
{
  mtx_shm.lock();
  cloud.resize(sf_cloud.data.size());
  cloud.width = sf_cloud.cols;
  cloud.height = sf_cloud.rows;
  cloud.is_dense = false;
  for (unsigned i=0; i<sf_cloud.data.size(); i++)
  {
    const Surfel &s = sf_cloud.data[i];
    pcl::PointXYZRGBNormal &o = cloud.points[i];
    o.getVector3fMap() = s.pt;
    o.r = s.r;
    o.g = s.g;
    o.b = s.b;
    o.getNormalVector3fMap() = s.n;
  }
  timestamp = sf_timestamp;
  pose = sf_pose;
  mtx_shm.unlock();
}

void TSFOptimizeClouds::getSurfelCloud(v4r::DataMatrix2D<Surfel> &cloud, Eigen::Matrix4f &pose, double &timestamp)
{
  mtx_shm.lock();
  cloud = sf_cloud;
  timestamp = sf_timestamp;
  pose = sf_pose;
  mtx_shm.unlock();
}



/**
 * setCameraParameter
 */
void TSFOptimizeClouds::setCameraParameter(const cv::Mat &_intrinsic, const cv::Mat &_dist_coeffs)
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
}

/**
 * setCameraParameter
 */
void TSFOptimizeClouds::setCameraParameterTgt(const cv::Mat &_intrinsic, int _width, int _height)
{
  width = _width;
  height = _height;

  if (_intrinsic.type() != CV_64F)
    _intrinsic.convertTo(tgt_intrinsic, CV_64F);
  else tgt_intrinsic = _intrinsic;

  reset();
}
/**
 * @brief TSFOptimizeClouds::setParameter
 * @param p
 */
void TSFOptimizeClouds::setParameter(const Parameter &p)
{
  param = p;
  sqr_cam_distance_select_frame = param.cam_distance_select_frame*param.cam_distance_select_frame;
  cos_delta_angle_select_frame = cos(param.angle_select_frame*M_PI/180.);
  if (param.batch_size_clouds<3)
    throw std::runtime_error("[TSFOptimizeClouds::setParameter] batch_size_clouds need to be > 2");
}


}












