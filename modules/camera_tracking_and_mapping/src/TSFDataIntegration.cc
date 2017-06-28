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


#include <v4r/camera_tracking_and_mapping/TSFDataIntegration.hh>
#include <v4r/keypoints/impl/invPose.hpp>
#include <v4r/common/convertImage.h>
#include <pcl/common/transforms.h>



namespace v4r
{


using namespace std;


std::vector<cv::Vec4i> TSFDataIntegration::npat = std::vector<cv::Vec4i>();



/************************************************************************************
 * Constructor/Destructor
 */
TSFDataIntegration::TSFDataIntegration(const Parameter &p)
 : run(false), have_thread(false), data(NULL)
{ 
  setParameter(p);
}

TSFDataIntegration::~TSFDataIntegration()
{
  if (have_thread) stop();
}

/**
 * operate
 */
void TSFDataIntegration::operate()
{
  bool have_todo;

  pcl::PointCloud<pcl::PointXYZRGB> cloud;  ///// new cloud
  Eigen::Matrix4f pose;      /// global pose of the current frame (depth, gray, points[1], ....)
  uint64_t timestamp = 0;
  v4r::DataMatrix2D<Surfel> filt_cloud;
  Eigen::Matrix4f filt_pose;

  while(run)
  {
    have_todo = false;
    data->lock();
    if (data->timestamp!=data->filt_timestamp)
    {
      have_todo = true;
      cloud = data->cloud;
      filt_cloud = *data->filt_cloud;
      timestamp = data->timestamp;
      pose = data->pose;
      filt_pose = data->filt_pose;
    }
    data->unlock();

    if (have_todo)
    {
      //v4r::ScopeTime t("TSFDataIntegration::operate");
      if ((int)cloud.width!=filt_cloud.cols || (int)cloud.height!=filt_cloud.rows)
      {
        initCloud(cloud, filt_cloud);
      }
      else
      {
        integrateData(cloud, pose, filt_pose, filt_cloud);
        //computeNormals(filt_cloud);
      }

      data->lock();
      *data->filt_cloud = filt_cloud;
      data->filt_timestamp = timestamp;
      data->filt_pose = pose;
      data->nb_frames_integrated++;
      if (std::isnan(data->last_pose_map(0,0)) || selectFrame(data->last_pose_map, pose))
      {
        if (data->filt_cloud->data.size()>0 && data->nb_frames_integrated>param.min_frames_integrated)
        {
          data->map_frames.push( TSFFrame::Ptr( new TSFFrame(-1,pose,*data->filt_cloud,(data->cnt_pose_lost_map>0?false:true)) ) );
          data->cnt_pose_lost_map = 0;
          data->last_pose_map = pose;
        }
      }
      data->unlock();
    }

    if (!have_todo) usleep(10000);
  }
}

/**
 * @brief TSFDataIntegration::selectintegrateDataFrame
 * @param pose0
 * @param pose1
 * @return
 */
bool TSFDataIntegration::selectFrame(const Eigen::Matrix4f &pose0, const Eigen::Matrix4f &pose1)
{
  invPose(pose0, inv_pose0);
  invPose(pose1, inv_pose1);
  if ( (inv_pose0.block<3,1>(0,2).dot(inv_pose1.block<3,1>(0,2)) > cos_diff_delta_angle_map) &&
       (inv_pose0.block<3,1>(0,3)-inv_pose1.block<3,1>(0,3)).squaredNorm() < sqr_diff_cam_distance_map )
    return false;
  return true;
}

/**
 * @brief TSFDataIntegration::initCloud
 * @param cloud
 * @param sf_cloud
 */
void TSFDataIntegration::initCloud(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, v4r::DataMatrix2D<Surfel> &sf_cloud)
{
  if (sf_cloud.rows!=(int)cloud.height || sf_cloud.cols!=(int)cloud.width)
  {
    sf_cloud.resize(cloud.height, cloud.width);
    for (unsigned v=0; v<cloud.height; v++)
    {
      for (unsigned u=0; u<cloud.width; u++)
      {
        sf_cloud(v,u) = Surfel(cloud(u,v));
      }
    }
  }
}



/**
 * @brief TSFDataIntegration::addCloud
 */
void TSFDataIntegration::integrateData(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const Eigen::Matrix4f &pose, const Eigen::Matrix4f &filt_pose, v4r::DataMatrix2D<Surfel> &filt_cloud)
{
  if (intrinsic.empty())
    throw std::runtime_error("[TSFDataIntegration::addCloud] Camera parameter not set!");

  cv::Point2f im_pt;
  int width = cloud.width;
  int height = cloud.height;
  double *C = &intrinsic(0,0);
  double invC0 = 1./C[0];
  double invC4 = 1./C[4];
  Eigen::Matrix4f inv_sf, inc_pose;
  v4r::invPose(filt_pose, inv_sf);
  inc_pose = pose*inv_sf;


  //transform to current frame
  Eigen::Vector3f pt,n;
  Eigen::Matrix3f R = inc_pose.topLeftCorner<3,3>();
  Eigen::Vector3f t = inc_pose.block<3,1>(0,3);
  int x, y;
  float ax, ay, inv_z, norm;
  depth_norm = cv::Mat_<float>::zeros(height, width);
  depth_weight = cv::Mat_<float>::zeros(height, width);
  tmp_z = cv::Mat_<float>::zeros(height, width);
  nan_z = cv::Mat_<float>(height,width);

  // get occlusion map
  if (param.filter_occlusions)
    occ.compute(cloud, occ_mask);

  // tranform filt cloud to current frame and update
  for (int v=0; v<filt_cloud.rows; v++)
  {
    for (int u=0; u<filt_cloud.cols; u++)
    {
      const Surfel &s = filt_cloud(v,u);

      if (std::isnan(s.pt[0])||std::isnan(s.pt[1])||std::isnan(s.pt[2]))
        continue;

      pt = R*s.pt+t;
      inv_z = 1./pt[2];
      im_pt.x = C[0]*pt[0]*inv_z + C[2];
      im_pt.y = C[4]*pt[1]*inv_z + C[5];
      x = (int)(im_pt.x);
      y = (int)(im_pt.y);

      if (x<=0 || y<=0 || x>=width-1 || y>=height-1)
        continue;

      ax = im_pt.x-x;
      ay = im_pt.y-y;
      float *dn = &depth_norm(y,x);
      float *dw = &depth_weight(y,x);
      float *tz = &tmp_z(y,x);

      {
        norm = 0;
        const float &d = cloud(x,y).z;
        if (!std::isnan(d))
        {
          int err_idx = (int)(fabs(inv_z-1./d)*1000.);
          if (err_idx<1000) norm = exp_error_lookup[err_idx];
        }
        else
        {
          if (dn[0]<=std::numeric_limits<float>::epsilon())
          {
            norm = 1.;
            nan_z(y,x) = pt[2];
          }
          else
          {
            int err_idx = (int)(fabs(inv_z-1./nan_z(y,x))*1000.);
            if (err_idx<1000) norm = exp_error_lookup[err_idx];
          }
        }
        norm *= (1.-ax) * (1.-ay);
        tz[0] += norm*pt[2];
        dw[0] += norm*s.weight;
        dn[0] += norm;
      }
      {
        norm = 0;
        const float &d = cloud(x+1,y).z;
        if (!std::isnan(d))
        {
          int err_idx = (int)(fabs(inv_z-1./d)*1000.);
          if (err_idx<1000) norm = exp_error_lookup[err_idx];
        }
        else
        {
          if (dn[1]<=std::numeric_limits<float>::epsilon())
          {
            norm = 1.;
            nan_z(y,x+1) = pt[2];
          }
          else
          {
            int err_idx = (int)(fabs(inv_z-1./nan_z(y,x+1))*1000.);
            if (err_idx<1000) norm = exp_error_lookup[err_idx];
          }
        }
        norm *= ax * (1.-ay);
        tz[1] += norm*pt[2];
        dw[1] += norm*s.weight;
        dn[1] += norm;
      }
      {
        norm = 0;
        const float &d = cloud(x, y+1).z;
        if (!std::isnan(d))
        {
          int err_idx = (int)(fabs(inv_z-1./d)*1000.);
          if (err_idx<1000) norm = exp_error_lookup[err_idx];
        }
        else
        {
          if (dn[width]<=std::numeric_limits<float>::epsilon())
          {
            norm = 1.;
            nan_z(y+1,x) = pt[2];
          }
          else
          {
            int err_idx = (int)(fabs(inv_z-1./nan_z(y+1,x))*1000.);
            if (err_idx<1000) norm = exp_error_lookup[err_idx];
          }
        }
        norm *= (1.-ax) *  ay;
        tz[width] += norm*pt[2];
        dw[width] += norm*s.weight;
        dn[width] += norm;
      }
      {
        norm = 0;
        const float &d = cloud(x+1, y+1).z;
        if (!std::isnan(d))
        {
          int err_idx = (int)(fabs(inv_z-1./d)*1000.);
          if (err_idx<1000) norm = exp_error_lookup[err_idx];
        }
        else
        {
          if (dn[width+1]<=std::numeric_limits<float>::epsilon())
          {
            norm = 1.;
            nan_z(y+1,x+1) = pt[2];
          }
          else
          {
            int err_idx = (int)(fabs(inv_z-1./nan_z(y+1,x+1))*1000.);
            if (err_idx<1000) norm = exp_error_lookup[err_idx];
          }
        }
        norm *= ax * ay;
        tz[width+1] += norm*pt[2];
        dw[width+1] += norm*s.weight;
        dn[width+1] += norm;
      }
    }
  }

  // integrate new data
  float inv_norm;
  for (unsigned v=0; v<cloud.height; v++)
  {
    for (unsigned u=0; u<cloud.width; u++)
    {
      const float &dw = depth_weight(v,u);
      Surfel &sf = filt_cloud(v,u);
      const pcl::PointXYZRGB &pt = cloud(u,v);

      if (!param.filter_occlusions || occ_mask(v,u)<128)
      {
        if (fabs(dw)<=std::numeric_limits<float>::epsilon())
        {
          sf = Surfel(pt);
        }
        else
        {
          const float &tz = tmp_z(v,u);
          inv_norm = 1./depth_norm(v,u);
          sf.pt[2] = tz*inv_norm;
          sf.weight = dw*inv_norm;
          if (!isnan(pt.z))
            sf.pt[2] = (sf.pt[2]*(float)sf.weight + pt.z) / (float)(sf.weight+1.);
          sf.pt[0] = sf.pt[2]*((u-C[2])*invC0);
          sf.pt[1] = sf.pt[2]*((v-C[5])*invC4);
          if (sf.weight<param.max_integration_frames) sf.weight+=1;
        }
      } else sf = Surfel();
      sf.r = pt.r;
      sf.g = pt.g;
      sf.b = pt.b;
    }
  }
}





/***************************************************************************************/

/**
 * start
 */
void TSFDataIntegration::start()
{
  if (data==NULL)
    throw std::runtime_error("[TSFDataIntegration::start] No data available! Did you call 'setData'?");

  if (have_thread) stop();

  run = true;
  th_obectmanagement = boost::thread(&TSFDataIntegration::operate, this);  
  have_thread = true;
}

/**
 * stop
 */
void TSFDataIntegration::stop()
{
  run = false;
  th_obectmanagement.join();
  have_thread = false;
}




/**
 * reset
 */
void TSFDataIntegration::reset()
{
  stop();
}


/**
 * @brief TSFDataIntegration::computeRadius
 * @param sf_cloud
 */
void TSFDataIntegration::computeRadius(v4r::DataMatrix2D<Surfel> &sf_cloud, const cv::Mat_<double> &intrinsic)
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
 * @brief TSFDataIntegration::computeNormals
 * @param sf_cloud
 */
void TSFDataIntegration::computeNormals(v4r::DataMatrix2D<Surfel> &sf_cloud, int nb_dist)
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



/**
 * setCameraParameter
 */
void TSFDataIntegration::setCameraParameter(const cv::Mat &_intrinsic)
{
  if (_intrinsic.type() != CV_64F)
    _intrinsic.convertTo(intrinsic, CV_64F);
  else intrinsic = _intrinsic;

  reset();
}

/**
 * @brief TSFDataIntegration::setParameter
 * @param p
 */
void TSFDataIntegration::setParameter(const Parameter &p)
{
  param = p;
  exp_error_lookup.resize(1000);
  for (unsigned i=0; i<exp_error_lookup.size(); i++)
  {
    exp_error_lookup[i] = exp(-sqr(((float)i)/1000.)/(2.*sqr(param.sigma_depth)));
  }
  sqr_diff_cam_distance_map = param.diff_cam_distance_map*param.diff_cam_distance_map;
  cos_diff_delta_angle_map = cos(param.diff_delta_angle_map*M_PI/180.);
}


}












