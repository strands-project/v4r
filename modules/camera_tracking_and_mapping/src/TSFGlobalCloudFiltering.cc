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


#include <v4r/camera_tracking_and_mapping/TSFGlobalCloudFiltering.hh>
#include <v4r/keypoints/impl/invPose.hpp>
#include <v4r/common/convertImage.h>
#include "pcl/common/transforms.h"

#include "opencv2/highgui/highgui.hpp"

//#define DEBUG_WEIGHTING

namespace v4r
{


using namespace std;






/************************************************************************************
 * Constructor/Destructor
 */
TSFGlobalCloudFiltering::TSFGlobalCloudFiltering(const Parameter &p)
{
  setParameter(p);
  npat.resize(4);
  npat[0] = cv::Vec4i(1,0,0,1);
  npat[1] = cv::Vec4i(0,1,-1,0);
  npat[2] = cv::Vec4i(-1,0,0,-1);
  npat[3] = cv::Vec4i(0,-1,0,1);
}

TSFGlobalCloudFiltering::~TSFGlobalCloudFiltering()
{
}


/**
 * @brief TSFGlobalCloudFiltering::computeReliability
 * @param frames
 */
void TSFGlobalCloudFiltering::computeReliability(const std::vector<TSFFrame::Ptr> &frames)
{
  reliability.resize(frames.size());


  for (unsigned i=0; i<frames.size(); i++)
  {
    const v4r::DataMatrix2D<v4r::Surfel> &frame = frames[i]->sf_cloud;

#ifdef DEBUG_WEIGHTING
        cv::Mat_<unsigned char> im = cv::Mat_<unsigned char>::zeros(frame.rows,frame.cols);
#endif
    cv::Mat_<float> &rel = reliability[i];
    rel = cv::Mat_<float>::zeros(frame.rows, frame.cols);
    for (int v=0; v<frame.rows; v++)
    {
      for (int u=0; u<frame.cols; u++)
      {
        const v4r::Surfel &s = frame(v,u);
        if (isnan(s.n[0]) || isnan(s.pt[0]) || isnan(s.n[1]) || isnan(s.pt[1]) || isnan(s.n[2]) || isnan(s.pt[2]))
          continue;
        double cosa = s.n.dot(-s.pt.normalized());
        if (cosa>cos_thr_angle)
        {
          double d_cnt = (s.weight>param.max_weight?0.:param.max_weight - s.weight);
          rel(v,u) = cosa * exp(d_cnt*d_cnt*neg_inv_sqr_sigma_pts) * 1./s.pt[2];
#ifdef DEBUG_WEIGHTING
          im(v,u) = rel(v,u) * 100;
#endif
        }
      }
    }
#ifdef DEBUG_WEIGHTING
    cv::imshow("dbg", im);
    cv::waitKey(0);
#endif
  }

}

/**
 * @brief TSFGlobalCloudFiltering::maxReliabilityIndexing
 * @param frames
 * @param poses
 */
void TSFGlobalCloudFiltering::maxReliabilityIndexing(const std::vector<TSFFrame::Ptr> &frames)
{
  if (intrinsic.empty())
    throw std::runtime_error("[TSFGlobalCloudFiltering::addCloud] Camera parameter not set!");

  cv::Point2f im_pt;
  double *C = &intrinsic(0,0);
  double invC0 = 1./C[0];
  double invC4 = 1./C[4];
  Eigen::Matrix4f inv_pose_j, inc_pose;

  //transform to current frame
  Eigen::Vector3f pt, n;
  Eigen::Matrix3f R;
  Eigen::Vector3f t;
  int x, y;
  float inv_z, ax, ay;
  float weight;
  cv::Mat_<double> norm;
  cv::Mat_<double> depth;
  cv::Mat_<cv::Vec3d> col;

  for (unsigned i=0; i<frames.size(); i++)
  {
    v4r::DataMatrix2D<v4r::Surfel> &frame_i = frames[i]->sf_cloud;
    const cv::Mat_<float> &rel_i = reliability[i];
    const Eigen::Matrix4f &pose_i = frames[i]->pose;

    // init
    norm = cv::Mat_<double>(frame_i.rows, frame_i.cols);
    depth = cv::Mat_<double>(frame_i.rows, frame_i.cols);
    col = cv::Mat_<cv::Vec3d>(frame_i.rows, frame_i.cols);
    for (unsigned j=0; j<frame_i.data.size(); j++)
    {
      const v4r::Surfel &s = frame_i[j];
      const float &n = rel_i(j);
      norm(j) = n;
      depth(j) = s.pt[2]*n;
      col(j) = cv::Vec3d(n*s.b, n*s.g, n*s.r);
    }

    // integrate data
    for (unsigned j=0; j<frames.size(); j++)
    {
      if (i==j) continue;

      const v4r::DataMatrix2D<v4r::Surfel> &frame_j = frames[j]->sf_cloud;
      const cv::Mat_<float> &rel_j = reliability[j];
      v4r::invPose(frames[j]->pose, inv_pose_j);
      inc_pose = pose_i*inv_pose_j;
      R = inc_pose.topLeftCorner<3,3>();
      t = inc_pose.block<3,1>(0,3);
      int width = frame_j.cols;

      for (int v=0; v<frame_j.rows; v++)
      {
        for (int u=0; u<frame_j.cols; u++)
        {
          const v4r::Surfel &s_j = frame_j(v,u);
          if (isnan(s_j.n[0]) || isnan(s_j.pt[0]) || isnan(s_j.n[1]) || isnan(s_j.pt[1]) || isnan(s_j.n[2]) || isnan(s_j.pt[2]))
            continue;

          pt = R*s_j.pt+t;
          n = R*s_j.n;
          inv_z = 1./pt[2];
          im_pt.x = C[0]*pt[0]*inv_z + C[2];
          im_pt.y = C[4]*pt[1]*inv_z + C[5];
          x = (int)(im_pt.x);
          y = (int)(im_pt.y);

          if (x>=0 && y>=0 && x<frame_i.cols-1 && y<frame_i.rows-1)
          {
            ax = im_pt.x-x;
            ay = im_pt.y-y;
            double *dn = &norm(y,x);
            cv::Vec3d *dc = &col(y,x);
            double *dz = &depth(y,x);
            // 00
            {
              const v4r::Surfel &s_i = frame_i(y,x);
              if (!isnan(s_i.n[0]) &&  !isnan(s_i.pt[2]))
              {
                if (n.dot(-s_i.pt.normalized())>cos_thr_angle)   // do not consider backfacing points
                {
                  if (fabs(inv_z-1./s_i.pt[2]) < param.z_cut_off_integration)
                  {
                    int err_idx = (int)(fabs(inv_z-1./s_i.pt[2])*1000.);
                    weight = rel_j(v,u);
                    weight *= (err_idx<1000 ? exp_error_lookup[err_idx] : 0.);
//                    cout<<exp_error_lookup[err_idx];
                    weight *= (1.-ax) * (1.-ay);
                    dz[0] += weight*pt[2];
                    dc[0] += cv::Vec3d(weight*s_j.b, weight*s_j.g, weight*s_j.r);
                    dn[0] += weight;
                  }
//                  cout<<") "<<endl;
                }
              }
            }
            // 10
            {
              const v4r::Surfel &s_i = frame_i(y,x+1);
              if (!isnan(s_i.n[0]) &&  !isnan(s_i.pt[2]))
              {
                if (n.dot(-s_i.pt.normalized())>cos_thr_angle)   // do not consider backfacing points
                {
                  if (fabs(inv_z-1./s_i.pt[2]) < param.z_cut_off_integration)
                  {
                    int err_idx = (int)(fabs(inv_z-1./s_i.pt[2])*1000.);
                    weight = rel_j(v,u);
                    weight *= (err_idx<1000 ? exp_error_lookup[err_idx] : 0.);
                    weight *= ax * (1.-ay);
                    dz[1] += weight*pt[2];
                    dc[1] += cv::Vec3d(weight*s_j.b, weight*s_j.g, weight*s_j.r);
                    dn[1] += weight;
                  }
                }
              }
            }
            // 01
            {
              const v4r::Surfel &s_i = frame_i(y+1,x);
              if (!isnan(s_i.n[0]) &&  !isnan(s_i.pt[2]))
              {
                if (n.dot(-s_i.pt.normalized())>cos_thr_angle)   // do not consider backfacing points
                {
                  if (fabs(inv_z-1./s_i.pt[2]) < param.z_cut_off_integration)
                  {
                    int err_idx = (int)(fabs(inv_z-1./s_i.pt[2])*1000.);
                    weight = rel_j(v,u);
                    weight *= (err_idx<1000 ? exp_error_lookup[err_idx] : 0.);
                    weight *= (1.-ax) *  ay;
                    dz[width] += weight*pt[2];
                    dc[width] += cv::Vec3d(weight*s_j.b, weight*s_j.g, weight*s_j.r);
                    dn[width] += weight;
                  }
                }
              }
            }
            // 11
            {
              const v4r::Surfel &s_i = frame_i(y+1,x+1);
              if (!isnan(s_i.n[0]) &&  !isnan(s_i.pt[2]))
              {
                if (n.dot(-s_i.pt.normalized())>cos_thr_angle)   // do not consider backfacing points
                {
                  if (fabs(inv_z-1./s_i.pt[2]) < param.z_cut_off_integration)
                  {
                    int err_idx = (int)(fabs(inv_z-1./s_i.pt[2])*1000.);
                    weight = rel_j(v,u);
                    weight *= (err_idx<1000 ? exp_error_lookup[err_idx] : 0.);
                    weight *= ax * ay;
                    dz[width+1] += weight*pt[2];
                    dc[width+1] += cv::Vec3d(weight*s_j.b, weight*s_j.g, weight*s_j.r);
                    dn[width+1] += weight;
                  }
                }
              }
            }
          }
        }
      }
    }
    // integrate new data
    double inv_norm;
    for (int v=0; v<frame_i.rows; v++)
    {
      for (int u=0; u<frame_i.cols; u++)
      {
        const double &dn = norm(v,u);
        v4r::Surfel &sf = frame_i(v,u);
        if (fabs(dn)<=std::numeric_limits<double>::epsilon())
          continue;
        inv_norm = 1./dn;
        sf.pt[2] = (float)depth(v,u)*inv_norm;
        sf.pt[0] = sf.pt[2]*((u-C[2])*invC0);
        sf.pt[1] = sf.pt[2]*((v-C[5])*invC4);
        sf.r = inv_norm*col(v,u)[2];
        sf.g = inv_norm*col(v,u)[1];
        sf.b = inv_norm*col(v,u)[0];
      }
    }
    // update normals
    computeNormals(frame_i);
  }
}



/**
 * @brief TSFDataIntegration::computeNormals
 * @param sf_cloud
 */
void TSFGlobalCloudFiltering::computeNormals(v4r::DataMatrix2D<v4r::Surfel> &sf_cloud)
{
  v4r::Surfel *s1, *s2, *s3;
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
 * @brief TSFGlobalCloudFiltering::getMaxPoints
 * @param frames
 * @param poses
 * @param cloud
 */
void TSFGlobalCloudFiltering::getMaxPoints(const std::vector<TSFFrame::Ptr> &frames, pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud)
{
  cloud.clear();

  if (frames.size()==0)
    return;

  Eigen::Matrix4f inv_pose;
  Eigen::Vector3f pt, n;
  Eigen::Matrix3f R;
  Eigen::Vector3f t;
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr tmp_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
  pcl::PointCloud<pcl::PointXYZRGBNormal> &ref = *tmp_cloud;

  oc_cloud.reset(new AlignedPointXYZRGBNormalVector () );
  octree.reset(new pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZRGBNormal,pcl::octree::OctreeVoxelCentroidContainerXYZRGBNormal<pcl::PointXYZRGBNormal> >(param.voxel_size));
  octree->setResolution(param.voxel_size);
  int cnt_all=0;

  // transform points and add to octree
  for (unsigned i=0; i<frames.size(); i++)
  {
    ref.clear();
    const v4r::DataMatrix2D<v4r::Surfel> &frame = frames[i]->sf_cloud;
    const cv::Mat_<float> &rel = reliability[i];
    v4r::invPose(frames[i]->pose, inv_pose);
    R = inv_pose.topLeftCorner<3,3>();
    t = inv_pose.block<3,1>(0,3);
    for (int v=0; v<frame.rows; v++)
    {
      for (int u=0; u<frame.cols; u++)
      {
        const v4r::Surfel &s = frame(v,u);
        if (isnan(s.n[0]) || isnan(s.pt[0]) || isnan(s.n[1]) || isnan(s.pt[1]) || isnan(s.n[2]) || isnan(s.pt[2]))
          continue;
        cnt_all++;
        if (s.pt[2]>param.max_dist_integration)
          continue;
        if (s.n.dot(-s.pt.normalized())<cos_thr_angle)
          continue;
//        if (rel(v,u)<0.2)
//          continue;
        pt = R*s.pt+t;
        n = R*s.n;
        ref.points.push_back(pcl::PointXYZRGBNormal());
        pcl::PointXYZRGBNormal &pcl_pt = ref.points.back();
        pcl_pt.getVector3fMap() = pt;
        pcl_pt.getNormalVector3fMap() = n;
        pcl_pt.r = s.r;
        pcl_pt.g = s.g;
        pcl_pt.b = s.b;
      }
    }    
    ref.height = 1;
    ref.width = ref.points.size();
    ref.is_dense = true;
    octree->setInputCloud(tmp_cloud);
    octree->addPointsFromInputCloud();
  }

  // return point cloud
  octree->getVoxelCentroids(*oc_cloud);
  const AlignedPointXYZRGBNormalVector &oc = *oc_cloud;
  cloud.resize(oc.size());
  for (unsigned i=0; i<oc.size(); i++)
      cloud.points[i] = oc[i];
  cloud.height=1;
  cloud.width=cloud.points.size();
  cloud.is_dense = true;

  cout<<"num of filtered points: "<<cloud.points.size()<<"/"<<cnt_all<<endl;
}




/***************************************************************************************/


/**
 * @brief TSFGlobalCloudFiltering::computeRadius
 * @param sf_cloud
 */
void TSFGlobalCloudFiltering::computeRadius(v4r::DataMatrix2D<v4r::Surfel> &sf_cloud)
{
  const float norm = 1./sqrt(2)*(2./(intrinsic(0,0)+intrinsic(1,1)));
  for (int v=0; v<sf_cloud.rows; v++)
  {
    for (int u=0; u<sf_cloud.cols; u++)
    {
      v4r::Surfel &s  = sf_cloud(v,u);
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
 * @brief filter
 * @param frames
 */
void TSFGlobalCloudFiltering::filter(const std::vector<TSFFrame::Ptr> &frames, pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud, const Eigen::Matrix4f &base_transform)
{
  cout<<"[TSFGlobalCloudFiltering::filter] compute point reliability..."<<endl;

  computeReliability(frames);

  cout<<"[TSFGlobalCloudFiltering::filter] point projection and max. reliability indexing..."<<endl;

  maxReliabilityIndexing(frames);

  cout<<"[TSFGlobalCloudFiltering::filter] TODO..."<<endl;

  // clean up mamory
  reliability = std::vector< cv::Mat_<float> >();

  getMaxPoints(frames, cloud);

  // clean up mamory
  oc_cloud.reset(new AlignedPointXYZRGBNormalVector () );
  octree.reset(new pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZRGBNormal,pcl::octree::OctreeVoxelCentroidContainerXYZRGBNormal<pcl::PointXYZRGBNormal> >(param.voxel_size));
}


/**
 * setCameraParameter
 */
void TSFGlobalCloudFiltering::setCameraParameter(const cv::Mat &_intrinsic)
{
  if (_intrinsic.type() != CV_64F)
    _intrinsic.convertTo(intrinsic, CV_64F);
  else intrinsic = _intrinsic;
}

/**
 * @brief TSFGlobalCloudFiltering::setParameter
 * @param p
 */
void TSFGlobalCloudFiltering::setParameter(const Parameter &p)
{
  param = p;
  cos_thr_angle = cos(p.thr_angle*M_PI/180.);
  neg_inv_sqr_sigma_pts = -1./(2.*p.sigma_pts*p.sigma_pts);
  exp_error_lookup.resize(1000);
  for (unsigned i=0; i<exp_error_lookup.size(); i++)
  {
    exp_error_lookup[i] = exp(-sqr(((float)i)/1000.)/(2.*sqr(param.sigma_depth)));
  }
}


}












