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
#include "ObjectSegmentation.h"

#include <cmath>
#include <boost/filesystem.hpp>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/filters/voxel_grid.h>

#include <opencv2/highgui/highgui.hpp>
#include <v4r/keypoints/impl/invPose.hpp>
#include <v4r/keypoints/impl/PoseIO.hpp>
#include <v4r/common/convertCloud.h>
#include <v4r/common/convertNormals.h>
#include <v4r/common/convertImage.h>
#include <v4r/reconstruction/impl/projectPointToImage.hpp>
#include <v4r/common/noise_models.h>
#include <v4r/registration/noise_model_based_cloud_integration.h>
#include <v4r/registration/MvLMIcp.h>

//#define ER_TEST_MLS

#ifdef ER_TEST_MLS
#include <pcl/surface/mls.h>
#endif
#endif

using namespace std;


/**
 * @brief ObjectSegmentation::ObjectSegmentation
 */
ObjectSegmentation::ObjectSegmentation()
 : cmd(UNDEF), m_run(false), image_idx(-1), first_click(true),
   have_roi(false), roi_pose(Eigen::Matrix4f::Identity()), bb_min(Eigen::Vector3f(0.,0.,0.)), bb_max(Eigen::Vector3f(0.,0.,0.)),
   object_base_transform(Eigen::Matrix4f::Identity()), use_roi_segmentation(false), roi_offs(0.01), use_dense_mv(false)
{
  v4r::ZAdaptiveNormals::Parameter n_param;
  n_param.adaptive = true;
  nest.reset(new v4r::ZAdaptiveNormals(n_param));

  v4r::ClusterNormalsToPlanes::Parameter p_param;
  p_param.thrAngle=45;
  p_param.inlDist=0.01;
  p_param.minPoints=5000;
  p_param.least_squares_refinement=false;
  p_param.smooth_clustering=true;
  p_param.thrAngleSmooth=30;
  p_param.inlDistSmooth=0.01;
  p_param.minPointsSmooth=3;
  pest.reset(new v4r::ClusterNormalsToPlanes(p_param));

  vx_size = 0.001;
  max_dist = 0.01f;
  max_iterations = 50;
  diff_type = 1;

  tmp_cloud1.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
  tmp_cloud2.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
  oc_cloud.reset(new Sensor::AlignedPointXYZRGBVector () );
}

/**
 * @brief ObjectSegmentation::~ObjectSegmentation
 */
ObjectSegmentation::~ObjectSegmentation()
{

}



/******************************** public *******************************/

/**
 * @brief ObjectSegmentation::start
 * @param cam_id
 */
void ObjectSegmentation::start()
{
  QThread::start();
}

/**
 * @brief ObjectSegmentation::stop
 */
void ObjectSegmentation::stop()
{
  if(m_run)
  {
    m_run = false;
    this->wait();
  }
}

/**
 * @brief ObjectSegmentation::isRunning
 * @return
 */
bool ObjectSegmentation::isRunning()
{
  return m_run;
}


/**
 * @brief ObjectSegmentation::finishedSegmentation
 */
void ObjectSegmentation::finishSegmentation()
{
  cmd = FINISH_OBJECT_MODELLING;
  start();
}

/**
 * @brief ObjectSegmentation::optimizeMultiview
 */
bool ObjectSegmentation::optimizeMultiview()
{
  if (clouds.get()!=0 && clouds->size()>0)
  {
    cmd = OPTIMIZE_MULTIVIEW;
    start();
    return true;
  }
  return false;
}

/**
 * @brief ObjectSegmentation::activateROI
 * @param enable
 */
void ObjectSegmentation::activateROI(bool enable)
{
  if (fabs(bb_min[0]-bb_max[0])<0.00001)
    have_roi=false;
  else have_roi=enable;
}

/**
 * @brief ObjectSegmentation::set_segmentation_params
 * @param use_roi_segm
 * @param offs
 * @param _use_dense_mv
 */
void ObjectSegmentation::set_segmentation_params(bool use_roi_segm, const double &offs, bool _use_dense_mv, const double &_edge_radius_px)
{
  use_roi_segmentation = use_roi_segm;
  roi_offs = offs;
  use_dense_mv = _use_dense_mv;
  om_params.edge_radius_px = _edge_radius_px;
}

/**
 * @brief ObjectSegmentation::set_roi
 * @param _bb_min
 * @param _bb_max
 * @param _roi_pose
 */
void ObjectSegmentation::set_roi(const Eigen::Vector3f &_bb_min, const Eigen::Vector3f &_bb_max, const Eigen::Matrix4f &_roi_pose)
{
  bb_min = _bb_min;
  bb_max = _bb_max;
  object_base_transform = roi_pose = _roi_pose;
  have_roi=true;
}

/**
 * @brief ObjectSegmentation::cam_params_changed
 * @param _cam_params
 */
void ObjectSegmentation::cam_params_changed(const RGBDCameraParameter &_cam_params)
{
  intrinsic = cv::Mat_<double>::eye(3,3);

  intrinsic(0,0) = _cam_params.f[0]; intrinsic(1,1) = _cam_params.f[1];
  intrinsic(0,2) = _cam_params.c[0]; intrinsic(1,2) = _cam_params.c[1];
}

/**
 * @brief ObjectSegmentation::segmentation_parameter_changed
 * @param param
 */
void ObjectSegmentation::segmentation_parameter_changed(const SegmentationParameter& param)
{
  seg_params = param;

  v4r::ClusterNormalsToPlanes::Parameter p_param;

  p_param.thrAngle = 45;
  p_param.inlDist = seg_params.inl_dist_plane;
  p_param.minPoints = seg_params.min_points_plane;
  p_param.least_squares_refinement = false;
  p_param.smooth_clustering = true;
  p_param.thrAngleSmooth = seg_params.thr_angle;
  p_param.inlDistSmooth = 0.01;
  p_param.minPointsSmooth = 3;

  pest.reset(new v4r::ClusterNormalsToPlanes(p_param));
}

/**
 * @brief ObjectSegmentation::object_modelling_parameter_changed
 * @param param
 */
void ObjectSegmentation::object_modelling_parameter_changed(const ObjectModelling& param)
{
  om_params = param;
}

/**
 * @brief ObjectSegmentation::storeMasks
 * @param _folder
 */
void ObjectSegmentation::storeMasks(const std::string &_folder)
{
  char filename[PATH_MAX];

  std::string mask_names = _folder+"/mask_%04d.png";

  for (unsigned i=0; i<masks.size(); i++)
  {
    snprintf(filename,PATH_MAX, mask_names.c_str(), i);
    cv::imwrite(filename, masks[i]);
  }
}

/**
 * @brief ObjectSegmentation::storePointCloudModel
 * @param _folder
 */
bool ObjectSegmentation::storePointCloudModel(const std::string &_folder)
{
  pcl::PointCloud<pcl::PointXYZRGB> &cloud = *tmp_cloud1;
  const Sensor::AlignedPointXYZRGBVector &oc = *oc_cloud;

  cloud.resize(oc.size());
  cloud.width = oc.size();
  cloud.height = 1;
  cloud.is_dense = true;

  for (unsigned i=0; i<oc.size(); i++)
    cloud.points[i] = oc[i];

  if (cloud.points.size()>0)
  {
    pcl::io::savePCDFileBinary(_folder+"/model.pcd", cloud);
    return true;
  }

  return false;
}

/**
 * @brief ObjectSegmentation::savePointClouds
 * @param _folder
 * @param _modelname
 * @return
 */
bool ObjectSegmentation::savePointClouds(const std::string &_folder, const std::string &_modelname)
{
  if (octree_cloud.get()==0 || big_normals.get()==0 || clouds.get()==0 || clouds->empty() ||
      octree_cloud->empty() || octree_cloud->points.size()!=big_normals->points.size() || clouds->size()!=indices.size())
    return false;

  char filename[PATH_MAX];
  boost::filesystem::create_directories(_folder + "/models/" + _modelname + "/views" );

  // create model cloud with normals and save it
  pcl::PointCloud<pcl::PointXYZRGBNormal> ncloud;
  pcl::concatenateFields(*octree_cloud, *big_normals, ncloud);
  pcl::io::savePCDFileBinary(_folder + "/models/" + _modelname + "/3D_model.pcd", ncloud);

  std::string cloud_names = _folder + "/models/" + _modelname + "/views/cloud_%08d.pcd";
  std::string image_names = _folder + "/models/" + _modelname + "/views/image_%08d.jpg";
  std::string pose_names = _folder + "/models/" + _modelname + "/views/pose_%08d.txt";
  std::string mask_names = _folder + "/models/" + _modelname + "/views/mask_%08d.png";
  std::string idx_names = _folder + "/models/" + _modelname + "/views/object_indices_%08d.txt";


  for (unsigned i=0; i<clouds->size(); i++)
  {
    if (indices[i].empty()) continue;

    // store indices
    snprintf(filename, PATH_MAX, idx_names.c_str(), i);
    std::ofstream mask_f (filename);
    for(unsigned j=0; j < indices[i].size(); j++)
        mask_f << indices[i][j] << std::endl;
    mask_f.close();

    // store cloud
    snprintf(filename,PATH_MAX, cloud_names.c_str(), i);
    pcl::io::savePCDFileBinary(filename, *clouds->at(i).second);

    // store image
    v4r::convertImage(*clouds->at(i).second, image);
    snprintf(filename,PATH_MAX, image_names.c_str(), i);
    cv::imwrite(filename, image);

    // store poses
    snprintf(filename,PATH_MAX, pose_names.c_str(), i);
    v4r::writePose(filename, std::string(), inv_poses[i]);

    // store masks
    snprintf(filename,PATH_MAX, mask_names.c_str(), i);
    cv::imwrite(filename, masks[i]);
  }

  return true;
}


/**
 * @brief ObjectSegmentation::set_image
 * @param idx
 */
void ObjectSegmentation::set_image(int idx)
{
  if (clouds->size()==0 || idx<0 || idx >= int(clouds->size()))
    return;

  image_idx = idx;

  if (masks.size()==clouds->size())
    getMaskedImage(*(*clouds)[image_idx].second, masks[image_idx], image, 0.5);
  else convertImage(*clouds->at(idx).second, image);

  emit new_image(clouds->at(idx).second, image);
  emit update_visualization();
}

/**
 * @brief ObjectSegmentation::segment_image
 * @param x
 * @param y
 */
void ObjectSegmentation::segment_image(int x, int y)
{
  if (image_idx<0 || image_idx>=int(clouds->size()) || labels.size()!=clouds->size())
    return;

  pcl::PointCloud<pcl::PointXYZRGB> &c_cloud = *(*clouds)[image_idx].second;
  const Eigen::Matrix4f &glob_pose = cameras[(*clouds)[image_idx].first];

  // init masks
  if (first_click)
  {
    for (unsigned i=0; i<masks.size(); i++)
      masks[i] = cv::Mat_<unsigned char>::zeros(c_cloud.height, c_cloud.width);
  }

  cv::Mat_<int> &ls = labels[image_idx];
  cv::Mat_<unsigned char> &mask = masks[image_idx];

  if (mask.cols != int(c_cloud.width) || mask.rows!= int(c_cloud.height) || ls.cols != int(c_cloud.width) || ls.rows!= int(c_cloud.height))
    return;

  std::vector<v4r::ClusterNormalsToPlanes::Plane::Ptr> &cls = planes[image_idx];
  int segm_idx = ls(y,x);

  if (segm_idx>=int(cls.size()))
    return;

  unsigned char mark_obj = (first_click || mask(y,x)<128?255:0);

  v4r::ClusterNormalsToPlanes::Plane &plane = *cls[segm_idx];

  // mark object
  for (unsigned i=0; i<plane.size(); i++)
    mask(plane.indices[i]) = mark_obj;

  // propagate to all other images
  if (first_click)
  {
    first_click = false;
    Eigen::Matrix4f inv_pose;
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(c_cloud, plane.indices, centroid);
    v4r::invPose(glob_pose,inv_pose);
    Eigen::Vector3f glob_centroid = (inv_pose * centroid).segment<3>(0);

    for (unsigned i=0; i<masks.size(); i++)
    {
      if (int(i)==image_idx) continue;

      Eigen::Matrix4f &pose = cameras[(*clouds)[i].first];
      Eigen::Vector3f pt = pose.topLeftCorner<3,3>() * glob_centroid + pose.block<3,1>(0,3);

      cv::Point2f im_pt, tmp_pt;

      if (dist_coeffs.empty())
        v4r::projectPointToImage(&pt[0],&intrinsic(0,0), &im_pt.x);
      else v4r::projectPointToImage(&pt[0],&intrinsic(0,0), &dist_coeffs(0,0), &im_pt.x);

      int segm_idx = INT_MAX;

      if (im_pt.x>=0 && im_pt.x<mask.cols && im_pt.y>=0 && im_pt.y<mask.rows)
        segm_idx = labels[i]((int)(im_pt.y+.5), (int)(im_pt.x+.5));


      if (segm_idx>(int)planes[i].size())
      {
        for (int v=-1; v<=1; v++)
        {
          for (int u=-1; u<=1; u++)
          {
            tmp_pt = im_pt + cv::Point2f(u,v);

            if (tmp_pt.x>=0 && tmp_pt.x<mask.cols && tmp_pt.y>=0 && tmp_pt.y<mask.rows)
            {
              segm_idx = labels[i]((int)(tmp_pt.y+.5), (int)(tmp_pt.x+.5));
            }
          }
        }
      }

      if (segm_idx>=(int)planes[i].size()) continue;

      v4r::ClusterNormalsToPlanes::Plane &plane = *planes[i][segm_idx];
      cv::Mat_<unsigned char> &mask = masks[i];

      // mark object
      for (unsigned j=0; j<plane.size(); j++)
         mask(plane.indices[j]) = mark_obj;
    }
  }

  // get masked image
  getMaskedImage(*(*clouds)[image_idx].second, mask, image, 0.5);

  emit new_image((*clouds)[image_idx].second, image);
  emit update_visualization();
}


/**
 * @brief ObjectSegmentation::setData
 * @param _cameras
 * @param _clouds
 */
void ObjectSegmentation::setData(const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &_cameras, const boost::shared_ptr< std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > > &_clouds)
{
  cameras = _cameras;
  clouds = _clouds;

  if(clouds->size()<=0)
    return;

  // segment planes and smooth clusters
  labels.resize(clouds->size());
  masks.resize(clouds->size());
  planes.resize(clouds->size());
  normals.resize(clouds->size());

  v4r::DataMatrix2D<Eigen::Vector3f>::Ptr kp_cloud( new v4r::DataMatrix2D<Eigen::Vector3f>() );
  v4r::DataMatrix2D<Eigen::Vector3f>::Ptr kp_normals( new v4r::DataMatrix2D<Eigen::Vector3f>() );

  for (unsigned i=0; i<clouds->size(); i++)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &c = (*clouds)[i].second;
    cv::Mat_<int> &ls = labels[i];
    ls = cv::Mat_<int>::ones(c->height, c->width)*INT_MAX-1;

    v4r::convertCloud(*c, *kp_cloud);

    if (have_roi && use_roi_segmentation)
      createMaskFromROI(*kp_cloud, masks[i], object_base_transform, bb_min, bb_max, roi_offs);
    else masks[i] = cv::Mat_<unsigned char>::ones(c->height, c->width)*255;

    nest->compute(*kp_cloud, *kp_normals);
    pest->compute(*kp_cloud, *kp_normals, planes[i]);

    normals[i].reset(new pcl::PointCloud<pcl::Normal>());
    v4r::convertNormals(*kp_normals, *normals[i]);

//cv::Mat_<cv::Vec3b> tmp_labels(cv::Mat_<cv::Vec3b>::zeros(c->height, c->width));

    for (unsigned j=0; j<planes[i].size(); j++)
    {
      v4r::ClusterNormalsToPlanes::Plane &plane = *planes[i][j];
//cv::Vec3b col(rand()%255,rand()%255,rand()%255);

      for (unsigned k=0; k<plane.size(); k++)
{
        ls(plane.indices[k]) = j;
//tmp_labels(plane.indices[k]) = col;
}
    }

//cv::imwrite("log/labels.png", tmp_labels);
  }

  // vis. first image
  if (have_roi && use_roi_segmentation)
    first_click = false;
  else first_click = true;

  image_idx = 0;

  if (have_roi && use_roi_segmentation)
    getMaskedImage(*clouds->at(image_idx).second, masks[image_idx], image, 0.5);
  else convertImage(*clouds->at(image_idx).second, image);

  emit new_image(clouds->at(0).second, image);
  emit update_visualization();
}

/**
 * @brief ObjectSegmentation::drawObjectCloud
 */
void ObjectSegmentation::drawObjectCloud()
{
  //createObjectCloud();
  //createObjectCloudFiltered();

  emit update_model_cloud(oc_cloud);
  emit update_visualization();
}




/*********************************** private *******************************************/
/**
 * @brief ObjectSegmentation::run
 * main loop
 */
void ObjectSegmentation::run()
{
  m_run=true;

  switch (cmd)
  {
  case FINISH_OBJECT_MODELLING:
  {
    if (clouds.get()!=0 && clouds->size()>0)
      postProcessingSegmentation(false);

    emit set_object_base_transform(object_base_transform);
    emit finishedObjectSegmentation();
    break;
  }
  case OPTIMIZE_MULTIVIEW:
  {
    if (clouds.get()!=0 && clouds->size()>0)
      postProcessingSegmentation(true);

    emit set_object_base_transform(object_base_transform);
    emit finishedObjectSegmentation();
    break;
  }
  default:
    break;
  }

  cmd = UNDEF;
  m_run=false;
}

/**
 * @brief ObjectSegmentation::postProcessingSegmentation
 */
void ObjectSegmentation::postProcessingSegmentation(bool do_mv)
{
  if (!have_roi)
  {
    detectCoordinateSystem(roi_pose);
  }

  unsigned cnt=0;
  Eigen::Vector3d centroid(Eigen::Vector3d::Zero());
  Eigen::Matrix4f inv_pose, transform_centroid(Eigen::Matrix4f::Identity());

  for (unsigned i=0; i<clouds->size() && i<masks.size(); i++)
  {
    const pcl::PointCloud<pcl::PointXYZRGB> &cloud = *clouds->at(i).second;
    const cv::Mat_<unsigned char> &mask = masks[i];
    v4r::invPose(cameras[clouds->at(i).first]*roi_pose,inv_pose);
    Eigen::Matrix3f R = inv_pose.topLeftCorner<3,3>();
    Eigen::Vector3f t = inv_pose.block<3,1>(0,3);

    for (unsigned j=0; j<cloud.points.size(); j++)
    {
      const Eigen::Vector3f &pt = cloud.points[j].getVector3fMap();
      if (mask(j) > 128 && !std::isnan(pt[0]) && !std::isnan(pt[1]) && !std::isnan(pt[2]) )
      {
        cnt++;
        centroid += (R*pt+t).cast<double>();
      }
    }
  }

  if (cnt>0)
  {
    transform_centroid(0,3) = centroid[0]/(double)cnt;
    transform_centroid(1,3) = centroid[1]/(double)cnt;
  }

  object_base_transform = roi_pose*transform_centroid;

  // use Aitor's mv optimization
  if(use_dense_mv || do_mv)
  {
    emit printStatus("Status: Dense multiview optimization ... Please be patient...");

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_segmented(new pcl::PointCloud<pcl::PointXYZRGB>);
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds_filtered(clouds->size());
    std::vector< pcl::PointCloud<pcl::Normal>::Ptr > normals_segmented;

    inv_poses.resize(clouds->size());
    indices.resize(clouds->size());
    normals_segmented.resize(clouds->size());

    for (unsigned i=0; i<clouds->size() && i<masks.size(); i++)
    {
      indices[i].clear();
      clouds_filtered[i].reset(new pcl::PointCloud<pcl::PointXYZRGB>());
      normals_segmented[i].reset(new pcl::PointCloud<pcl::Normal>());

      const pcl::PointCloud<pcl::PointXYZRGB> &cloud = *clouds->at(i).second;
      const cv::Mat_<unsigned char> &mask = masks[i];

      for (int j=0; j<mask.rows*mask.cols; j++)
        if (mask(j)>128 && !std::isnan(cloud.points[j].x))
          indices[i].push_back(j);

      pcl::copyPointCloud(cloud, indices[i], *cloud_segmented);

      pcl::VoxelGrid<pcl::PointXYZRGB> filter;
      filter.setInputCloud(cloud_segmented);
      filter.setDownsampleAllData(true);
      filter.setLeafSize(vx_size,vx_size,vx_size);
      filter.filter(*clouds_filtered[i]);

      pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> normal_est;
      normal_est.setInputCloud(clouds_filtered[i]);
      normal_est.setRadiusSearch(0.015f);
      normal_est.compute(*normals_segmented[i]);

      v4r::invPose(cameras[clouds->at(i).first]*object_base_transform, inv_poses[i]);
    }

    v4r::Registration::MvLMIcp<pcl::PointXYZRGB> nl_icp;
    nl_icp.setInputClouds(clouds_filtered);
    nl_icp.setPoses(inv_poses);
    nl_icp.setMaxCorrespondenceDistance(max_dist);
    nl_icp.setMaxIterations(max_iterations);
    nl_icp.setDiffType(diff_type);
    //nl_icp.setNormals(normals_segmented);
    //nl_icp.setNormalDot(0.5f);
    nl_icp.compute();

    inv_poses = nl_icp.getFinalPoses();

    Eigen::Matrix4f inv_base, tmp;
    v4r::invPose(object_base_transform, inv_base);

    for (unsigned i=0; i<clouds->size() && i<masks.size(); i++)
    {
      v4r::invPose(inv_poses[i], tmp);
      cameras[clouds->at(i).first] = tmp*inv_base;
    }
  }

  //createObjectCloud();
  createObjectCloudFiltered();
}


/**
 * @brief ObjectSegmentation::createMaskFromROI
 * @param cloud
 * @param mask
 * @param object_base_transform
 * @param bb_min
 * @param bb_max
 * @param roi_offs
 */
void ObjectSegmentation::createMaskFromROI(const v4r::DataMatrix2D<Eigen::Vector3f> &cloud, cv::Mat_<unsigned char> &mask, const Eigen::Matrix4f &object_base_transform, const Eigen::Vector3f &bb_min, const Eigen::Vector3f &bb_max, const double &roi_offs)
{
  Eigen::Vector3f pt;
  Eigen::Matrix4f inv_pose;

  v4r::invPose(object_base_transform,inv_pose);

  Eigen::Matrix3f R = inv_pose.topLeftCorner<3,3>();
  Eigen::Vector3f t = inv_pose.block<3,1>(0,3);

  mask = cv::Mat_<unsigned char>::zeros(cloud.rows, cloud.cols);

  for (unsigned i=0; i<cloud.data.size(); i++)
  {
    const Eigen::Vector3f &pt0 = cloud.data[i];

    if (!std::isnan(pt0[0]) && !std::isnan(pt0[1]) && !std::isnan(pt0[2]))
    {
      pt = R*pt0 + t;

      if (pt[0]>bb_min[0] && pt[0]<bb_max[0] && pt[1]>bb_min[1] && pt[1]<bb_max[1] && pt[2]>roi_offs && pt[2]<bb_max[2])
        mask(i) = 255;
    }
  }
}


/**
 * @brief ObjectSegmentation::createObjectCloud
 */
void ObjectSegmentation::createObjectCloud()
{
  oc_cloud->clear();

  if (clouds->size()==0 || masks.size()!=clouds->size())
    return;

  std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > &ref_clouds = *clouds;

  if (ref_clouds.size()>0)
  {
    Eigen::Matrix4f inv_pose;
    octree.reset(new pcl::octree::OctreePointCloudVoxelCentroid<pcl::PointXYZRGB,pcl::octree::OctreeVoxelCentroidContainerXYZRGB<pcl::PointXYZRGB> >(om_params.vx_size_object));

    for (unsigned i=0; i<ref_clouds.size(); i++)
    {
      v4r::invPose(cameras[ref_clouds[i].first], inv_pose);

      segmentObject(*ref_clouds[i].second, masks[i], *tmp_cloud1);

      pcl::transformPointCloud(*tmp_cloud1, *tmp_cloud2, inv_pose);
      octree->setInputCloud(tmp_cloud2);
      octree->addPointsFromInputCloud();
    }

    octree->getVoxelCentroids(*oc_cloud);
  }
}

/**
 * @brief ObjectSegmentation::createObjectCloudFiltered
 */
void ObjectSegmentation::createObjectCloudFiltered()
{
  oc_cloud->clear();

  if (clouds->size()==0 || masks.size()!=clouds->size())
    return;

  v4r::NguyenNoiseModel<pcl::PointXYZRGB>::Parameter nmparam;
  nmparam.edge_radius_ = om_params.edge_radius_px;
  v4r::NguyenNoiseModel<pcl::PointXYZRGB> nm(nmparam);
  std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > &ref_clouds = *clouds;
  std::vector< std::vector<std::vector<float> > > pt_properties (ref_clouds.size());
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr > ptr_clouds(ref_clouds.size());
  inv_poses.clear();
  indices.clear();
  inv_poses.resize(ref_clouds.size());
  indices.resize(ref_clouds.size());

  if (!ref_clouds.empty())
  {
    for (unsigned i=0; i<ref_clouds.size(); i++)
    {
      v4r::invPose(cameras[ref_clouds[i].first]*object_base_transform, inv_poses[i]);
      ptr_clouds[i] = ref_clouds[i].second;

      nm.setInputCloud(ref_clouds[i].second);
      nm.setInputNormals(normals[i]);
      nm.compute();
      pt_properties[i] = nm.getPointProperties();

      cv::Mat_<unsigned char> &m = masks[i];

      for (int j=0; j<m.rows*m.cols; j++)
        if (m(j)>128)
          indices[i].push_back(j);
    }

    v4r::NMBasedCloudIntegration<pcl::PointXYZRGB>::Parameter nmparam;
    nmparam.octree_resolution_ = om_params.vx_size_object;
    nmparam.edge_radius_px_ = om_params.edge_radius_px;
    nmparam.min_points_per_voxel_ = 1;
    octree_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    v4r::NMBasedCloudIntegration<pcl::PointXYZRGB> nmIntegration(nmparam);
    nmIntegration.setInputClouds(ptr_clouds);
    nmIntegration.setTransformations(inv_poses);
    nmIntegration.setInputNormals(normals);
    nmIntegration.setIndices(indices);
    nmIntegration.setPointProperties(pt_properties);
    nmIntegration.compute(octree_cloud);
    big_normals.reset(new pcl::PointCloud<pcl::Normal>);
    nmIntegration.getOutputNormals(big_normals);

    //test mls
#ifdef ER_TEST_MLS
    pcl::PointCloud<pcl::PointXYZRGB> mls_points;
    pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGB> mls;
    mls.setComputeNormals (true);
    mls.setInputCloud(octree_cloud);
    mls.setPolynomialFit (true);
    mls.setSearchRadius (0.01);
    mls.process(mls_points);
    *octree_cloud=mls_points;
#endif

    Sensor::AlignedPointXYZRGBVector &ref_oc = *oc_cloud;
    pcl::PointCloud<pcl::PointXYZRGB> &ref_occ = *octree_cloud;

    ref_oc.resize(ref_occ.points.size());
    for (unsigned i=0; i<ref_occ.size(); i++)
      ref_oc[i] = ref_occ.points[i];

  }
}

/**
 * @brief ObjectSegmentation::segmentObject
 * @param cloud
 * @param mask
 * @param seg_cloud
 */
void ObjectSegmentation::segmentObject(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const cv::Mat_<unsigned char> &mask, pcl::PointCloud<pcl::PointXYZRGB> &seg_cloud)
{
  seg_cloud.points.clear();

  for (unsigned i=0; i<cloud.points.size(); i++)
  {
    if (mask(i)>128 && !isnan(cloud.points[i].getVector3fMap()))
    {
      seg_cloud.points.push_back(cloud.points[i]);
    }
  }

  seg_cloud.width = seg_cloud.points.size();
  seg_cloud.height = 1;
  seg_cloud.is_dense = true;
}

/**
 * @brief ObjectSegmentation::getInplaneCoordinateSystem
 * @param coeffs
 * @param centroid
 * @param pose
 */
void ObjectSegmentation::getInplaneTransform(const Eigen::Vector3f &pt, const Eigen::Vector3f &normal, Eigen::Matrix4f &pose)
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

//  //transform to camera
//  Eigen::Matrix4f inv;
//  v4r::invPose(pose,inv);
//  pose = inv;
}


/**
 * @brief ObjectSegmentation::convertImage
 * @param cloud
 * @param image
 */
void ObjectSegmentation::convertImage(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, cv::Mat_<cv::Vec3b> &image)
{
  image = cv::Mat_<cv::Vec3b>(cloud.height, cloud.width);

  for (unsigned v = 0; v < cloud.height; v++)
  {
    for (unsigned u = 0; u < cloud.width; u++)
    {
      cv::Vec3b &cv_pt = image.at<cv::Vec3b> (v, u);
      const pcl::PointXYZRGB &pt = cloud(u,v);

      cv_pt[2] = pt.r;
      cv_pt[1] = pt.g;
      cv_pt[0] = pt.b;
    }
  }
}

/**
 * @brief ObjectSegmentation::getMaskedImage
 * @param cloud
 * @param mask
 * @param image
 * @param alpha
 */
void ObjectSegmentation::getMaskedImage(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const cv::Mat_<unsigned char> &mask, cv::Mat_<cv::Vec3b> &image, float alpha)
{
  image = cv::Mat_<cv::Vec3b>(cloud.height, cloud.width);

  for (unsigned v = 0; v < cloud.height; v++)
  {
    for (unsigned u = 0; u < cloud.width; u++)
    {
      cv::Vec3b &cv_pt = image.at<cv::Vec3b> (v, u);
      const pcl::PointXYZRGB &pt = cloud(u,v);

      if (mask(v,u)>128)
      {
        cv_pt[2] = pt.r;
        cv_pt[1] = pt.g;
        cv_pt[0] = pt.b;
      }
      else
      {
        cv_pt[2] = pt.r*0.5;
        cv_pt[1] = pt.g*0.5;
        cv_pt[0] = pt.b*0.5;
      }
    }
  }
}

/**
 * @brief StoreTrackingModel::detectCoordinateSystem
 * @param pose
 */
void ObjectSegmentation::detectCoordinateSystem(Eigen::Matrix4f &pose)
{
  pose.setIdentity();

  pcl::PointXYZRGB pt;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr gcloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::PointCloud<pcl::PointXYZRGB> &gc = *gcloud;
  Eigen::Vector3d centroid(0.,0.,0.);
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

        if (std::isnan(pt1.x) || std::isnan(pt1.y) || std::isnan(pt1.z)) continue;

        if (mask(v,u)<128)
        {
          pt.getVector3fMap() = R*pt1.getVector3fMap() + t;
          gc.points.push_back(pt);
          centroid += pt.getVector3fMap().cast<double>();
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

  centroid /= double(cnt);

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

  if (centroid.dot(pz.cast<double>()) > 0) pz *= -1;
  px = (Eigen::Vector3f(1,0,0).cross(pz)).normalized();
  py = pz.cross(px);

  pose.block<3,1>(0,0) = px;
  pose.block<3,1>(0,1) = py;
  pose.block<3,1>(0,2) = pz;
  pose.block<3,1>(0,3) = centroid.cast<float>();
  pose.block<3,1>(0,3)[2] = -(plane_coeffs[0]*centroid[0]+plane_coeffs[1]*centroid[1]+plane_coeffs[3])/plane_coeffs[2];
}
