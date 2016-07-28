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
#include "MultiSession.h"

#include <cmath>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/filters/voxel_grid.h>
#include <v4r/common/noise_models.h>
#include <v4r/registration/noise_model_based_cloud_integration.h>
#include <v4r/registration/MvLMIcp.h>
#include <v4r/registration/MultiSessionModelling.h>
#include <v4r/registration/FeatureBasedRegistration.h>
#include <v4r/registration/StablePlanesRegistration.h>
#include <v4r/common/convertImage.h>
#include <pcl/features/normal_3d_omp.h>
#include <v4r/keypoints/impl/invPose.hpp>
#include <v4r/keypoints/impl/PoseIO.hpp>
#include <pcl/io/pcd_io.h>
//#include <pcl/visualization/pcl_visualizer.h>
#endif

using namespace std;

/**
 * @brief MultiSession::MultiSession
 */
MultiSession::MultiSession()
 : cmd(UNDEF), m_run(false)
{
  oc_cloud.reset(new Sensor::AlignedPointXYZRGBVector());
  clouds.reset(new std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> >() );
  use_stable_planes_ = true;
  use_features_ = true;

  vx_size = 0.005;
  max_dist = 0.01f;
  max_iterations = 10;
  diff_type = 2;
}

/**
 * @brief MultiSession::~MultiSession
 */
MultiSession::~MultiSession()
{

}



/******************************** public *******************************/

/**
 * @brief MultiSession::start
 * @param cam_id
 */
void MultiSession::start()
{
  QThread::start();
}

/**
 * @brief MultiSession::stop
 */
void MultiSession::stop()
{
  if(m_run)
  {
    m_run = false;
    this->wait();
  }
}

/**
 * @brief MultiSession::isRunning
 * @return
 */
bool MultiSession::isRunning()
{
  return m_run;
}

/**
 * MultiSession::addSequences
 * _cameras transforms a point from object coordinates to camera coordinates
 * _clouds point cloud and an index to the corresponding camera
 * _object_indices
 */
void MultiSession::addSequences(const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &_cameras, const boost::shared_ptr< std::vector<std::pair<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > > &_clouds, const std::vector<std::vector<int> > &_object_indices, const Eigen::Matrix4f &_object_base_transform)
{
  if (_clouds.get()==0 || _clouds->size()==0)
  {
    emit printStatus("No segmented point clouds available!");
    return;
  }

  std::pair<int, int> this_session;
  this_session.first = sessions_clouds_.size();
  this_session.second = this_session.first + _clouds->size() - 1;


  for(size_t i=0; i < _clouds->size(); i++)
  {
      Eigen::Matrix4f inv = (_cameras[_clouds->at(i).first]*_object_base_transform).inverse();
      sessions_cloud_poses_.push_back(inv);
      sessions_clouds_.push_back(_clouds->at(i).second);
      sessions_cloud_indices_.push_back(_object_indices[i]);
  }

  if (this_session.second >= this_session.first)
    session_ranges_.push_back(this_session);

  emit printStatus("Status: Added a sequence");
}

/**
 * @brief MultiSession::clear
 */
void MultiSession::clear()
{
  session_ranges_.clear();
  sessions_cloud_poses_.clear();
  sessions_cloud_indices_.clear();
  sessions_clouds_.clear();
}



/**
 * @brief MultiSession::alignSequences
 */
void MultiSession::alignSequences()
{
  emit printStatus("Status: Aligning multiple sequences... Please be patient...");
  cmd = MULTI_SESSION_ALIGNMENT;
  start();
}

void MultiSession::optimizeSequences()
{
  cmd = MULTI_SESSION_MULTI_VIEW;
  start();
}


/**
 * @brief ObjectSegmentation::object_modelling_parameter_changed
 * @param param
 */
void MultiSession::object_modelling_parameter_changed(const ObjectModelling& param)
{
  om_params = param;
}

/**
 * @brief MultiSession::savePointClouds
 * @param _folder
 * @param _modelname
 * @return
 * TODO: same code in ObjectSegmentation -> move...
 */
bool MultiSession::savePointClouds(const std::string &_folder, const std::string &_modelname)
{
  if (octree_cloud.get()==0 || big_normals.get()==0 || clouds.get()==0 || clouds->empty() ||
      octree_cloud->empty() || octree_cloud->points.size()!=big_normals->points.size() || clouds->size()!=sessions_cloud_indices_.size())
    return false;

  char filename[PATH_MAX];
  boost::filesystem::create_directories(_folder + "/models/" + _modelname + "/views" );

  // create model cloud with normals and save it
  pcl::PointCloud<pcl::PointXYZRGBNormal> ncloud;
  pcl::concatenateFields(*octree_cloud, *big_normals, ncloud);
  pcl::io::savePCDFileBinary(_folder + "/models/" + _modelname + "/3D_model.pcd", ncloud);

  // store data
  cv::Mat image;

  std::string cloud_names = _folder + "/models/" + _modelname + "/views/cloud_%08d.pcd";
  std::string image_names = _folder + "/models/" + _modelname + "/views/image_%08d.jpg";
  std::string pose_names = _folder + "/models/" + _modelname + "/views/pose_%08d.txt";
  std::string mask_names = _folder + "/models/" + _modelname + "/views/mask_%08d.png";
  std::string idx_names = _folder + "/models/" + _modelname + "/views/object_indices_%08d.txt";

  for (unsigned i=0; i<clouds->size(); i++)
  {
    if (sessions_cloud_indices_[i].empty()) continue;

    // store indices
    snprintf(filename, PATH_MAX, idx_names.c_str(), i);
    std::ofstream mask_f (filename);
    for(unsigned j=0; j < sessions_cloud_indices_[i].size(); j++)
        mask_f << sessions_cloud_indices_[i][j] << std::endl;
    mask_f.close();

    // store cloud
    snprintf(filename, PATH_MAX, cloud_names.c_str(), i);
    pcl::io::savePCDFileBinary(filename, *clouds->at(i).second);

    // store image
    v4r::convertImage(*clouds->at(i).second, image);
    snprintf(filename, PATH_MAX, image_names.c_str(), i);
    cv::imwrite(filename, image);

    // store poses
    snprintf(filename, PATH_MAX, pose_names.c_str(), i);
    v4r::writePose(filename, std::string(), output_poses[i]);

    // store masks
    snprintf(filename, PATH_MAX, mask_names.c_str(), i);
    cv::imwrite(filename, masks[i]);
  }

  return true;
}






/*********************************** private *******************************************/
/**
 * @brief MultiSession::run
 * main loop
 */
void MultiSession::run()
{
  m_run=true;

  switch (cmd)
  {
  case MULTI_SESSION_ALIGNMENT:
  {
    if (session_ranges_.size()>0)
    {
      for(size_t k=0; k < session_ranges_.size(); k++)
      {
        std::cout << session_ranges_[k].first << " up to " << session_ranges_[k].second << std::endl;
      }

      //compute normals
      normals.resize(sessions_clouds_.size());

      for(size_t i=0; i < sessions_clouds_.size(); i++)
      {
        normals[i].reset(new pcl::PointCloud<pcl::Normal>);
        pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
        ne.setRadiusSearch(0.01f);
        ne.setInputCloud (sessions_clouds_[i]);
        ne.compute (*normals[i]);
      }

      //instantiate stuff
      v4r::Registration::MultiSessionModelling<pcl::PointXYZRGB> msm;
      msm.setInputData(sessions_clouds_, sessions_cloud_poses_, sessions_cloud_indices_, session_ranges_);
      msm.setInputNormals(normals);

      //define registration algorithms
      if(use_features_)
      {
        boost::shared_ptr< v4r::Registration::FeatureBasedRegistration<pcl::PointXYZRGB> > fbr;
        fbr.reset(new v4r::Registration::FeatureBasedRegistration<pcl::PointXYZRGB>);

        //TODO: extract parameters
        fbr->setDoCG(true);
        fbr->setGCThreshold(15);
        fbr->setInlierThreshold(0.015);

        boost::shared_ptr< v4r::Registration::PartialModelRegistrationBase<pcl::PointXYZRGB > > cast_alg;
        cast_alg = boost::static_pointer_cast< v4r::Registration::PartialModelRegistrationBase<pcl::PointXYZRGB > > (fbr);

        msm.addRegAlgorithm(cast_alg);
      }

      if(use_stable_planes_)
      {
        boost::shared_ptr< v4r::Registration::StablePlanesRegistration<pcl::PointXYZRGB> > fbr;
        fbr.reset(new v4r::Registration::StablePlanesRegistration<pcl::PointXYZRGB>);
        boost::shared_ptr< v4r::Registration::PartialModelRegistrationBase<pcl::PointXYZRGB > > cast_alg;
        cast_alg = boost::static_pointer_cast< v4r::Registration::PartialModelRegistrationBase<pcl::PointXYZRGB > > (fbr);

        msm.addRegAlgorithm(cast_alg);
      }

      msm.compute();

      //std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > output_poses;
      msm.getOutputPoses(output_poses);

      cameras.resize(output_poses.size());
      clouds->resize(output_poses.size());
      masks.resize(output_poses.size());

      for (unsigned i=0; i<cameras.size(); i++)
      {
        v4r::invPose(output_poses[i], cameras[i]);
        clouds->at(i) = make_pair(i,sessions_clouds_[i]);
        createMask(sessions_cloud_indices_[i], masks[i], clouds->at(i).second->width, clouds->at(i).second->height);
      }

      /*pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

    for(size_t i=0; i < output_poses.size(); i++)
    {

        std::cout << output_poses[i] << std::endl;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans (new pcl::PointCloud<pcl::PointXYZRGB> ());
        pcl::copyPointCloud(*sessions_clouds_[i], sessions_cloud_indices_[i], *trans);
        pcl::transformPointCloud(*trans, *trans, output_poses[i]);
        *merged_cloud += *trans;
    }

    pcl::visualization::PCLVisualizer vis("merged model");
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler(merged_cloud);
    vis.addPointCloud<pcl::PointXYZRGB>(merged_cloud, handler, "merged_cloud");
    vis.spin();*/

      createObjectCloudFiltered();

      emit update_model_cloud(oc_cloud);
      emit update_visualization();

      emit printStatus("Status: Finished the alignment of multiple sequences!");
      emit finishedAlignment(true);
    }
    else
    {
      emit printStatus("Status: No data available!");

      emit finishedAlignment(false);
    }

    break;
  }

  case MULTI_SESSION_MULTI_VIEW:
    optimizeDenseMultiview();
    break;

  default:
    break;
  }

  cmd = UNDEF;
  m_run=false;
}

/**
 * @brief MultiSession::createMask
 * @param indices
 * @param mask
 * @param width
 * @param height
 */
void MultiSession::createMask(const std::vector<int> &indices, cv::Mat_<unsigned char> &mask, int width, int height)
{
  mask = cv::Mat_<unsigned char>::zeros(height,width);

  for (unsigned i=0; i<indices.size(); i++)
    mask(indices[i]) = 255;
}

/**
 * @brief MultiSession::optimizeDenseMultiview
 */
void MultiSession::optimizeDenseMultiview()
{
  emit printStatus("Status: Dense multiview optimization ... Please be patient...");

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_segmented(new pcl::PointCloud<pcl::PointXYZRGB>);
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds_filtered(sessions_clouds_.size());

  for (unsigned i=0; i<sessions_clouds_.size(); i++)
  {
    clouds_filtered[i].reset(new pcl::PointCloud<pcl::PointXYZRGB>());

    pcl::copyPointCloud(*sessions_clouds_[i], sessions_cloud_indices_[i], *cloud_segmented);

    pcl::VoxelGrid<pcl::PointXYZRGB> filter;
    filter.setInputCloud(cloud_segmented);
    filter.setDownsampleAllData(true);
    filter.setLeafSize(vx_size,vx_size,vx_size);
    filter.filter(*clouds_filtered[i]);
  }

  v4r::Registration::MvLMIcp<pcl::PointXYZRGB> nl_icp;
  nl_icp.setInputClouds(clouds_filtered);
  nl_icp.setPoses(output_poses);
  nl_icp.setMaxCorrespondenceDistance(max_dist);
  nl_icp.setMaxIterations(max_iterations);
  nl_icp.setDiffType(diff_type);
  nl_icp.compute();

  output_poses = nl_icp.getFinalPoses();

  for (unsigned i=0; i<output_poses.size(); i++)
  {
    v4r::invPose(output_poses[i], cameras[i]);
  }

  createObjectCloudFiltered();

  emit update_model_cloud(oc_cloud);
  emit update_visualization();
}


/**
 * @brief MultiSession::createObjectCloudFiltered
 * TODO: same code is in ObjectSegmentation -> move to somewhere
 */
void MultiSession::createObjectCloudFiltered()
{
  oc_cloud->clear();

  if (clouds->size()==0 || masks.size()!=clouds->size())
    return;

  v4r::NguyenNoiseModel<pcl::PointXYZRGB>::Parameter nmparam;
  nmparam.edge_radius_ = om_params.edge_radius_px;
  v4r::NguyenNoiseModel<pcl::PointXYZRGB> nm(nmparam);
  std::vector< std::vector<std::vector<float> > > pt_properties (sessions_clouds_.size());

  if (!sessions_clouds_.empty())
  {
    for (unsigned i=0; i<sessions_clouds_.size(); i++)
    {
      nm.setInputCloud(sessions_clouds_[i]);
      nm.setInputNormals(normals[i]);
      nm.compute();
      pt_properties[i] = nm.getPointProperties();
    }

    v4r::NMBasedCloudIntegration<pcl::PointXYZRGB>::Parameter nmparam;
    nmparam.octree_resolution_ = om_params.vx_size_object;
    nmparam.edge_radius_px_ = om_params.edge_radius_px;
    nmparam.min_points_per_voxel_ = 1;
    octree_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    big_normals.reset(new pcl::PointCloud<pcl::Normal>);
    v4r::NMBasedCloudIntegration<pcl::PointXYZRGB> nmIntegration(nmparam);
    nmIntegration.setInputClouds(sessions_clouds_);
    nmIntegration.setPointProperties(pt_properties);
    nmIntegration.setTransformations(output_poses);
    nmIntegration.setInputNormals(normals);
    nmIntegration.setIndices( sessions_cloud_indices_ );
    nmIntegration.compute(octree_cloud);
    nmIntegration.getOutputNormals(big_normals);

    Sensor::AlignedPointXYZRGBVector &ref_oc = *oc_cloud;
    pcl::PointCloud<pcl::PointXYZRGB> &ref_occ = *octree_cloud;

    ref_oc.resize(ref_occ.points.size());
    for (unsigned i=0; i<ref_occ.size(); i++)
      ref_oc[i] = ref_occ.points[i];
  }
}
