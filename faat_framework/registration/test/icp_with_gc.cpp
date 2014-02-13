/*
 * icp_with_gc.cpp
 *
 *  Created on: Mar 20, 2013
 *      Author: aitor
 */

#include <pcl/console/parse.h>
#include <faat_pcl/utils/filesystem_utils.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <faat_pcl/registration/icp_with_gc.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_estimation_normal_shooting.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <faat_pcl/registration/lm_icp.h>

using namespace pcl;
int
main (int argc, char ** argv)
{
  std::string cloud1, cloud2;
  int icp_iterations_ = 30;
  float max_corresp_dist_ = 0.1f;
  bool use_cg = false;
  bool survival_of_the_fittest = true;
  bool point_to_plane_baseline = false;
  float voxel_grid_size = 0.005f;
  float data_scale = 1.f;
  bool taco_data = false;
  float Z_DIST_0 = 0.5f;
  float Z_DIST_1 = 1.5f;
  float cut_x = 0.3f;
  int mean_k = 50;
  float std_dev = 1.0;

  pcl::console::parse_argument (argc, argv, "-mean_k", mean_k);
  pcl::console::parse_argument (argc, argv, "-std_dev", std_dev);
  pcl::console::parse_argument (argc, argv, "-z_dist_0", Z_DIST_0);
  pcl::console::parse_argument (argc, argv, "-z_dist_1", Z_DIST_1);
  pcl::console::parse_argument (argc, argv, "-taco_data", taco_data);
  pcl::console::parse_argument (argc, argv, "-cut_x", cut_x);

  pcl::console::parse_argument (argc, argv, "-cloud1", cloud1);
  pcl::console::parse_argument (argc, argv, "-cloud2", cloud2);
  pcl::console::parse_argument (argc, argv, "-icp_iterations", icp_iterations_);
  pcl::console::parse_argument (argc, argv, "-max_corresp_dist", max_corresp_dist_);
  pcl::console::parse_argument (argc, argv, "-use_cg", use_cg);
  pcl::console::parse_argument (argc, argv, "-use_best", survival_of_the_fittest);
  pcl::console::parse_argument (argc, argv, "-point_to_plane_baseline", point_to_plane_baseline);
  pcl::console::parse_argument (argc, argv, "-vx_size", voxel_grid_size);
  pcl::console::parse_argument (argc, argv, "-data_scale", data_scale);

  Eigen::Matrix4f taco_total_transform;
  taco_total_transform.setIdentity();

  typedef pcl::PointXYZ PointType;
  pcl::PointCloud<PointType>::Ptr cloud_1 (new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr cloud_2 (new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr cloud_11 (new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr cloud_22 (new pcl::PointCloud<PointType>);

  pcl::io::loadPCDFile (cloud1, *cloud_11);
  pcl::io::loadPCDFile (cloud2, *cloud_22);

  pcl::PointCloud<PointType>::Ptr cloud_1_orig (new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr cloud_2_orig (new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr cloud_2_orig_total (new pcl::PointCloud<PointType>);

  pcl::copyPointCloud(*cloud_11, *cloud_1_orig);
  pcl::copyPointCloud(*cloud_22, *cloud_2_orig);
  pcl::copyPointCloud(*cloud_22, *cloud_2_orig_total);

  if(taco_data)
  {
    Eigen::Matrix3f rz;
    rz = Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitZ());
    Eigen::Matrix4f rotate;
    rotate.setIdentity();
    rotate.block<3,3>(0,0) = rz;
    pcl::transformPointCloud(*cloud_22, *cloud_22, rotate);
    pcl::transformPointCloud(*cloud_2_orig, *cloud_2_orig, rotate);

    taco_total_transform *= rotate;

    {
      pcl::PointCloud<pcl::PointXYZ>::Ptr filtered (new pcl::PointCloud<pcl::PointXYZ>);

      {
        float leaf=0.002f;
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud (cloud_22);
        sor.setLeafSize (leaf,leaf,leaf);
        sor.filter (*filtered);
      }

      pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
      sor.setInputCloud (filtered);
      sor.setMeanK (mean_k);
      sor.setStddevMulThresh (std_dev);
      sor.filter (*cloud_22);
    }
  }

  pcl::visualization::PCLVisualizer vis ("TEST");
  int v1, v2;
  vis.createViewPort (0, 0, 0.5, 1, v1);
  vis.createViewPort (0.5, 0, 1, 1, v2);

  {

    if(data_scale != 1.f)
    {
      for(size_t k=0; k < cloud_11->points.size(); k++)
      {
        cloud_11->points[k].getVector3fMap() *= data_scale;
      }
    }
    pcl::VoxelGrid<PointType> grid_;
    grid_.setInputCloud (cloud_11);
    grid_.setLeafSize (voxel_grid_size, voxel_grid_size, voxel_grid_size);
    grid_.setDownsampleAllData (true);
    grid_.filter (*cloud_1);
  }

  {
    if(data_scale != 1.f)
    {
      for(size_t k=0; k < cloud_22->points.size(); k++)
      {
        cloud_22->points[k].getVector3fMap() *= data_scale;
      }
    }

    pcl::VoxelGrid<PointType> grid_;
    grid_.setInputCloud (cloud_22);
    grid_.setLeafSize (voxel_grid_size, voxel_grid_size, voxel_grid_size);
    grid_.setDownsampleAllData (true);
    grid_.filter (*cloud_2);
  }

  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass_;
    pass_.setFilterLimits (Z_DIST_0, Z_DIST_1);
    pass_.setFilterFieldName ("z");
    pass_.setInputCloud (cloud_1);
    pass_.setKeepOrganized (false);
    pass_.filter (*filtered);

    pass_.setFilterLimits (-cut_x, cut_x);
    pass_.setFilterFieldName ("x");
    pass_.setInputCloud (filtered);
    pass_.setKeepOrganized (false);
    pass_.filter (*cloud_1);
  }

  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass_;
    pass_.setFilterLimits (Z_DIST_0, Z_DIST_1);
    pass_.setFilterFieldName ("z");
    pass_.setInputCloud (cloud_2);
    pass_.setKeepOrganized (false);
    pass_.filter (*filtered);

    pass_.setFilterLimits (-cut_x, cut_x);
    pass_.setFilterFieldName ("x");
    pass_.setInputCloud (filtered);
    pass_.setKeepOrganized (false);
    pass_.filter (*cloud_2);
  }

  {
    pcl::visualization::PointCloudColorHandlerCustom<PointType> handler (cloud_1, 255, 0, 0);
    vis.addPointCloud (cloud_1, handler, "cloud_1", v1);
  }

  {
    pcl::visualization::PointCloudColorHandlerCustom<PointType> handler (cloud_2, 0, 255, 0);
    vis.addPointCloud (cloud_2, handler, "cloud_2", v1);
  }

  vis.spin();

  typedef pcl::PointNormal PointTInternal;

  //compute normals for the clouds...
  pcl::PointCloud<PointTInternal>::Ptr cloud_1_internal (new pcl::PointCloud<PointTInternal>);
  pcl::PointCloud<PointTInternal>::Ptr cloud_2_internal (new pcl::PointCloud<PointTInternal>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_1 (new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_2 (new pcl::PointCloud<pcl::Normal>);

  {
    pcl::NormalEstimation<PointType, pcl::Normal> ne;
    ne.setInputCloud (cloud_1);
    pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType> ());
    ne.setSearchMethod (tree);
    ne.setRadiusSearch (0.03);
    ne.compute (*cloud_normals_1);
  }

  {
    pcl::NormalEstimation<PointType, pcl::Normal> ne;
    ne.setInputCloud (cloud_2);
    pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType> ());
    ne.setSearchMethod (tree);
    ne.setRadiusSearch (0.03);
    ne.compute (*cloud_normals_2);
  }

  pcl::copyPointCloud (*cloud_1, *cloud_1_internal);
  pcl::copyPointCloud (*cloud_2, *cloud_2_internal);
  pcl::copyPointCloud (*cloud_normals_1, *cloud_1_internal);
  pcl::copyPointCloud (*cloud_normals_2, *cloud_2_internal);

  {
    vis.addPointCloudNormals<PointTInternal, PointTInternal> (cloud_1_internal, cloud_1_internal, 1, 0.02, "cloud_1_normals", v1);
    vis.addPointCloudNormals<PointTInternal, PointTInternal> (cloud_2_internal, cloud_2_internal, 1, 0.02, "cloud_2_normals", v1);
  }

  typename pcl::registration::TransformationEstimationPointToPlaneLLS<PointTInternal, PointTInternal>::Ptr
                                                                                                           trans_lls (
                                                                                                                      new pcl::registration::TransformationEstimationPointToPlaneLLS<
                                                                                                                          PointTInternal,
                                                                                                                          PointTInternal>);

  typename pcl::registration::CorrespondenceRejectorSampleConsensus<PointTInternal>::Ptr
                                                                                         rej (
                                                                                              new pcl::registration::CorrespondenceRejectorSampleConsensus<
                                                                                                  PointTInternal> ());

  rej->setMaximumIterations (1000);
  rej->setInlierThreshold (0.005f);

  pcl::registration::CorrespondenceEstimationNormalShooting<PointTInternal, PointTInternal, PointTInternal>::Ptr
   cens (new pcl::registration::CorrespondenceEstimationNormalShooting<PointTInternal,PointTInternal,PointTInternal>);

  typename pcl::PointCloud<PointTInternal>::Ptr pp_out (new pcl::PointCloud<PointTInternal>);
  Eigen::Matrix4f final_trans;
  if(use_cg)
  {
    faat_pcl::IterativeClosestPointWithGC<PointTInternal, PointTInternal> icp;
    icp.setMaximumIterations (icp_iterations_);
    icp.setMaxCorrespondenceDistance (max_corresp_dist_);
    //icp.setTransformationEstimation (trans_lls);
    icp.setTransformationEpsilon (0.000001 * 0.000001);
    icp.setMinNumCorrespondences(5);
    //icp.setUseReciprocalCorrespondences(true);
    //icp.addCorrespondenceRejector (rej);
    icp.setUseCG(use_cg);
    icp.setSurvivalOfTheFittest(survival_of_the_fittest);
    //icp.setCorrespondenceEstimation (cens);

    pcl::registration::DefaultConvergenceCriteria<float>::Ptr convergence_criteria;
    convergence_criteria = icp.getConvergeCriteria ();
    convergence_criteria->setAbsoluteMSE (1e-9);
    convergence_criteria->setMaximumIterationsSimilarTransforms (15);
    convergence_criteria->setFailureAfterMaximumIterations (false);

    icp.setInputTarget (cloud_1_internal);
    rej->setInputTarget (cloud_1_internal);
    cens->setInputTarget (cloud_1_internal);

    icp.setInputSource (cloud_2_internal);
    rej->setInputSource (cloud_2_internal);
    cens->setSourceNormals (cloud_2_internal);
    cens->setInputSource (cloud_2_internal);
    icp.align (*pp_out);
    final_trans = icp.getFinalTransformation();
    pcl::registration::DefaultConvergenceCriteria<float>::ConvergenceState conv_state;
    conv_state = convergence_criteria->getConvergenceState ();

    if (conv_state != pcl::registration::DefaultConvergenceCriteria<float>::CONVERGENCE_CRITERIA_ITERATIONS && conv_state
        != pcl::registration::DefaultConvergenceCriteria<float>::CONVERGENCE_CRITERIA_NOT_CONVERGED && conv_state
        != pcl::registration::DefaultConvergenceCriteria<float>::CONVERGENCE_CRITERIA_NO_CORRESPONDENCES)
    {
      PCL_INFO("icp converged... conv state: %d\n", conv_state);
    }
    else
    {
      PCL_WARN("icp did not converge... conv state: %d\n",conv_state);
      //pp_out = cloud_2_internal;
    }

    taco_total_transform = final_trans * taco_total_transform;
  }
  else
  {
    pcl::IterativeClosestPoint<PointTInternal, PointTInternal> icp;
    icp.setMaximumIterations (icp_iterations_);
    icp.setMaxCorrespondenceDistance (max_corresp_dist_);

    /*if(point_to_plane_baseline)
      icp.setTransformationEstimation (trans_lls);*/

    icp.setTransformationEpsilon (0);
    icp.addCorrespondenceRejector (rej);
    //icp.setCorrespondenceEstimation (cens);

    pcl::registration::DefaultConvergenceCriteria<float>::Ptr convergence_criteria;
    convergence_criteria = icp.getConvergeCriteria ();
    convergence_criteria->setAbsoluteMSE (0);
    convergence_criteria->setMaximumIterationsSimilarTransforms (5);
    convergence_criteria->setFailureAfterMaximumIterations (false);

    icp.setInputTarget (cloud_1_internal);
    rej->setInputTarget (cloud_1_internal);
    cens->setInputTarget (cloud_1_internal);

    icp.setInputSource (cloud_2_internal);
    rej->setInputSource (cloud_2_internal);
    cens->setSourceNormals (cloud_2_internal);
    cens->setInputSource (cloud_2_internal);
    icp.align (*pp_out);
    final_trans = icp.getFinalTransformation();

    pcl::registration::DefaultConvergenceCriteria<float>::ConvergenceState conv_state;
    conv_state = convergence_criteria->getConvergenceState ();

    if (conv_state != pcl::registration::DefaultConvergenceCriteria<float>::CONVERGENCE_CRITERIA_ITERATIONS && conv_state
        != pcl::registration::DefaultConvergenceCriteria<float>::CONVERGENCE_CRITERIA_NOT_CONVERGED && conv_state
        != pcl::registration::DefaultConvergenceCriteria<float>::CONVERGENCE_CRITERIA_NO_CORRESPONDENCES)
    {
      PCL_INFO("icp converged... conv state: %d\n", conv_state);
    }
    else
    {
      PCL_WARN("icp did not converge... conv state: %d\n",conv_state);
      //pp_out = cloud_2_internal;
    }
  }

  {

    {
      pcl::visualization::PointCloudColorHandlerCustom<PointType> handler (cloud_1_orig, 255, 0, 0);
      vis.addPointCloud (cloud_1_orig, handler, "cloud_11", v2);
    }

    {
      pcl::transformPointCloud(*cloud_2_orig, *cloud_2_orig, final_trans);
      pcl::transformPointCloud(*cloud_2, *cloud_2, final_trans);
      pcl::visualization::PointCloudColorHandlerCustom<PointType> handler (cloud_2_orig, 0, 255, 0);
      vis.addPointCloud (cloud_2_orig, handler, "aligned", v2);
    }
  }

  vis.spin ();

  vis.removeAllPointClouds(v2);

  {
    pcl::visualization::PointCloudColorHandlerCustom<PointType> handler (cloud_1_orig, 255, 0, 0);
    vis.addPointCloud (cloud_1_orig, handler, "cloud_11", v2);
  }

  {
    std::cout << taco_total_transform << std::endl;
    Eigen::AngleAxisf aa(taco_total_transform.block<3,3>(0,0));
    Eigen::Quaternionf q(aa);
    std::cout << taco_total_transform.block<3,1>(0,3) << std::endl;
    q.normalize();
    std::cout << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    pcl::transformPointCloud(*cloud_2_orig_total, *cloud_2_orig_total, taco_total_transform);
    pcl::visualization::PointCloudColorHandlerCustom<PointType> handler (cloud_2_orig_total, 0, 255, 0);
    vis.addPointCloud (cloud_2_orig_total, handler, "aligned", v2);
  }

  vis.spin ();

  //do ICP lm for fine registration in the end...
  {
    faat_pcl::registration::NonLinearICP icp_nl;
    icp_nl.setInputCloud(cloud_1);
    icp_nl.setTargetCloud(cloud_2);
    icp_nl.compute ();
    Eigen::Matrix4d final;
    icp_nl.getFinalTransformation(final);

    /*typename pcl::registration::CorrespondenceRejectorSampleConsensus<PointType>::Ptr
                                                                                           rej (
                                                                                                new pcl::registration::CorrespondenceRejectorSampleConsensus<
                                                                                                PointType> ());

    rej->setMaximumIterations (1000);
    rej->setInlierThreshold (0.005f);

    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaximumIterations (50);
    icp.setMaxCorrespondenceDistance (max_corresp_dist_);
    icp.setTransformationEpsilon (0);
    icp.addCorrespondenceRejector (rej);

    pcl::registration::DefaultConvergenceCriteria<float>::Ptr convergence_criteria;
    convergence_criteria = icp.getConvergeCriteria ();
    convergence_criteria->setAbsoluteMSE (0);
    convergence_criteria->setMaximumIterationsSimilarTransforms (5);
    convergence_criteria->setFailureAfterMaximumIterations (false);

    icp.setInputTarget (cloud_1);
    rej->setInputTarget (cloud_1);

    icp.setInputSource (cloud_2);
    rej->setInputSource (cloud_2);

    pcl::PointCloud<PointType>::Ptr out (new pcl::PointCloud<PointType>);
    icp.align (*out);
    Eigen::Matrix4f final;
    final = icp.getFinalTransformation();

    std::cout << final << std::endl;

    vis.removeAllPointClouds(v2);

    {
      pcl::visualization::PointCloudColorHandlerCustom<PointType> handler (cloud_1_orig, 255, 0, 0);
      vis.addPointCloud (cloud_1_orig, handler, "cloud_11", v2);
    }

    {
      pcl::transformPointCloud(*cloud_2_orig, *cloud_2_orig, final);
      pcl::visualization::PointCloudColorHandlerCustom<PointType> handler (cloud_2_orig, 0, 255, 0);
      vis.addPointCloud (cloud_2_orig, handler, "aligned", v2);
    }

    vis.spin ();*/

  }

}
