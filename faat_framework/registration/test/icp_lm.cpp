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
#include <faat_pcl/registration/lm_icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/icp.h>
#include "pcl/visualization/pcl_visualizer.h"
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/keypoints/uniform_sampling.h>

using namespace pcl;
int
main (int argc, char ** argv)
{

    std::string cloud1, cloud2;
    int icp_iterations_ = 30;
    float max_corresp_dist_ = 0.1f;
    float voxel_grid_size = 0.005f;
    float data_scale = 1.f;
    bool use_point_to_plane = false;
    float voxel_grid_size_c = 0.001f;
    bool use_octree = false;
    float inlier_threshold = 0.005f;

    pcl::console::parse_argument (argc, argv, "-inlier_threshold", inlier_threshold);
    pcl::console::parse_argument (argc, argv, "-use_octree", use_octree);
    pcl::console::parse_argument (argc, argv, "-cloud1", cloud1);
    pcl::console::parse_argument (argc, argv, "-cloud2", cloud2);
    pcl::console::parse_argument (argc, argv, "-icp_iterations", icp_iterations_);
    pcl::console::parse_argument (argc, argv, "-max_corresp_dist", max_corresp_dist_);
    pcl::console::parse_argument (argc, argv, "-vx_size", voxel_grid_size);
    pcl::console::parse_argument (argc, argv, "-data_scale", data_scale);
    pcl::console::parse_argument (argc, argv, "-p2p", use_point_to_plane);

    float ry,rx,rz;
    ry = rx = rz = 0.f;
    pcl::console::parse_argument (argc, argv, "-rx", rx);
    pcl::console::parse_argument (argc, argv, "-ry", ry);
    pcl::console::parse_argument (argc, argv, "-rz", rz);

    typedef pcl::PointXYZ PointType;
    pcl::PointCloud<PointType>::Ptr cloud_11 (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr cloud_22 (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr cloud_1 (new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr cloud_2 (new pcl::PointCloud<PointType>);

    pcl::io::loadPCDFile (cloud1, *cloud_11);
    pcl::io::loadPCDFile (cloud2, *cloud_22);

    {

        if (data_scale != 1.f)
        {
            for (size_t k = 0; k < cloud_11->points.size (); k++)
            {
                cloud_11->points[k].getVector3fMap () *= data_scale;
            }
        }

        pcl::UniformSampling<PointType> grid_;
        grid_.setRadiusSearch(voxel_grid_size_c);
        grid_.setInputCloud (cloud_11);
        pcl::PointCloud<int> keypoints;
        grid_.compute(keypoints);

        pcl::PointIndices ii;
        for(size_t i=0; i < keypoints.points.size(); i++)
            ii.indices.push_back(keypoints.points[i]);
        pcl::copyPointCloud(*cloud_11, ii.indices, *cloud_1);

        /*pcl::VoxelGrid<PointType> grid_;
        grid_.setInputCloud (cloud_11);
        grid_.setLeafSize (voxel_grid_size_c, voxel_grid_size_c, voxel_grid_size_c);
        grid_.setDownsampleAllData (true);
        grid_.filter (*cloud_1);*/

        //cloud_1->points.resize(cloud_1->points.size() / 2.f);
    }

    {
        if (data_scale != 1.f)
        {
            for (size_t k = 0; k < cloud_22->points.size (); k++)
            {
                cloud_22->points[k].getVector3fMap () *= data_scale;
            }
        }

        /*pcl::VoxelGrid<PointType> grid_;
        grid_.setInputCloud (cloud_22);
        grid_.setLeafSize (voxel_grid_size_c, voxel_grid_size_c, voxel_grid_size_c);
        grid_.setDownsampleAllData (true);
        grid_.filter (*cloud_2);*/

        pcl::UniformSampling<PointType> grid_;
        grid_.setRadiusSearch(voxel_grid_size_c);
        grid_.setInputCloud (cloud_22);
        pcl::PointCloud<int> keypoints;
        grid_.compute(keypoints);

        pcl::PointIndices ii;
        for(size_t i=0; i < keypoints.points.size(); i++)
            ii.indices.push_back(keypoints.points[i]);
        pcl::copyPointCloud(*cloud_22, ii.indices, *cloud_2);

    }

    faat_pcl::registration::NonLinearICP icp_nl(voxel_grid_size);
    icp_nl.setInputCloud(cloud_1);
    icp_nl.setTargetCloud(cloud_2);

    if(use_point_to_plane)
    {
        // Create the normal estimation class, and pass the input dataset to it
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud (cloud_2);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
        ne.setSearchMethod (tree);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
        //ne.setKSearch(10);
        ne.setRadiusSearch(0.03f);
        ne.compute (*cloud_normals);
        icp_nl.setTargetNormals(cloud_normals);
    }

    Eigen::Matrix4d trans;
    icp_nl.setInlierThreshold(inlier_threshold);
    icp_nl.setIterations(icp_iterations_);
    icp_nl.setUseOctree(use_octree);
    icp_nl.compute ();
    icp_nl.getFinalTransformation(trans);

    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr Final(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*cloud_1, *Final, trans);

        pcl::visualization::PCLVisualizer vis_("Usual icp..");
        int v1,v2;
        vis_.createViewPort(0,0,0.5,1,v1);
        vis_.createViewPort(0.5,0,1,1,v2);

        {
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler (Final, 255, 0, 0);
            vis_.addPointCloud (Final, handler, "aligned", v2);
        }

        {
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler (cloud_1, 255, 0, 0);
            vis_.addPointCloud (cloud_1, handler, "source", v1);
        }

        {
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler (cloud_2, 255, 255, 0);
            vis_.addPointCloud (cloud_2, handler, "target");
        }

        vis_.spin();
    }
}
