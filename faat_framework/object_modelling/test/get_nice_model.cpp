/*
 * GO3D.cpp
 *
 *  Created on: Oct 24, 2013
 *      Author: aitor
 */

#include <faat_pcl/utils/filesystem_utils.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <pcl/common/transforms.h>
#include <fstream>
#include <pcl/visualization/pcl_visualizer.h>
#include <faat_pcl/utils/noise_models.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/octree/octree_pointcloud_pointvector.h>
#include <pcl/octree/impl/octree_iterator.hpp>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/surface/mls.h>
#include <pcl/apps/dominant_plane_segmentation.h>

namespace bf = boost::filesystem;

struct IndexPoint
{
    int idx;
};

template<typename PointT>
void getAveragedCloudFromOctree(typename pcl::PointCloud<PointT>::Ptr & octree_big_cloud,
                                typename pcl::PointCloud<PointT>::Ptr & filtered_big_cloud,
                                float octree_resolution,
                                bool median = false)
{
    typename pcl::octree::OctreePointCloudPointVector<PointT> octree(octree_resolution);
    octree.setInputCloud(octree_big_cloud);
    octree.addPointsFromInputCloud();

    unsigned int leaf_node_counter = 0;
    typename pcl::octree::OctreePointCloudPointVector<PointT>::LeafNodeIterator it2;
    const typename pcl::octree::OctreePointCloudPointVector<PointT>::LeafNodeIterator it2_end = octree.leaf_end();

    filtered_big_cloud->points.resize(octree_big_cloud->points.size());

    int kept = 0;
    for (it2 = octree.leaf_begin(); it2 != it2_end; ++it2)
    {
        ++leaf_node_counter;
        pcl::octree::OctreeContainerPointIndices& container = it2.getLeafContainer();
        // add points from leaf node to indexVector
        std::vector<int> indexVector;
        container.getPointIndices (indexVector);
        //std::cout << "Number of points in this leaf:" << indexVector.size() << std::endl;

        /*if(indexVector.size() <= 1)
      continue;*/

        PointT p;
        p.getVector3fMap() = Eigen::Vector3f::Zero();
        p.getNormalVector3fMap() = Eigen::Vector3f::Zero();
        std::vector<int> rs, gs, bs;
        int r,g,b;
        r = g = b = 0;
        int used = 0;
        for(size_t k=0; k < indexVector.size(); k++)
        {
            Eigen::Vector3f normal = octree_big_cloud->points[indexVector[k]].getNormalVector3fMap();
            normal.normalize();
            p.getVector3fMap() = p.getVector3fMap() +  octree_big_cloud->points[indexVector[k]].getVector3fMap();
            p.getNormalVector3fMap() = p.getNormalVector3fMap() + normal;
            r += octree_big_cloud->points[indexVector[k]].r;
            g += octree_big_cloud->points[indexVector[k]].g;
            b += octree_big_cloud->points[indexVector[k]].b;
            rs.push_back(octree_big_cloud->points[indexVector[k]].r);
            gs.push_back(octree_big_cloud->points[indexVector[k]].g);
            bs.push_back(octree_big_cloud->points[indexVector[k]].b);
        }

        p.getVector3fMap() = p.getVector3fMap() / static_cast<int>(indexVector.size());
        p.getNormalVector3fMap() = p.getNormalVector3fMap() / static_cast<int>(indexVector.size());
        p.getNormalVector3fMap()[3] = 0;
        p.r = r / static_cast<int>(indexVector.size());
        p.g = g / static_cast<int>(indexVector.size());
        p.b = b / static_cast<int>(indexVector.size());

        if(median)
        {
            std::sort(rs.begin(), rs.end());
            std::sort(bs.begin(), bs.end());
            std::sort(gs.begin(), gs.end());
            int size = rs.size() / 2;
            p.r = rs[size];
            p.g = gs[size];
            p.b = bs[size];
        }

        filtered_big_cloud->points[kept] = p;
        kept++;
    }

    filtered_big_cloud->points.resize(kept);
    filtered_big_cloud->width = kept;
    filtered_big_cloud->height = 1;
}

void getCloudFromOctree(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & octree_big_cloud,
                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr & filtered_big_cloud,
                        std::vector<float> & weights_big_cloud,
                        std::vector<float> & weights_filtered,
                        float octree_resolution,
                        float w_t)
{
    pcl::octree::OctreePointCloudPointVector<pcl::PointXYZRGB> octree(octree_resolution);
    octree.setInputCloud(octree_big_cloud);
    octree.addPointsFromInputCloud();

    unsigned int leaf_node_counter = 0;
    pcl::octree::OctreePointCloudPointVector<pcl::PointXYZRGB>::LeafNodeIterator it2;
    const pcl::octree::OctreePointCloudPointVector<pcl::PointXYZRGB>::LeafNodeIterator it2_end = octree.leaf_end();

    /*weights_filtered = weights_big_cloud;
  for (it2 = octree.leaf_begin(); it2 != it2_end; ++it2)
  {
    pcl::octree::OctreeContainerPointIndices& container = it2.getLeafContainer();
    std::vector<int> indexVector;
    container.getPointIndices (indexVector);
    for(size_t k=0; k < indexVector.size(); k++)
    {
      weights_filtered[indexVector[k]] *= 1.f - 1.f / static_cast<float>(indexVector.size());
    }
  }*/

    filtered_big_cloud->points.resize(octree_big_cloud->points.size());
    weights_filtered.resize(filtered_big_cloud->points.size());

    int kept = 0;
    for (it2 = octree.leaf_begin(); it2 != it2_end; ++it2)
    {
        ++leaf_node_counter;
        pcl::octree::OctreeContainerPointIndices& container = it2.getLeafContainer();
        // add points from leaf node to indexVector
        std::vector<int> indexVector;
        container.getPointIndices (indexVector);
        //std::cout << "Number of points in this leaf:" << indexVector.size() << std::endl;

        /*if(indexVector.size() < 2)
      continue;*/

        float max_weight = -1.f;
        int max_index = -1;
        for(size_t k=0; k < indexVector.size(); k++)
        {
            if(weights_big_cloud[indexVector[k]] > max_weight)
            {
                max_index = indexVector[k];
                max_weight = weights_big_cloud[indexVector[k]];
            }
        }

        if(max_weight < w_t)
            continue;

        if(max_index != -1)
        {
            filtered_big_cloud->points[kept] = octree_big_cloud->points[max_index];
            weights_filtered[kept] = weights_big_cloud[max_index];
            kept++;
        }
    }

    weights_filtered.resize(kept);
    filtered_big_cloud->points.resize(kept);
    filtered_big_cloud->width = kept;
    filtered_big_cloud->height = 1;
}

void transformNormals(pcl::PointCloud<pcl::Normal>::Ptr & normals_cloud,
                      pcl::PointCloud<pcl::Normal>::Ptr & normals_aligned,
                      Eigen::Matrix4f & transform)
{
    normals_aligned.reset (new pcl::PointCloud<pcl::Normal>);
    normals_aligned->points.resize (normals_cloud->points.size ());
    normals_aligned->width = normals_cloud->width;
    normals_aligned->height = normals_cloud->height;
    for (size_t k = 0; k < normals_cloud->points.size (); k++)
    {
        Eigen::Vector3f nt (normals_cloud->points[k].normal_x, normals_cloud->points[k].normal_y, normals_cloud->points[k].normal_z);
        normals_aligned->points[k].normal_x = static_cast<float> (transform (0, 0) * nt[0] + transform (0, 1) * nt[1]
                                                                  + transform (0, 2) * nt[2]);
        normals_aligned->points[k].normal_y = static_cast<float> (transform (1, 0) * nt[0] + transform (1, 1) * nt[1]
                                                                  + transform (1, 2) * nt[2]);
        normals_aligned->points[k].normal_z = static_cast<float> (transform (2, 0) * nt[0] + transform (2, 1) * nt[1]
                                                                  + transform (2, 2) * nt[2]);
    }
}

//./bin/get_nice_model -input_dir /media/DATA/models/coffee_container/seq1_aligned/ -organized_normals 1 -w_t 0.9 -lateral_sigma 0.001 -octree_resolution 0.0015 -mls_radius 0.002 -visualize 1 -structure_for_recognition /media/DATA/models/recognition_structure/coffee_container.pcd -save_model_to /media/DATA/models/nice_models/coffee_container.pcd

int
main (int argc, char ** argv)
{
    bool organized_normals = true;
    float w_t = 0.75f;
    bool depth_edges = true;
    float max_angle = 60.f;
    float lateral_sigma = 0.002f;
    int max_vis = std::numeric_limits<int>::max();
    float octree_resolution = 0.0015f;
    float mls_radius_ = 0.003f;
    std::string input_dir_;
    bool median = true;
    bool mls = true;
    std::string save_model_to = "model.pcd";
    bool visualize = true;
    std::string structure_for_recognition = "";
    bool bring_to_plane = true;
    bool pose_inverse = false;

    pcl::console::parse_argument (argc, argv, "-pose_inverse", pose_inverse);
    pcl::console::parse_argument (argc, argv, "-input_dir", input_dir_);
    pcl::console::parse_argument (argc, argv, "-organized_normals", organized_normals);
    pcl::console::parse_argument (argc, argv, "-w_t", w_t);
    pcl::console::parse_argument (argc, argv, "-depth_edges", depth_edges);
    pcl::console::parse_argument (argc, argv, "-max_angle", max_angle);
    pcl::console::parse_argument (argc, argv, "-lateral_sigma", lateral_sigma);
    pcl::console::parse_argument (argc, argv, "-max_vis", max_vis);
    pcl::console::parse_argument (argc, argv, "-octree_resolution", octree_resolution);
    pcl::console::parse_argument (argc, argv, "-mls_radius", mls_radius_);
    pcl::console::parse_argument (argc, argv, "-median", median);
    pcl::console::parse_argument (argc, argv, "-save_model_to", save_model_to);
    pcl::console::parse_argument (argc, argv, "-mls", mls);
    pcl::console::parse_argument (argc, argv, "-visualize", visualize);
    pcl::console::parse_argument (argc, argv, "-structure_for_recognition", structure_for_recognition);
    pcl::console::parse_argument (argc, argv, "-bring_to_plane", bring_to_plane);

    bf::path input = input_dir_;
    std::vector<std::string> scene_files;
    std::vector<std::string> indices_files;
    std::vector<std::string> transformation_files;
    std::string pattern_scenes = ".*cloud_.*.pcd";
    std::string pattern_indices = ".*object_indices_.*.pcd";
    std::string transformations_pattern = ".*pose.*.txt";

    faat_pcl::utils::getFilesInDirectory(input, scene_files, pattern_scenes);
    faat_pcl::utils::getFilesInDirectory(input, indices_files, pattern_indices);
    faat_pcl::utils::getFilesInDirectory(input, transformation_files, transformations_pattern);

    std::cout << "Number of clouds:" << scene_files.size() << std::endl;
    std::cout << "Number of models:" << indices_files.size() << std::endl;
    std::cout << "Number of transformations:" << transformation_files.size() << std::endl;

    std::sort(scene_files.begin(), scene_files.end());
    std::sort(indices_files.begin(), indices_files.end());
    std::sort(transformation_files.begin(), transformation_files.end());

    std::vector < Eigen::Matrix4f > transforms_to_global;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr> aligned_clouds;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> original_clouds;
    std::vector<std::vector<float> > weights_;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr big_cloud_not_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr big_cloud_from_transforms_(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::Normal>::Ptr big_cloud_normals_from_transforms_(new pcl::PointCloud<pcl::Normal>);

    Eigen::Vector4f table_plane;

    for(size_t i=0; i < std::min(scene_files.size(), static_cast<size_t>(max_vis)); i++)
    {

        std::cout << scene_files[i] << " " << transformation_files[i] << std::endl;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene(new pcl::PointCloud<pcl::PointXYZRGB>);

        {
            std::stringstream load;
            load << input_dir_ << "/" << scene_files[i];
            pcl::io::loadPCDFile(load.str(), *scene);
            aligned_clouds.push_back(scene);
            original_clouds.push_back(scene);
        }

        {
            Eigen::Matrix4f trans;
            std::stringstream load;
            load << input_dir_ << "/" << transformation_files[i];
            faat_pcl::utils::readMatrixFromFile(load.str(), trans);
            if(pose_inverse)
            {
                trans = trans.inverse().eval();
            }
            transforms_to_global.push_back(trans);
        }

        pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
        if(organized_normals)
        {
            std::cout << "Organized normals" << std::endl;
            pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
            ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
            ne.setMaxDepthChangeFactor (0.02f);
            ne.setNormalSmoothingSize (20.0f);
            ne.setBorderPolicy (pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal>::BORDER_POLICY_MIRROR);
            ne.setInputCloud (scene);
            ne.compute (*normal_cloud);
        }
        else
        {
            std::cout << "Not organized normals" << std::endl;
            pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
            ne.setInputCloud (scene);
            pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
            ne.setSearchMethod (tree);
            ne.setRadiusSearch (0.02);
            ne.compute (*normal_cloud);
        }

        if(i == 0 && bring_to_plane)
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr points (new pcl::PointCloud<pcl::PointXYZRGB>());
            pcl::PassThrough<pcl::PointXYZRGB> pass_;
            pass_.setFilterLimits (0.f, 1);
            pass_.setFilterFieldName ("z");
            pass_.setInputCloud (scene);
            pass_.setKeepOrganized (true);
            pass_.filter (*points);

            //compute table plane
            pcl::apps::DominantPlaneSegmentation<pcl::PointXYZRGB> dps;
            dps.setInputCloud(points);
            dps.setDownsamplingSize(0.01f);
            dps.compute_table_plane();
            dps.getTableCoefficients(table_plane);
            table_plane = transforms_to_global[0] * table_plane;
        }

        faat_pcl::utils::noise_models::NguyenNoiseModel<pcl::PointXYZRGB> nm;
        nm.setInputCloud(scene);
        nm.setInputNormals(normal_cloud);
        nm.setLateralSigma(lateral_sigma);
        nm.setMaxAngle(max_angle);
        nm.setUseDepthEdges(depth_edges);
        nm.compute();
        std::vector<float> weights;
        nm.getWeights(weights);

        pcl::PointIndices obj_indices;
        if(indices_files.size() == scene_files.size())
        {
            pcl::PointCloud<IndexPoint> obj_indices_cloud;
            std::stringstream oi_file;
            oi_file << input_dir_ << "/" << indices_files[i];
            pcl::io::loadPCDFile (oi_file.str(), obj_indices_cloud);
            obj_indices.indices.resize(obj_indices_cloud.points.size());
            for(size_t kk=0; kk < obj_indices_cloud.points.size(); kk++)
            {
                obj_indices.indices[kk] = obj_indices_cloud.points[kk].idx;
            }
        }
        else
        {
            obj_indices.indices.resize(scene->points.size());
            for(size_t kk=0; kk < scene->points.size(); kk++)
            {
                obj_indices.indices[kk] = kk;
            }
        }

        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_trans(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::copyPointCloud(*scene, obj_indices, *scene_trans);
            pcl::transformPointCloud(*scene_trans, *scene_trans, transforms_to_global[i]);
            *big_cloud_not_filtered += *scene_trans;
        }

        int valid = 0;
        for(size_t k=0; k < obj_indices.indices.size(); k++)
        {
            if(weights[obj_indices.indices[k]] > w_t)
            {
                obj_indices.indices[valid] = obj_indices.indices[k];
                valid++;
            }
        }

        obj_indices.indices.resize(valid);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered;
        std::vector<int> valid_indices;
        nm.getFilteredCloudRemovingPoints(filtered, w_t, valid_indices);

        pcl::PointCloud<pcl::Normal>::Ptr valid_normals(new pcl::PointCloud<pcl::Normal>);
        pcl::copyPointCloud(*normal_cloud, obj_indices, *valid_normals);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_trans2(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_trans(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::copyPointCloud(*scene, obj_indices, *scene_trans2);
        pcl::transformPointCloud(*scene_trans2, *scene_trans, transforms_to_global[i]);
        aligned_clouds[i] = scene_trans;

        std::vector<float> object_indices_weights;
        for(size_t kk=0; kk < obj_indices.indices.size(); kk++)
        {
            object_indices_weights.push_back(weights[obj_indices.indices[kk]]);
        }
        weights_.push_back(object_indices_weights);

        /*pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_trans2(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_trans(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::copyPointCloud(*filtered, obj_indices, *scene_trans2);
        pcl::transformPointCloud(*scene_trans2, *scene_trans, transforms_to_global[i]);*/

        pcl::PointCloud<pcl::Normal>::Ptr transformed_normals (new pcl::PointCloud<pcl::Normal>);
        transformNormals(valid_normals, transformed_normals, transforms_to_global[i]);

        *big_cloud_from_transforms_ += *scene_trans;
        *big_cloud_normals_from_transforms_ += *transformed_normals;
    }

    pcl::visualization::PCLVisualizer vis ("registered cloud");
    int v1, v2, v3;
    vis.createViewPort (0, 0, 0.33, 1, v1);
    vis.createViewPort (0.33, 0, 0.66, 1, v2);
    vis.createViewPort (0.66, 0, 1, 1, v3);
    vis.setBackgroundColor(0,0,0);
    vis.addCoordinateSystem(0.1f);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (big_cloud_not_filtered);
    vis.addPointCloud (big_cloud_not_filtered, handler, "big", v1);
    vis.addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (big_cloud_from_transforms_, big_cloud_normals_from_transforms_, 10, 0.03, "normals_big", v1);

    if(visualize)
        vis.spin ();
    else
        vis.spinOnce(100, true);

    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr octree_big_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    std::vector<float> weights_big_cloud;
    for(size_t i=0; i < aligned_clouds.size(); i++)
    {
        //*octree_big_cloud += *aligned_clouds[i];
        weights_big_cloud.insert(weights_big_cloud.end(), weights_[i].begin(), weights_[i].end());
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_big_cloud (new pcl::PointCloud<pcl::PointXYZRGB>(*big_cloud_from_transforms_));

    //std::vector<float> weights_filtered = weights_big_cloud;
    //getCloudFromOctree(octree_big_cloud, filtered_big_cloud, weights_big_cloud, weights_filtered, octree_resolution, w_t);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered (new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::Normal>::Ptr filtered_normals (new pcl::PointCloud<pcl::Normal>);

    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> ror(true);
    ror.setMeanK(10);
    ror.setStddevMulThresh(3.f);
    ror.setInputCloud(filtered_big_cloud);
    ror.setNegative(true);
    ror.filter(*filtered);

    pcl::PointIndices::Ptr removed(new pcl::PointIndices);
    ror.getRemovedIndices(*removed);

    pcl::copyPointCloud(*filtered_big_cloud, *removed, *filtered);
    pcl::copyPointCloud(*big_cloud_normals_from_transforms_, *removed, *filtered_normals);

    std::cout << removed->indices.size() << " " << filtered->points.size() << " " << filtered_normals->points.size() << std::endl;

    /*pcl::copyPointCloud(*filtered_big_cloud, *filtered);
    pcl::copyPointCloud(*big_cloud_normals_from_transforms_, *filtered_normals);*/

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr filtered_with_normals_oriented (new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    pcl::copyPointCloud(*filtered, *filtered_with_normals_oriented);
    pcl::copyPointCloud(*filtered_normals, *filtered_with_normals_oriented);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr filtered_with_normals (new pcl::PointCloud<pcl::PointXYZRGBNormal>());

    if(mls)
    {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr mls_points(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
        pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGBNormal> mls;
        mls.setComputeNormals (true);
        // Set parameters
        mls.setInputCloud (filtered);
        mls.setPolynomialFit (true);
        mls.setSearchMethod (tree);
        mls.setSearchRadius (mls_radius_);
        // Reconstruct
        mls.process (*mls_points);
        std::cout << filtered->points.size() << " " << mls_points->points.size() << std::endl;

        //pcl::copyPointCloud(*filtered_normals, *filtered_with_normals);

        pcl::NormalEstimationOMP<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> ne;
        ne.setRadiusSearch(0.02f);
        ne.setInputCloud (mls_points);
        ne.compute (*mls_points);

        pcl::copyPointCloud(*mls_points, *filtered_with_normals);

        //assert(mls_points->points.size() == filtered_normals->points.size());

        pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGBNormal> octree(0.002);
        octree.setInputCloud(filtered_with_normals_oriented);
        octree.addPointsFromInputCloud();

        for(size_t i=0; i < filtered_with_normals->points.size(); i++)
        {
            std::vector<int> indices;
            std::vector<float> distances;
            if (octree.nearestKSearch (filtered_with_normals->points[i], 1, indices, distances) > 0)
            {
                //check normal dotproduct
                const Eigen::Vector3f & normal_oriented = filtered_with_normals_oriented->points[indices[0]].getNormalVector3fMap();
                const Eigen::Vector3f & normal_mls = filtered_with_normals->points[i].getNormalVector3fMap();
                if(normal_oriented.dot(normal_mls) < 0)
                {
                    filtered_with_normals->points[i].getNormalVector3fMap() = filtered_with_normals->points[i].getNormalVector3fMap() * -1.f;
                }
            }
        }
    }
    else
    {
        filtered_with_normals = filtered_with_normals_oriented;
    }


    {
        //std::cout << "Number of leaves:" << leaf_node_counter << " " << kept << " " <<  " " << filtered->points.size() << " " << octree_big_cloud->points.size() << std::endl;
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> handler (filtered_with_normals);
        vis.addPointCloud<pcl::PointXYZRGBNormal> (filtered_with_normals, handler, "big_filtered", v2);
        vis.addPointCloudNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> (filtered_with_normals, filtered_with_normals, 10, 0.01, "normals_v2", v2);

        if(visualize)
            vis.spin ();
        else
            vis.spinOnce(100, true);
    }

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr filtered_big_cloud2 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    {
        getAveragedCloudFromOctree<pcl::PointXYZRGBNormal>(filtered_with_normals, filtered_big_cloud2, octree_resolution, median);

        if(bring_to_plane)
        {
            //center cloud
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*filtered_big_cloud2, centroid);
            pcl::demeanPointCloud(*filtered_big_cloud2, centroid, *filtered_big_cloud2);

            Eigen::Matrix4f center_transform;
            center_transform.setIdentity();
            center_transform(0,3) = -centroid[0];
            center_transform(1,3) = -centroid[1];
            center_transform(2,3) = -centroid[2];

            std::cout << table_plane << std::endl;
            Eigen::Vector3f normal_plane;
            normal_plane[0] = table_plane[0];
            normal_plane[1] = table_plane[1];
            normal_plane[2] = table_plane[2];
            normal_plane.normalize();
            std::cout << normal_plane << std::endl;

            Eigen::Matrix4f transform;
            transform.setIdentity();

            //ATTENTION: Make sure that the determinant is 1 (meaning that we have an invertible rotation matrix, otherwise weird flips...)
            transform.block<3,1>(0,2) = normal_plane;
            transform.block<3,1>(0,1) = Eigen::Vector3f::UnitZ().cross(transform.block<3,1>(0,2));
            transform.block<3,1>(0,0) = transform.block<3,1>(0,1).cross(transform.block<3,1>(0,2));
            transform.block<3,1>(0,1).normalize();
            transform.block<3,1>(0,0).normalize();

            assert(transform.determinant() > 0);

            transform = transform.inverse().eval();

            pcl::transformPointCloudWithNormals(*filtered_big_cloud2, *filtered_big_cloud2, transform);

            Eigen::Vector4f min_pt, max_pt;
            pcl::getMinMax3D(*filtered_big_cloud2, min_pt, max_pt);

            Eigen::Matrix4f translation;
            translation.setIdentity();
            translation(2,3) = -min_pt[2];
            pcl::transformPointCloudWithNormals(*filtered_big_cloud2, *filtered_big_cloud2, translation);

            for(size_t i=0; i < transforms_to_global.size(); i++)
            {
                transforms_to_global[i] = translation * transform * center_transform * transforms_to_global[i];
            }
        }

        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> handler (filtered_big_cloud2);
        vis.addPointCloud (filtered_big_cloud2, handler, "big_filtered_averaged", v3);
        vis.addPointCloudNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> (filtered_big_cloud2, filtered_big_cloud2, 10, 0.01, "normals", v3);
        if(visualize)
            vis.spin ();
        else
            vis.spinOnce(100, true);
    }

    /*if(visualize)
    {
        pcl::visualization::PCLVisualizer vis("checking");
        int v1,v2, v3;
        vis.createViewPort(0,0,0.33,1,v1);
        vis.createViewPort(0.33,0,0.66,1,v2);
        vis.createViewPort(0.66,0,1,1,v3);
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> handler (filtered_big_cloud2);
        vis.addPointCloud (filtered_big_cloud2, handler, "big_filtered_averaged", v1);

        for(size_t i=0; i < transforms_to_global.size(); i++)
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene (new pcl::PointCloud<pcl::PointXYZRGB>(*original_clouds[i]));
            std::stringstream name;
            name << "cloud_" << i;
            pcl::transformPointCloud(*scene, *scene, transforms_to_global[i]);
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (scene);
            vis.addPointCloud (scene, handler, name.str(), v2);
        }

        vis.spin();
    }*/

    for(size_t i=0; i < filtered_big_cloud2->points.size(); i++)
    {
        Eigen::Vector3f normal = filtered_big_cloud2->points[i].getNormalVector3fMap();
        normal.normalize();
        filtered_big_cloud2->points[i].getNormalVector3fMap() = normal;
        (filtered_big_cloud2->points[i].getNormalVector4fMap())[3] = 0.f;
    }

    pcl::io::savePCDFileBinary(save_model_to, *filtered_big_cloud2);

    if(structure_for_recognition.compare("") != 0)
    {
        PCL_WARN("creating recognition structure");
        std::cout << structure_for_recognition << std::endl;
        //move object clouds and object indices
        //save new transforms

        bf::path aligned_output = structure_for_recognition;
        if(!bf::exists(aligned_output))
        {
            bf::create_directory(aligned_output);
        }

        for(size_t k=0; k < original_clouds.size(); k++)
        {
            {
                std::stringstream temp;
                temp << structure_for_recognition << "/cloud_";
                temp << setw( 8 ) << setfill( '0' ) << static_cast<int>(k) << ".pcd";
                std::string scene_name;
                temp >> scene_name;
                std::cout << scene_name << std::endl;
                pcl::io::savePCDFileBinary(scene_name, *original_clouds[k]);
            }

            //write pose
            {
                std::stringstream temp;
                temp << structure_for_recognition << "/pose_";
                temp << setw( 8 ) << setfill( '0' ) << static_cast<int>(k) << ".txt";
                std::string scene_name;
                temp >> scene_name;
                std::cout << scene_name << std::endl;
                faat_pcl::utils::writeMatrixToFile(scene_name, transforms_to_global[k]);
            }

            //write object indices
            {

                pcl::PointCloud<IndexPoint> obj_indices_cloud;
                std::stringstream oi_file;
                oi_file << input_dir_ << "/" << indices_files[k];
                pcl::io::loadPCDFile (oi_file.str(), obj_indices_cloud);
                std::stringstream temp;
                temp << structure_for_recognition << "/object_indices_";
                temp << setw( 8 ) << setfill( '0' ) << static_cast<int>(k) << ".pcd";
                std::string scene_name;
                temp >> scene_name;
                std::cout << scene_name << std::endl;
                pcl::io::savePCDFileBinary(scene_name, obj_indices_cloud);
            }
        }
    }
}

POINT_CLOUD_REGISTER_POINT_STRUCT (IndexPoint,
                                   (int, idx, idx)
                                   )
