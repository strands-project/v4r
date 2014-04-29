#include <faat_pcl/object_modelling/get_nice_model.h>
#include <pcl/io/pcd_io.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <faat_pcl/utils/filesystem_utils.h>
#include <faat_pcl/utils/noise_models.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/passthrough.h>
#include <pcl/apps/dominant_plane_segmentation.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/surface/mls.h>

namespace bf = boost::filesystem;

struct IndexPoint
{
    int idx;
};

template<typename ScanPointT, typename ModelPointT>
void
faat_pcl::modelling::NiceModelFromSequence<ScanPointT, ModelPointT>::computeFromInputClouds()
{
    big_cloud_from_transforms_.reset(new pcl::PointCloud<ScanPointT>);
    big_cloud_normals_from_transforms_.reset(new pcl::PointCloud<pcl::Normal>);

    for(size_t i=0; i < original_clouds_.size(); i++)
    {

        aligned_clouds_.push_back(original_clouds_[i]);

        pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
        if(organized_normals_)
        {
            std::cout << "Organized normals" << std::endl;
            pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
            ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
            ne.setMaxDepthChangeFactor (0.02f);
            ne.setNormalSmoothingSize (20.0f);
            ne.setBorderPolicy (pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal>::BORDER_POLICY_MIRROR);
            ne.setInputCloud (original_clouds_[i]);
            ne.compute (*normal_cloud);
        }
        else
        {
            std::cout << "Not organized normals" << std::endl;
            pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
            ne.setInputCloud (original_clouds_[i]);
            pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
            ne.setSearchMethod (tree);
            ne.setRadiusSearch (0.02);
            ne.compute (*normal_cloud);
        }

        faat_pcl::utils::noise_models::NguyenNoiseModel<pcl::PointXYZRGB> nm;
        nm.setInputCloud(original_clouds_[i]);
        nm.setInputNormals(normal_cloud);
        nm.setLateralSigma(lateral_sigma_);
        nm.setMaxAngle(60.f);
        nm.setUseDepthEdges(true);
        nm.compute();
        std::vector<float> weights;
        nm.getWeights(weights);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered;
        std::vector<int> valid_indices;
        nm.getFilteredCloudRemovingPoints(filtered, w_t, valid_indices);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_trans(new pcl::PointCloud<pcl::PointXYZRGB>);

        std::cout << transforms_to_global_[i] << std::endl;
        pcl::transformPointCloud(*filtered, *scene_trans, transforms_to_global_[i]);
        aligned_clouds_[i] = scene_trans;

        weights_.push_back(weights);

        pcl::PointCloud<pcl::Normal>::Ptr valid_normals (new pcl::PointCloud<pcl::Normal>);
        pcl::copyPointCloud(*normal_cloud, valid_indices, *valid_normals);

        pcl::PointCloud<pcl::Normal>::Ptr transformed_normals (new pcl::PointCloud<pcl::Normal>);
        transformNormals(valid_normals, transformed_normals, transforms_to_global_[i]);

        *big_cloud_from_transforms_ += *scene_trans;
        *big_cloud_normals_from_transforms_ += *transformed_normals;
    }

    std::vector< int > index;
    pcl::removeNaNFromPointCloud(*big_cloud_from_transforms_,*big_cloud_from_transforms_, index);
    for(size_t i=0; i < index.size(); i++)
    {
        big_cloud_normals_from_transforms_->points[i] = big_cloud_normals_from_transforms_->points[index[i]];
    }

    big_cloud_normals_from_transforms_->points.resize(index.size());

    std::cout << "get_nice_model --- going to visualize" << std::endl;
    pcl::visualization::PCLVisualizer vis ("get_nice_model");
    int v1, v2, v3;
    vis.createViewPort (0, 0, 0.33, 1, v1);
    vis.createViewPort (0.33, 0, 0.66, 1, v2);
    vis.createViewPort (0.66, 0, 1, 1, v3);
    vis.setBackgroundColor(0,0,0);
    vis.addCoordinateSystem(0.1f);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (big_cloud_from_transforms_);
    vis.addPointCloud (big_cloud_from_transforms_, handler, "big", v1);
    //vis.addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (big_cloud_from_transforms_, big_cloud_normals_from_transforms_, 10, 0.01, "normals_big", v1);

    if(visualize_)
        vis.spin ();
    else
        vis.spinOnce(100, true);

    //std::vector< int > index;
    //pcl::removeNaNFromPointCloud(*filtered_with_normals,*filtered_with_normals, index);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_big_cloud (new pcl::PointCloud<pcl::PointXYZRGB>(*big_cloud_from_transforms_));

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered (new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> ror(true);
    ror.setMeanK(10);
    ror.setStddevMulThresh(2.f);
    ror.setInputCloud(filtered_big_cloud);
    ror.setNegative(true);
    ror.filter(*filtered);

    pcl::PointIndices::Ptr removed(new pcl::PointIndices);
    ror.getRemovedIndices(*removed);

    pcl::PointCloud<pcl::Normal>::Ptr filtered_normals (new pcl::PointCloud<pcl::Normal>);

    pcl::copyPointCloud(*filtered_big_cloud, *removed, *filtered);
    pcl::copyPointCloud(*big_cloud_normals_from_transforms_, *removed, *filtered_normals);

    std::cout << removed->indices.size() << " " << filtered->points.size() << " " << filtered_normals->points.size() << std::endl;

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr filtered_with_normals_oriented (new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    pcl::copyPointCloud(*filtered, *filtered_with_normals_oriented);
    pcl::copyPointCloud(*filtered_normals, *filtered_with_normals_oriented);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr filtered_with_normals (new pcl::PointCloud<pcl::PointXYZRGBNormal>());

    if(mls_)
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
        pcl::copyPointCloud(*mls_points, *filtered_with_normals);

        pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGBNormal> octree(0.002);
        octree.setInputCloud(filtered_with_normals_oriented);
        octree.addPointsFromInputCloud();

        for(size_t i=0; i < filtered_with_normals->points.size(); i++)
        {
            std::vector<int> indices;
            std::vector<float> distances;

            if(pcl_isnan(filtered_with_normals->points[i].z))
                continue;
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
        vis.addPointCloud (filtered_with_normals, handler, "big_filtered", v2);
        //vis.addPointCloudNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> (filtered_with_normals, filtered_with_normals, 10, 0.01, "normals_v2", v2);

        if(visualize_)
            vis.spin ();
        else
            vis.spinOnce(100, true);
    }

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr filtered_big_cloud2 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    {
        octree_resolution_ = 0.002;
        getAveragedCloudFromOctree<pcl::PointXYZRGBNormal>(filtered_with_normals, filtered_big_cloud2, octree_resolution_, median_);

        for(size_t i=0; i < filtered_big_cloud2->points.size(); i++)
        {
            Eigen::Vector3f normal = filtered_big_cloud2->points[i].getNormalVector3fMap();
            normal.normalize();
            filtered_big_cloud2->points[i].getNormalVector3fMap() = normal;
            (filtered_big_cloud2->points[i].getNormalVector4fMap())[3] = 0.f;
        }

        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> handler (filtered_big_cloud2);
        vis.addPointCloud (filtered_big_cloud2, handler, "big_filtered_averaged", v3);
        //vis.addPointCloudNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> (filtered_big_cloud2, filtered_big_cloud2, 10, 0.01, "normals", v3);
        if(visualize_)
            vis.spin ();
        else
            vis.spinOnce(100, true);

        model_cloud_ = filtered_big_cloud2;
    }
}

template<typename ScanPointT, typename ModelPointT>
void
faat_pcl::modelling::NiceModelFromSequence<ScanPointT, ModelPointT>::readSequence()
{
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

    big_cloud_from_transforms_.reset(new pcl::PointCloud<ScanPointT>);
    big_cloud_normals_from_transforms_.reset(new pcl::PointCloud<pcl::Normal>);

    for(size_t i=0; i < scene_files.size(); i++)
    {

        std::cout << scene_files[i] << " " << transformation_files[i] << std::endl;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene(new pcl::PointCloud<pcl::PointXYZRGB>);

        {
            std::stringstream load;
            load << input_dir_ << "/" << scene_files[i];
            pcl::io::loadPCDFile(load.str(), *scene);
            aligned_clouds_.push_back(scene);
            original_clouds_.push_back(scene);
        }

        {
            Eigen::Matrix4f trans;
            std::stringstream load;
            load << input_dir_ << "/" << transformation_files[i];
            faat_pcl::utils::readMatrixFromFile(load.str(), trans);
            transforms_to_global_.push_back(trans);
        }

        pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
        if(organized_normals_)
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

        if(i == 0)
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
            dps.getTableCoefficients(table_plane_);
            table_plane_ = transforms_to_global_[0] * table_plane_;
        }

        faat_pcl::utils::noise_models::NguyenNoiseModel<pcl::PointXYZRGB> nm;
        nm.setInputCloud(scene);
        nm.setInputNormals(normal_cloud);
        nm.setLateralSigma(lateral_sigma_);
        nm.setMaxAngle(60.f);
        nm.setUseDepthEdges(true);
        nm.compute();
        std::vector<float> weights;
        nm.getWeights(weights);

        pcl::PointIndices obj_indices;
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

            original_indices_.push_back(obj_indices.indices);
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
        pcl::transformPointCloud(*scene_trans2, *scene_trans, transforms_to_global_[i]);
        aligned_clouds_[i] = scene_trans;

        std::vector<float> object_indices_weights;
        for(size_t kk=0; kk < obj_indices.indices.size(); kk++)
        {
            object_indices_weights.push_back(weights[obj_indices.indices[kk]]);
        }
        weights_.push_back(object_indices_weights);

        pcl::PointCloud<pcl::Normal>::Ptr transformed_normals (new pcl::PointCloud<pcl::Normal>);
        transformNormals(valid_normals, transformed_normals, transforms_to_global_[i]);

        *big_cloud_from_transforms_ += *scene_trans;
        *big_cloud_normals_from_transforms_ += *transformed_normals;
    }
}

template<typename ScanPointT, typename ModelPointT>
void
faat_pcl::modelling::NiceModelFromSequence<ScanPointT, ModelPointT>::compute()
{
    pcl::visualization::PCLVisualizer vis ("registered cloud");
    int v1, v2, v3;
    vis.createViewPort (0, 0, 0.33, 1, v1);
    vis.createViewPort (0.33, 0, 0.66, 1, v2);
    vis.createViewPort (0.66, 0, 1, 1, v3);
    vis.setBackgroundColor(0,0,0);
    vis.addCoordinateSystem(0.1f);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (big_cloud_from_transforms_);
    vis.addPointCloud (big_cloud_from_transforms_, handler, "big", v1);
    vis.addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (big_cloud_from_transforms_, big_cloud_normals_from_transforms_, 10, 0.01, "normals_big", v1);

    if(visualize_)
        vis.spin ();
    else
        vis.spinOnce(100, true);

    std::vector<float> weights_big_cloud;
    for(size_t i=0; i < aligned_clouds_.size(); i++)
    {
        weights_big_cloud.insert(weights_big_cloud.end(), weights_[i].begin(), weights_[i].end());
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_big_cloud (new pcl::PointCloud<pcl::PointXYZRGB>(*big_cloud_from_transforms_));

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered (new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> ror(true);
    ror.setMeanK(10);
    ror.setStddevMulThresh(3.f);
    ror.setInputCloud(filtered_big_cloud);
    ror.setNegative(true);
    ror.filter(*filtered);

    pcl::PointIndices::Ptr removed(new pcl::PointIndices);
    ror.getRemovedIndices(*removed);

    pcl::PointCloud<pcl::Normal>::Ptr filtered_normals (new pcl::PointCloud<pcl::Normal>);

    pcl::copyPointCloud(*filtered_big_cloud, *removed, *filtered);
    pcl::copyPointCloud(*big_cloud_normals_from_transforms_, *removed, *filtered_normals);

    std::cout << removed->indices.size() << " " << filtered->points.size() << " " << filtered_normals->points.size() << std::endl;

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr filtered_with_normals_oriented (new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    pcl::copyPointCloud(*filtered, *filtered_with_normals_oriented);
    pcl::copyPointCloud(*filtered_normals, *filtered_with_normals_oriented);

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr filtered_with_normals (new pcl::PointCloud<pcl::PointXYZRGBNormal>());

    if(mls_)
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
        pcl::copyPointCloud(*mls_points, *filtered_with_normals);

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


    {
        //std::cout << "Number of leaves:" << leaf_node_counter << " " << kept << " " <<  " " << filtered->points.size() << " " << octree_big_cloud->points.size() << std::endl;
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> handler (filtered_with_normals);
        vis.addPointCloud (filtered_with_normals, handler, "big_filtered", v2);
        vis.addPointCloudNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> (filtered_with_normals, filtered_with_normals, 10, 0.01, "normals_v2", v2);

        if(visualize_)
            vis.spin ();
        else
            vis.spinOnce(100, true);
    }

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr filtered_big_cloud2 (new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    {
        getAveragedCloudFromOctree<pcl::PointXYZRGBNormal>(filtered_with_normals, filtered_big_cloud2, octree_resolution_, median_);

        for(size_t i=0; i < filtered_big_cloud2->points.size(); i++)
        {
            Eigen::Vector3f normal = filtered_big_cloud2->points[i].getNormalVector3fMap();
            normal.normalize();
            filtered_big_cloud2->points[i].getNormalVector3fMap() = normal;
            (filtered_big_cloud2->points[i].getNormalVector4fMap())[3] = 0.f;
        }

        //center cloud
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*filtered_big_cloud2, centroid);
        pcl::demeanPointCloud(*filtered_big_cloud2, centroid, *filtered_big_cloud2);

        Eigen::Matrix4f center_transform;
        center_transform.setIdentity();
        center_transform(0,3) = -centroid[0];
        center_transform(1,3) = -centroid[1];
        center_transform(2,3) = -centroid[2];

        std::cout << table_plane_ << std::endl;
        Eigen::Vector3f normal_plane;
        normal_plane[0] = table_plane_[0];
        normal_plane[1] = table_plane_[1];
        normal_plane[2] = table_plane_[2];
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

        for(size_t i=0; i < transforms_to_global_.size(); i++)
        {
            transforms_to_global_[i] = translation * transform * center_transform * transforms_to_global_[i];
        }

        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> handler (filtered_big_cloud2);
        vis.addPointCloud (filtered_big_cloud2, handler, "big_filtered_averaged", v3);
        vis.addPointCloudNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> (filtered_big_cloud2, filtered_big_cloud2, 10, 0.01, "normals", v3);
        if(visualize_)
            vis.spin ();
        else
            vis.spinOnce(100, true);

        model_cloud_ = filtered_big_cloud2;
    }

    /*if(visualize_)
    {
        pcl::visualization::PCLVisualizer vis("checking");
        int v1,v2, v3;
        vis.createViewPort(0,0,0.33,1,v1);
        vis.createViewPort(0.33,0,0.66,1,v2);
        vis.createViewPort(0.66,0,1,1,v3);
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> handler (filtered_big_cloud2);
        vis.addPointCloud (filtered_big_cloud2, handler, "big_filtered_averaged", v1);

        for(size_t i=0; i < transforms_to_global_.size(); i++)
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene (new pcl::PointCloud<pcl::PointXYZRGB>(*original_clouds_[i]));
            std::stringstream name;
            name << "cloud_" << i;
            pcl::transformPointCloud(*scene, *scene, transforms_to_global_[i]);
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler (scene);
            vis.addPointCloud (scene, handler, name.str(), v2);
        }

        vis.spin();
    }*/
}

POINT_CLOUD_REGISTER_POINT_STRUCT (IndexPoint,
                                   (int, idx, idx)
                                   )

template class faat_pcl::modelling::NiceModelFromSequence<pcl::PointXYZRGB, pcl::PointXYZRGBNormal>;

