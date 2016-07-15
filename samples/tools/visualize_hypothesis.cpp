/*
 *  Created on: Aug, 2015
 *      Author: Thomas Faeulhammer
 *
 */

#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/io/eigen.h>
#include <v4r/io/filesystem.h>
#include <v4r/recognition/model_only_source.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <fstream>


int main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PointT;
    typedef v4r::Model<PointT> ModelT;
    typedef boost::shared_ptr<ModelT> ModelTPtr;

    std::string path_s, path_p, path_m, model_name;
    pcl::console::parse_argument (argc, argv, "-scene_path", path_s);
    pcl::console::parse_argument (argc, argv, "-pose_file", path_p);
    pcl::console::parse_argument (argc, argv, "-model_path", path_m);
    pcl::console::parse_argument (argc, argv, "-model_name", model_name);

    std::cout << path_s << " " << path_p << " " << path_m << std::endl;

    boost::shared_ptr < v4r::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB> > source;
    source.reset (new v4r::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>);
    source->setPath (path_m);
    source->setLoadViews (false);
    source->setLoadIntoMemory(false);
    source->generate ();
//    source->createVoxelGridAndDistanceTransform (0.005f);

    pcl::visualization::PCLVisualizer vis;
    // Loading file and corresponding indices file (if it exists)
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::io::loadPCDFile (path_s, *cloud);

    Eigen::Matrix4f transform = v4r::io::readMatrixFromFile(path_p);

//    std::vector < std::string > parts;
//    boost::split (parts, path_p, boost::is_any_of ("/"));
//    std::string filename = parts.back();
//    boost::replace_all(filename, ".txt", "");
//    boost::split (parts, filename, boost::is_any_of ("_"));
//    filename = filename.substr(0, filename.length() - parts.back().length());
//    const std::string model_name =  parts[1] +  parts[2];


    ModelTPtr model;
    bool found = source->getModelById (model_name, model);
    pcl::PointCloud<PointT>::ConstPtr model_cloud = model->getAssembled(3);
    pcl::PointCloud<PointT>::Ptr model_cloud_transformed(new pcl::PointCloud<PointT>(*model_cloud));
    pcl::transformPointCloud(*model_cloud, *model_cloud_transformed, v4r::RotTrans2Mat4f(cloud->sensor_orientation_, cloud->sensor_origin_) * transform);

    vis.addPointCloud (cloud, "original_cloud");
    vis.addPointCloud (model_cloud_transformed, "model_cloud");
    vis.spin();
    return true;
}
