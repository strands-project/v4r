/*
 *  Created on: Aug, 2015
 *      Author: Thomas Faeulhammer
 *
 */

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <fstream>

namespace po = boost::program_options;

Eigen::Matrix4f readMatrixFromFile(const std::string &file, int padding = 0);
Eigen::Matrix4f RotTrans2Mat4f(const Eigen::Quaternionf &q, const Eigen::Vector3f &trans);
Eigen::Matrix4f RotTrans2Mat4f(const Eigen::Quaternionf &q, const Eigen::Vector4f &trans);

Eigen::Matrix4f
readMatrixFromFile(const std::string &file, int padding)
{

    // check if file exists
    boost::filesystem::path path = file;
    if ( ! (boost::filesystem::exists ( path ) && boost::filesystem::is_regular_file(path)) )
        throw std::runtime_error ("Given file path to read Matrix does not exist!");

    std::ifstream in (file.c_str (), std::ifstream::in);

    char linebuf[1024];
    in.getline (linebuf, 1024);
    std::string line (linebuf);
    std::vector < std::string > strs_2;
    boost::split (strs_2, line, boost::is_any_of (" "));

    Eigen::Matrix4f matrix;
    for (int i = 0; i < 16; i++)
        matrix (i / 4, i % 4) = static_cast<float> (atof (strs_2[padding+i].c_str ()));

    return matrix;
}

Eigen::Matrix4f
RotTrans2Mat4f(const Eigen::Quaternionf &q, const Eigen::Vector4f &trans)
{
    Eigen::Matrix4f tf = Eigen::Matrix4f::Identity();
    tf.block<3,3>(0,0) = q.normalized().toRotationMatrix();
    tf.block<4,1>(0,3) = trans;
    return tf;
}

int main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PointT;

    std::string path_s, path_p, path_m;
    po::options_description desc("Visualize scene with object hypotheses in a given pose\n======================================\n**Required options");
    desc.add_options()
            ("help,h", "produce help message")
            ("scene_path,s", po::value<std::string>(&path_s)->required(), "Filename of the point cloud of the scene")
            ("model_path,m", po::value<std::string>(&path_m)->required(), "Filename of the point cloud of the model (e.g. 3D_model.pcd)")
            ("model_pose,p", po::value<std::string>(&path_p)->required(), "Filename of the 4x4 object pose stored in row-major order")
            ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) { std::cout << desc << std::endl; return false; }
    try  { po::notify(vm); }
    catch(std::exception& e)  {  std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl; return false; }

    pcl::visualization::PCLVisualizer vis;

    pcl::PointCloud<PointT>::Ptr scene_cloud (new pcl::PointCloud<PointT>);
    pcl::io::loadPCDFile (path_s, *scene_cloud);

    pcl::PointCloud<PointT>::Ptr model_cloud (new pcl::PointCloud<PointT>);
    pcl::io::loadPCDFile (path_m, *model_cloud);

    const Eigen::Matrix4f transform = readMatrixFromFile(path_p);
    const Eigen::Matrix4f scene_tf = RotTrans2Mat4f(scene_cloud->sensor_orientation_, scene_cloud->sensor_origin_);
    pcl::transformPointCloud(*model_cloud, *model_cloud,  scene_tf * transform);

    vis.addPointCloud (scene_cloud, "original_cloud");
    vis.addPointCloud (model_cloud, "model_cloud");
    vis.spin();
    return true;
}
