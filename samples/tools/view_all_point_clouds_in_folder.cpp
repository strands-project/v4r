/*
 * view all point clouds in a folder
 * (if indices file for segmentation exist, it will segment the object)
 * (if pose file exists, it will transform the point cloud by the corresponding pose)
 *
 *  Created on: Dec 04, 2014
 *      Author: Thomas Faeulhammer
 *
 */

#include <v4r/io/filesystem.h>
#include <v4r/io/eigen.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <math.h>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/common/pcl_opencv.h>
#include <v4r/io/filesystem.h>

int main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PointT;

    std::string path;
    int i_rows = 2;
    bool center = false;
    bool save_img = false;

    bool multi_view = true;
    std::string cloud_prefix = "cloud_";
    std::string pose_prefix = "pose_";
    std::string indices_prefix = "object_indices_";
    std::string out_img_dir = "/tmp/my_imgages/";

    pcl::console::parse_argument (argc, argv, "-path", path);
    pcl::console::parse_argument (argc, argv, "-rows", i_rows);
    pcl::console::parse_argument (argc, argv, "-center", center);
    pcl::console::parse_argument (argc, argv, "-save_img", save_img);
    pcl::console::parse_argument (argc, argv, "-out_img_dir", out_img_dir);
    pcl::console::parse_argument (argc, argv, "-cloud_prefix", cloud_prefix);
    pcl::console::parse_argument (argc, argv, "-pose_prefix", pose_prefix);
    pcl::console::parse_argument (argc, argv, "-indices_prefix", indices_prefix);
    pcl::console::parse_argument (argc, argv, "-multi_view", multi_view);

    size_t rows = static_cast<size_t>(i_rows);

    std::cout << "Visualizing all point clouds in folder " << path;
    const std::string filepattern = cloud_prefix + ".*.pcd";
    std::vector < std::string > files_intern = v4r::io::getFilesInDirectory (path, filepattern, false);
    if (files_intern.empty())
    {
        std::cerr << "Given path: " << path << " does not exist. "
                  << "Usage " << argv[0]
                  << "  -path <folder name>" << std::endl
                  << "  [-rows <number of rows used for displaying clouds ( default: " << i_rows << ")>]" << std::endl
                  << "  [-center <if true, centers the point cloud in its displaying window based on its centroid ( default: " << center << ")>]" << std::endl
                  << "  [-save_img <number of rows used for displaying files ( default: " << save_img << ")>]" << std::endl
                  << "  [-out_img_dir <path where images are stored ( default: " << out_img_dir << ")>]" << std::endl
                  << "  [-cloud_prefix <prefix of cloud filenames for displaying ( default: " << cloud_prefix << "  assumes .pcd file extension)>]" << std::endl
                  << "  [-pose_prefix <prefix of pose filenames for displaying ( default: " << pose_prefix << "  assumes .txt file extension)>]" << std::endl
                  << "  [-indices_prefix <prefix of object indices filenames for displaying ( default: " << indices_prefix << "  assumes .pcd file extension)>]" << std::endl
                  << "  [-multi_view <if true, displays each point clouds in seperate viewport ( default: " << multi_view << ")>]" << std::endl
                  << std::endl;
        return -1;
    }

    std::vector < std::string > cloud_files;
    for (size_t file_id=0; file_id < files_intern.size(); file_id++) {
        if ( files_intern[file_id].find(indices_prefix) == std::string::npos ) {    // this checks if the .pcd file has indices or cloud data stored
            cloud_files.push_back( files_intern [file_id] );
        }
    }

    // setting up visualizer
    pcl::visualization::PCLVisualizer vis;
    size_t cols = std::ceil(cloud_files.size()/ static_cast<float> (rows));  // will be ignored if multi_view = false
    std::vector<int> viewport(rows * cols); // will be ignored if multi_view = false

    std::sort(cloud_files.begin(), cloud_files.end());
    for (size_t file_id=0; file_id < cloud_files.size(); file_id++)
    {
        const std::string full_path = path + "/" + cloud_files[file_id];

        std::cout << "Visualizing " << full_path << std::endl;

        // Loading file and corresponding indices file (if it exists)
        pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
        pcl::io::loadPCDFile (full_path, *cloud);

        // Check if indices and pose file exist
        std::string indices_filename ( cloud_files[file_id] );
        boost::replace_first (indices_filename, cloud_prefix, indices_prefix);
        boost::replace_last (indices_filename, ".pcd", ".txt");
        const std::string full_indices_path = path + "/"  + indices_filename;

        if( v4r::io::existsFile( full_indices_path) )
        {
            std::vector<int> indices;
            std::ifstream f ( full_indices_path.c_str() );
            int idx;
            while (f >> idx)
                indices.push_back(idx);
            f.close();
            pcl::copyPointCloud(*cloud, indices, *cloud);
        }
        else
        {
            std::cout << "Indices file " << full_indices_path << " does not exist." << std::endl;
        }


        std::string pose_filename ( cloud_files[file_id] );
        boost::replace_first (pose_filename, cloud_prefix, pose_prefix);
        boost::replace_last (pose_filename, ".pcd", ".txt");
        const std::string full_pose_path = path + "/"  + pose_filename;
        if( v4r::io::existsFile( full_pose_path ) )
        {
            Eigen::Matrix4f tf = v4r::io::readMatrixFromFile(full_pose_path);
//            const Eigen::Matrix4f tf_inv = tf.inverse();
            pcl::transformPointCloud(*cloud, *cloud, tf);
        }
        else
        {
            std::cout << "Pose file " << full_pose_path << " does not exist." << std::endl;
        }

        if(save_img)
        {
            std::stringstream filename;
            v4r::io::createDirIfNotExist(out_img_dir);
            filename << out_img_dir << "/" << file_id << ".jpg";
            cv::Mat colorImage = v4r::ConvertPCLCloud2Image (*cloud);
            cv::imwrite( filename.str(), colorImage);
        }

        if(center)
        {
#if PCL_VERSION < 100702
            std::cout << "Centering is not implemented on your PCL version!" << std::endl;
#else
            PointT centroid;
            pcl::computeCentroid(*cloud, centroid);
            for(size_t pt_id=0; pt_id<cloud->points.size(); pt_id++) {
                cloud->points[pt_id].x -= centroid.x;
                cloud->points[pt_id].y -= centroid.y;
                cloud->points[pt_id].z -= centroid.z;
            }
#endif
        }
        if(multi_view) { // Setting up the visualization
            int col_id = file_id%cols;
            int row_id = std::floor(file_id/float(cols));
            vis.createViewPort( float(col_id)/cols, float(row_id)/rows, float(col_id + 1.0)/cols, float(row_id + 1.0)/rows, viewport[file_id]);
            vis.addPointCloud(cloud, cloud_files[file_id], viewport[file_id]);
        }
        else
            vis.addPointCloud(cloud, cloud_files[file_id]);
    }

    if(multi_view) { // display remaining windows with same background
        for (size_t file_id=cloud_files.size(); file_id < rows*cols; file_id++) {
           int col_id = file_id%cols;
           int row_id = std::floor(file_id/float(cols));
           vis.createViewPort( float(col_id)/cols, float(row_id)/rows, float(col_id + 1.0)/cols, float(row_id + 1.0)/rows, viewport[file_id]);
           vis.setBackgroundColor(1, 1, 1, viewport[file_id]);
        }
    }
    vis.spin();
}
