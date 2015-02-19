/*
 * view all point clouds in a folder.cpp
 *
 *  Created on: Dec 04, 2014
 *      Author: thomas f.
 */

#include <v4r/ORUtils/filesystem_utils.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <math.h>
#include <sstream>




int
main (int argc, char ** argv)
{
    std::string path;
    int i_rows;
    pcl::console::parse_argument (argc, argv, "-path", path);
    pcl::console::parse_argument (argc, argv, "-rows", i_rows);
    size_t rows = static_cast<size_t>(i_rows);
    bf::path path_bf = path;

    if(!bf::exists(path_bf))
    {
        std::cerr << "Given path: " << path << " does not exist. " << std::endl;
        return -1;
    }

    std::cout << "Visualizing all point clouds in folder " << path;
    std::vector < std::string > files_intern;
    faat_pcl::utils::getFilesInDirectory (path_bf, files_intern, "", ".*.pcd", true);
    std::sort(files_intern.begin(), files_intern.end());


    pcl::visualization::PCLVisualizer vis;
    size_t cols = std::ceil(files_intern.size()/ static_cast<float> (rows));
    std::vector<int> viewport(rows * cols);

    for (size_t file_id=0; file_id < files_intern.size(); file_id++)
    {
        std::stringstream full_path;
        full_path << path << "/" << files_intern[file_id];
        std::cout << "Visualizing " << full_path.str() << std::endl;

        int col_id = file_id%cols;
        int row_id = std::floor(file_id/float(cols));

        vis.createViewPort( float(col_id)/cols, float(row_id)/rows, float(col_id + 1.0)/cols, float(row_id + 1.0)/rows, viewport[file_id]);
        vis.setBackgroundColor(1,1,1,viewport[file_id]);
//        vis->createViewPort (float (i) / number_of_views, float (j) / number_of_subwindows_per_view, (float (i) + 1.0) / number_of_views,
//                             float (j + 1) / number_of_subwindows_per_view, viewportNr[number_of_subwindows_per_view * i + j]);


        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::io::loadPCDFile (full_path.str(), *cloud);
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_handler (cloud);
        vis.addPointCloud<pcl::PointXYZRGB> (cloud, rgb_handler, files_intern[file_id], viewport[file_id]);
    }

    for (size_t file_id=files_intern.size(); file_id < rows*cols; file_id++)
    {
       int col_id = file_id%cols;
       int row_id = std::floor(file_id/float(cols));
       vis.createViewPort( float(col_id)/cols, float(row_id)/rows, float(col_id + 1.0)/cols, float(row_id + 1.0)/rows, viewport[file_id]);
       vis.setBackgroundColor(1,1,1,viewport[file_id]);
    }
    vis.spin();
}
