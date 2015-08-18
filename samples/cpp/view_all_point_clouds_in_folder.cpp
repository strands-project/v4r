/*
 * view all point clouds in a folder
 * (if indices file for segmentation exist, it will segment the object)
 *
 *  Created on: Dec 04, 2014
 *      Author: Thomas Faeulhammer
 *
 */

#include <v4r/io/filesystem.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <math.h>
#include <sstream>
#include <v4r/common/faat_3d_rec_framework_defines.h>

typedef pcl::PointXYZRGB PointT;

int main (int argc, char ** argv)
{
    std::string path;
    int i_rows = 2;
    pcl::console::parse_argument (argc, argv, "-path", path);
    pcl::console::parse_argument (argc, argv, "-rows", i_rows);
    size_t rows = static_cast<size_t>(i_rows);

    std::cout << "Visualizing all point clouds in folder " << path;
    std::vector < std::string > files_intern;
    if (v4r::io::getFilesInDirectory (path, files_intern, "", ".*.pcd", true) == -1)
    {
        std::cerr << "Given path: " << path << " does not exist. Usage view_all_point_clouds_in_folder -path <folder name> -rows <number of rows used for displaying files>" << std::endl;
        return -1;
    }

    std::vector < std::string > cloud_files;
    for (size_t file_id=0; file_id < files_intern.size(); file_id++)
    {
        if ( files_intern[file_id].find("indices") == std::string::npos ) {
            cloud_files.push_back( files_intern [file_id] );
        }
    }

    std::sort(cloud_files.begin(), cloud_files.end());

    pcl::visualization::PCLVisualizer vis;
    size_t cols = std::ceil(cloud_files.size()/ static_cast<float> (rows));
    std::vector<int> viewport(rows * cols);

    for (size_t file_id=0; file_id < cloud_files.size(); file_id++)
    {
        std::stringstream full_path_ss;
#ifdef _WIN32
        full_path << path << "\\" << cloud_files[file_id];
#else
        full_path_ss << path << "/"  << cloud_files[file_id];
#endif

        std::cout << "Visualizing " << full_path_ss.str() << std::endl;


        // Setting up the visualization
        int col_id = file_id%cols;
        int row_id = std::floor(file_id/float(cols));
        vis.createViewPort( float(col_id)/cols, float(row_id)/rows, float(col_id + 1.0)/cols, float(row_id + 1.0)/rows, viewport[file_id]);
        vis.setBackgroundColor(1, 1, 1, viewport[file_id]);

        // Loading file and corresponding indices file (if it exists)
        pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
        pcl::io::loadPCDFile (full_path_ss.str(), *cloud);

        // Check if indices file exist
        std::string indices_filename( cloud_files[file_id] );
        boost::replace_all (indices_filename, "cloud", "object_indices");

        std::stringstream full_indices_path_ss;
#ifdef _WIN32
        full_indices_path_ss << path << "\\" << indices_filename;
#else
        full_indices_path_ss << path << "/"  << indices_filename;
#endif

        if( v4r::io::existsFile( full_indices_path_ss.str()) )
        {
            pcl::PointCloud<IndexPoint> obj_indices_cloud;
            pcl::io::loadPCDFile(full_indices_path_ss.str(), obj_indices_cloud);
            pcl::PointIndices indices;
            indices.indices.resize(obj_indices_cloud.points.size());
            for(size_t kk=0; kk < obj_indices_cloud.points.size(); kk++)
              indices.indices[kk] = obj_indices_cloud.points[kk].idx;
            pcl::copyPointCloud(*cloud, indices, *cloud);
        }
        else
        {
            std::cout << "Indices file " << full_indices_path_ss.str() << " does not exist." << std::endl;
        }

        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler (cloud);
        vis.addPointCloud<PointT> (cloud, rgb_handler, cloud_files[file_id], viewport[file_id]);

#ifdef DEMONSTRATE
        // Read depth and color values from each pixel value (just as an example how to access cloud elements)
        for(size_t pt_id = 0; pt_id < cloud->points.size(); pt_id++)
        {
            float x,y,z,r,g,b;
            PointT pt;
            pt = cloud->points[pt_id];

            x = pt.x;
            y = pt.y;
            z = pt.z;
            r = pt.r;
            g = pt.g;
            b = pt.b;

            double lab_l, lab_a, lab_b;
            v4r::Slic::convertRGBtoLAB(r, g, b, lab_l, lab_a, lab_b);
            if((pt_id % 50000)==0)    //just output every 50000th point, otherwise it will take too long and produce too much output
            {
                std::cout << "XYZ: " << x << ", " << y << ", " << z
                          << "; RGB " << r << ", " << g << ", " << b
                          << "; LAB " << lab_l << ", " << lab_a << ", " << lab_b << std::endl;
            }
        }
#endif
    }

    for (size_t file_id=cloud_files.size(); file_id < rows*cols; file_id++)
    {
       int col_id = file_id%cols;
       int row_id = std::floor(file_id/float(cols));
       vis.createViewPort( float(col_id)/cols, float(row_id)/rows, float(col_id + 1.0)/cols, float(row_id + 1.0)/rows, viewport[file_id]);
       vis.setBackgroundColor(1, 1, 1, viewport[file_id]);
    }
    vis.spin();
}
