/*
 * adds pose information given by pose_000x.txt into cloud_000x.pcd files (old dataset structure)
 * now we can use pcl_viewer *.pcd to show all point clouds in common coordinate system
 *
 *  Created on: Dec 04, 2014
 *      Author: Thomas Faeulhammer
 *
 */

#include <v4r/io/filesystem.h>
#include <v4r/io/eigen.h>
#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/common/miscellaneous.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <iostream>
#include <sstream>

typedef pcl::PointXYZRGB PointT;


int main (int argc, char ** argv)
{
    std::string path;
    bool use_indices = false;
    pcl::console::parse_argument (argc, argv, "-path", path);
    pcl::console::parse_argument (argc, argv, "-use_indices", use_indices);

    std::cout << "Processing all point clouds in folder " << path;
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

    for (size_t file_id=0; file_id < cloud_files.size(); file_id++)
    {
        std::stringstream full_path_ss;
#ifdef _WIN32
        full_path << path << "\\" << cloud_files[file_id];
#else
        full_path_ss << path << "/"  << cloud_files[file_id];
#endif
        std::cout << "Checking " << full_path_ss.str() << std::endl;

        // Loading file and corresponding indices file (if it exists)
        pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
        pcl::io::loadPCDFile (full_path_ss.str(), *cloud);

        if(use_indices)
        {
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
        }

        // Check if pose file exist
        std::string pose_filename( cloud_files[file_id] );
        boost::replace_all (pose_filename, ".pcd", ".txt");
        std::string full_pose_path;

#ifdef USE_WILLOW_DATASET
    boost::replace_all (pose_filename, "cloud_", "pose_");
    #ifdef _WIN32
            full_pose_path = path + "\\" + pose_filename;
    #else
            full_pose_path = path + "/"  + pose_filename;
    #endif
#else
    #ifdef _WIN32
            full_pose_path = path + "\\" + "transformation_" + pose_filename;
    #else
            full_pose_path = path + "/"  + "transformation_" + pose_filename;
    #endif
#endif

        if( v4r::io::existsFile( full_pose_path ) )
        {
            std::cout << "Transform to world coordinate system: " << std::endl;
            Eigen::Matrix4f global_trans;
            v4r::io::readMatrixFromFile(full_pose_path, global_trans, 1);
            std::cout << global_trans << std::endl << std::endl;
            v4r::common::setCloudPose(global_trans, *cloud);
            pcl::io::savePCDFileBinary(full_path_ss.str(), *cloud);
        }
        else
        {
            std::cout << "Pose file " << full_pose_path << " does not exist." << std::endl;
        }

    }
}
