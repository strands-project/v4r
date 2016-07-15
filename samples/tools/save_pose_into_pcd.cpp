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
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <iostream>
#include <sstream>

#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;


typedef pcl::PointXYZRGB PointT;

int main (int argc, char ** argv)
{
    std::string input_dir, out_dir = "/tmp/clouds_with_pose/";
    std::string cloud_prefix = "cloud_", pose_prefix = "pose_", indices_prefix = "object_indices_";
    bool invert_pose = false;
    bool use_indices = false;
    po::options_description desc("Save camera pose given by seperate pose file directly into header of .pcd file\n======================================\n**Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("input_dir,i", po::value<std::string>(&input_dir)->required(), "directory containing both .pcd and pose files")
            ("output_dir,o", po::value<std::string>(&out_dir)->default_value(out_dir), "output directory")
            ("use_indices,u", po::bool_switch(&use_indices), "if true, uses indices")
            ("invert_pose,t", po::bool_switch(&invert_pose), "if true, takes the inverse of the pose file (e.g. required for Willow Dataset)")
            ("cloud_prefix,c", po::value<std::string>(&cloud_prefix)->default_value(cloud_prefix)->implicit_value(""), "prefix of cloud names")
            ("pose_prefix,p", po::value<std::string>(&pose_prefix)->default_value(pose_prefix), "prefix of camera pose names (e.g. transformation_)")
            ("indices_prefix,d", po::value<std::string>(&indices_prefix)->default_value(indices_prefix), "prefix of object indices names")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return false;
    }

    try  { po::notify(vm); }
    catch(std::exception& e)  {
        std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
        return false;
    }

    std::vector< std::string> sub_folder_names = v4r::io::getFoldersInDirectory( input_dir );

    if( sub_folder_names.empty() )
        sub_folder_names.push_back("");

    for (const std::string &sub_folder : sub_folder_names)
    {
        const std::string path_extended = input_dir + "/" + sub_folder;

        std::cout << "Processing all point clouds in folder " << path_extended;
//        const std::string search_pattern = ".*." + cloud_prefix + ".*.pcd";
        const std::string search_pattern = ".*.pcd";
        std::vector < std::string > cloud_files = v4r::io::getFilesInDirectory (path_extended, search_pattern, true);

        for (const std::string &cloud_file : cloud_files)
        {
            const std::string full_path = path_extended + "/"  + cloud_file;
            const std::string out_fn = out_dir + "/" + sub_folder + "/" + cloud_file;

            // Loading file and corresponding indices file (if it exists)
            pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
            pcl::io::loadPCDFile (full_path, *cloud);

            if(use_indices)
            {
                // Check if indices file exist
                std::string indices_filename( cloud_file );
                boost::replace_first (indices_filename, cloud_prefix, indices_prefix);

                if( v4r::io::existsFile( path_extended+"/"+indices_filename ) )
                {
                    pcl::PointCloud<IndexPoint> obj_indices_cloud;
                    pcl::io::loadPCDFile(path_extended+"/"+indices_filename, obj_indices_cloud);
                    pcl::PointIndices indices;
                    indices.indices.resize(obj_indices_cloud.points.size());
                    for(size_t kk=0; kk < obj_indices_cloud.points.size(); kk++)
                        indices.indices[kk] = obj_indices_cloud.points[kk].idx;
                    pcl::copyPointCloud(*cloud, indices, *cloud);
                }
                else
                {
                    boost::replace_last( indices_filename, ".pcd", ".txt");
                    if( v4r::io::existsFile( path_extended+"/"+indices_filename ) )
                    {
                        std::vector<int> indices;
                        int idx_tmp;
                        std::ifstream mask_f( path_extended+"/"+indices_filename );

                        while (mask_f >> idx_tmp)
                            indices.push_back(idx_tmp);

                        mask_f.close();

                        pcl::copyPointCloud(*cloud, indices, *cloud);
                    }
                    else
                    {
                        std::cerr << "Indices file " << path_extended << "/" << indices_filename << " does not exist." << std::endl;
                    }
                }
            }

            // Check if pose file exist
            std::string pose_fn( cloud_file );
            boost::replace_first (pose_fn, cloud_prefix, pose_prefix);
            boost::replace_last (pose_fn, ".pcd", ".txt");
            const std::string full_pose_path = input_dir + pose_fn;

            if( v4r::io::existsFile( full_pose_path ) )
            {
                Eigen::Matrix4f global_trans = v4r::io::readMatrixFromFile(full_pose_path);

                if(invert_pose)
                    global_trans = global_trans.inverse();

                v4r::setCloudPose(global_trans, *cloud);
                v4r::io::createDirForFileIfNotExist(out_fn);
                pcl::io::savePCDFileBinary(out_fn, *cloud);
            }
            else
                std::cerr << "Pose file " << full_pose_path << " does not exist." << std::endl;
        }
    }
}
