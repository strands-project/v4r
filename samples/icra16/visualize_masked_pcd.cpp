//-scene_path /media/Data/datasets/TUW/test_set/set_00006/00000.pcd -mask_path /home/thomas/Documents/TUW_dol_eval/set_00006/burti_0_mask_85.txt

/*
 *  Created on: Aug, 2015
 *      Author: Thomas Faeulhammer
 *
 */

#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/io/filesystem.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <fstream>


int main (int argc, char ** argv)
{
    typedef pcl::PointXYZRGB PointT;
    std::string path_s, path_o;
    pcl::console::parse_argument (argc, argv, "-scene_path", path_s);
    pcl::console::parse_argument (argc, argv, "-mask_path", path_o);

    std::cout << path_s << " " << path_o << std::endl;

    pcl::visualization::PCLVisualizer vis;

    // Setting up the visualization
    int v1,v2;
    vis.createViewPort( 0,0,0.5,1,v1);
    vis.createViewPort(0.5,0,1,1,v2);

    // Loading file and corresponding indices file (if it exists)
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr obj_cloud (new pcl::PointCloud<PointT>);
    pcl::io::loadPCDFile (path_s, *cloud);

    if(0)
    {
        pcl::PointCloud<IndexPoint> obj_indices_cloud;
        pcl::io::loadPCDFile(path_o, obj_indices_cloud);
        pcl::PointIndices indices;
        indices.indices.resize(obj_indices_cloud.points.size());
        for(size_t kk=0; kk < obj_indices_cloud.points.size(); kk++)
            indices.indices[kk] = obj_indices_cloud.points[kk].idx;
        pcl::copyPointCloud(*cloud, indices, *obj_cloud);
    }
    else
    {
        std::ifstream mask_file;
        mask_file.open( path_o.c_str() );

        int idx_tmp;
        pcl::PointIndices pind;
        while (mask_file >> idx_tmp)
        {
            pind.indices.push_back(idx_tmp);
        }
        mask_file.close();
        pcl::copyPointCloud(*cloud, pind, *obj_cloud);
    }

    vis.addPointCloud (cloud, "original_cloud", v1);
    vis.addPointCloud (obj_cloud, "segmented_cloud", v2);
    vis.spin();
    return true;
}
