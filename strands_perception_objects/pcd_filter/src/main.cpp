#include "my_filter.h"
#include <pcl/io/pcd_io.h>
#include <pcl/io/io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <boost/filesystem.hpp>
#include <string>

#include <pcl/surface/convex_hull.h>

//#include <faat_pcl/utils/segmentation_utils.h>
#include <faat_pcl/3d_rec_framework/defines/faat_3d_rec_framework_defines.h>
#include <pcl/apps/dominant_plane_segmentation.h>

namespace bf = boost::filesystem;

typedef pcl::PointXYZRGB PointT;

void segmentObjectFromTableTop(pcl::PointCloud<PointT>::Ptr pInput_cloud, pcl::PointCloud<PointT>::Ptr pOutput_cloud, std::vector<pcl::PointIndices> &indices_above_plane, float z_max = 1.5f)
{
    std::vector<pcl::PointCloud<PointT>::Ptr > clusters;
    Eigen::Vector4f table_plane;
    //segmentation_utils::computeTablePlane<PointT>(pInput_cloud, table_plane, z_max);
    boost::shared_ptr< pcl::apps::DominantPlaneSegmentation<PointT>  > dps (new pcl::apps::DominantPlaneSegmentation<PointT>());
    dps->setInputCloud(pInput_cloud);
    dps->setMaxZBounds(z_max);
    //dps->compute_table_plane();
    dps->compute_fast(clusters);
    dps->getTableCoefficients(table_plane);
    dps->getIndicesClusters(indices_above_plane);

    /*for (int k = 0; k < pInput_cloud->points.size (); k++)
    {
        Eigen::Vector3f xyz_p = pInput_cloud->points[k].getVector3fMap ();

        if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
            continue;

        float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

        if (val >= 0.01)
        {
            indices_above_plane.push_back (static_cast<int> (k));
        }
    }*/

    pcl::copyPointCloud(*pInput_cloud, indices_above_plane, *pOutput_cloud);
}

void getFilenamesFromFilename(bf::path & dir, std::vector<std::string> & file_v)
{
    bf::directory_iterator end_itr;
    for (bf::directory_iterator itr (dir); itr != end_itr; ++itr)
    {
        std::string path;
#if BOOST_FILESYSTEM_VERSION == 3
          path = dir.string() + "/" +  (itr->path ().filename ()).string();
#else
          path = dir + "/" +  (itr->path ()).filename ();
#endif

        if (bf::is_directory (*itr))
        {
            bf::path path_bf = path;
            getFilenamesFromFilename(path_bf, file_v);
        }
        else
        {
            file_v.push_back(path);
        }
    }

}

int
main (int argc, char ** argv)
{
    ros::init(argc, argv, "pcd_filter_node");

    std::string indices_prefix = "object_indices_";
    std::vector<std::string> file_v;
    bool force_refilter = true;

    bf::path path = "/home/thomas/data/Cat50_TestDB_small/pcd_binary";
    getFilenamesFromFilename(path, file_v);

    for(size_t i=0; i<file_v.size(); i++)
    {
        std::string directory, filename;
        char sep = '/';
         #ifdef _WIN32
            sep = '\\';
         #endif

        size_t position = file_v[i].rfind(sep);
           if (position != std::string::npos)
           {
              directory = file_v[i].substr(0, position);
              filename = file_v[i].substr(position+1, file_v[i].length()-1);
           }

       std::stringstream path_oi;
       path_oi << directory << "/" << indices_prefix << filename ;

        if(bf::exists(path_oi.str()) && !force_refilter)
        {
            std::cout << filename << " is already filtered and no re-filtering desired. " << std::cout;
            continue;
        }

        if(filename.length() > indices_prefix.length())
        {
            if(filename.compare(0, indices_prefix.length(), indices_prefix)==0 )
            {
                std::cout << filename << " is not a point cloud. " << std::cout;
                continue;
            }
        }


        pcl::PointCloud<PointT>::Ptr pCloud, pSegmentedCloud;
        pCloud.reset(new pcl::PointCloud<PointT>());
        pSegmentedCloud.reset(new pcl::PointCloud<PointT>());
        pcl::io::loadPCDFile(file_v[i], *pCloud);
        std::vector<pcl::PointIndices> indices;
        segmentObjectFromTableTop(pCloud, pSegmentedCloud, indices, 1.2f);

        pcl::PointCloud<IndexPoint> obj_indices_cloud;
        obj_indices_cloud.width = indices[0].indices.size();
        obj_indices_cloud.height = 1;
        obj_indices_cloud.points.resize(indices[0].indices.size());

        for(size_t kk=0; kk < indices[0].indices.size(); kk++)
        {
            obj_indices_cloud.points[kk].idx = indices[0].indices[kk];
        }

        pcl::io::savePCDFileBinary(path_oi.str(), obj_indices_cloud);

        /*pcl::visualization::PCLVisualizer::Ptr vis;
        vis.reset(new pcl::visualization::PCLVisualizer("classifier visualization"));
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler (pSegmentedCloud);
        vis->addPointCloud<PointT> (pSegmentedCloud, rgb_handler, "classified_pcl");
        vis->spin();*/
    }
    return 0;
}
