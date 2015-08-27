
/*
 * view all point clouds in a folder
 * (if indices file for segmentation exist, it will segment the object)
 *
 *  Created on: Dec 04, 2014
 *      Author: Thomas Faeulhammer
 *
 */

#include <v4r/io/filesystem.h>
#include <pcl/common/centroid.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <sstream>
#include <fstab.h>
#include <map>

typedef pcl::PointXYZRGB PointT;

std::vector<int> visualization_framework (pcl::visualization::PCLVisualizer &vis,
                                          size_t number_of_views,
                                          size_t number_of_subwindows_per_view,
                                          const std::vector<std::string> &title_subwindows = std::vector<std::string>())
{
  vis.removeAllPointClouds();
  vis.removeAllShapes();
  std::vector<int> viewportNr (number_of_views * number_of_subwindows_per_view, 0);

  for (size_t i = 0; i < number_of_views; i++)
  {
    for (size_t j = 0; j < number_of_subwindows_per_view; j++)
    {
      vis.createViewPort (float (i) / number_of_views, float (j) / number_of_subwindows_per_view, (float (i) + 1.0) / number_of_views,
                           float (j + 1) / number_of_subwindows_per_view, viewportNr[number_of_subwindows_per_view * i + j]);

      vis.setBackgroundColor (1,1,1, viewportNr[number_of_subwindows_per_view * i + j]);

      vis.removeAllPointClouds(viewportNr[i * number_of_subwindows_per_view + j]);
      vis.removeAllShapes(viewportNr[i * number_of_subwindows_per_view + j]);
      std::stringstream window_id;
      window_id << "(" << i << ", " << j << ") ";
      if(title_subwindows.size()>j)
      {
          window_id << title_subwindows[j];
      }
      vis.addText (window_id.str (), 10, 10, window_id.str (), viewportNr[i * number_of_subwindows_per_view + j]);
    }
  }
  return viewportNr;
}


int main (int argc, char ** argv)
{
    std::string online_models, offline_models, map_file;
    pcl::console::parse_argument (argc, argv, "-offline_models", offline_models);
    pcl::console::parse_argument (argc, argv, "-online_models", online_models);
    pcl::console::parse_argument (argc, argv, "-map_file", map_file);

    std::map<std::string, std::vector<std::string> > prun2ob;

    std::ifstream in;
    in.open (map_file.c_str (), std::ifstream::in);
    char linebuf[1024];
    while(in.getline (linebuf, 1024))
    {
        std::string line (linebuf);
        std::vector < std::string > strs_2;
        boost::split (strs_2, line, boost::is_any_of (","));
        if (strs_2[2].length())
            continue;

        const std::string patrol_run = strs_2[0];
        const std::string obj = strs_2[1];
        std::map<std::string, std::vector<std::string> >::iterator it = prun2ob.find(obj);
        if (it != prun2ob.end() )
        {
            it->second.push_back(patrol_run);
        }
        else
        {
            std::vector<std::string> pr_tmp;
            pr_tmp.push_back(patrol_run);
            prun2ob[obj] = pr_tmp;
        }
    }

    pcl::visualization::PCLVisualizer vis("object models");
    std::vector<int> viewports = visualization_framework (vis, prun2ob.size(), 6);



    size_t col_id = 0;
    std::map<std::string, std::vector<std::string> >::iterator it;
    for ( it = prun2ob.begin(); it != prun2ob.end(); it++, col_id++ )
    {
        std::stringstream filename;
        filename << offline_models << "/" << it->first;
        pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
        pcl::io::loadPCDFile (filename.str(), *cloud);

        PointT centroid;
        pcl::computeCentroid(*cloud, centroid);
        for(size_t pt_id=0; pt_id<cloud->points.size(); pt_id++)
        {
            cloud->points[pt_id].x -= centroid.x;
            cloud->points[pt_id].y -= centroid.y;
            cloud->points[pt_id].z -= centroid.z;
        }

        vis.addPointCloud(cloud,it->first,viewports[col_id*6]);
        vis.addText(it->first, 10, 10, 12, 0, 0, 0, it->first, viewports[col_id*6]);

        for (size_t pr=0; pr<it->second.size();pr++)
        {
            std::string patrol_run = it->second[pr];
            std::stringstream filename_pr;
            filename_pr << online_models << "/" << patrol_run << "_object.pcd";
            pcl::io::loadPCDFile (filename_pr.str(), *cloud);

            pcl::computeCentroid(*cloud, centroid);
            for(size_t pt_id=0; pt_id<cloud->points.size(); pt_id++)
            {
                cloud->points[pt_id].x -= centroid.x;
                cloud->points[pt_id].y -= centroid.y;
                cloud->points[pt_id].z -= centroid.z;
            }

            vis.addPointCloud(cloud, patrol_run, viewports[col_id * 6 + pr+1]);
            vis.addText(patrol_run, 10, 10, 12, 0, 0, 0, patrol_run, viewports[col_id * 6 + pr+1]);
        }
    }

    vis.spin();

}
