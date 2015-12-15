
/*
 * view all point clouds in a folder
 * (if indices file for segmentation exist, it will segment the object)
 *
 *  Created on: Dec 04, 2014
 *      Author: Thomas Faeulhammer
 *
 */
//-map_file /media/Data/datasets/icra16/icra16_uncontrolled_ba_test/object_list.csv -offline_models /media/Data/datasets/icra16/models -online_models /media/Data/datasets/icra16/models_learnt/uncontrolled_002/models

#include <v4r/io/filesystem.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/conversions.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <sstream>
#include <fstab.h>
#include <map>
#include <opencv2/opencv.hpp>
#include <v4r/common/pcl_opencv.h>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

typedef pcl::PointXYZRGB PointT;

std::vector<int> visualization_framework (pcl::visualization::PCLVisualizer &vis,
                                          size_t number_of_views,
                                          size_t number_of_subwindows_per_view,
                                          const std::vector<std::string> &title_subwindows = std::vector<std::string>());

std::vector<int> visualization_framework (pcl::visualization::PCLVisualizer &vis,
                                          size_t number_of_views,
                                          size_t number_of_subwindows_per_view,
                                          const std::vector<std::string> &title_subwindows)
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

bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}


int main (int argc, char ** argv)
{
    std::string controlled_models, uncontrolled_models, turntable_model_dir, info_file_controlled, info_file_uncontrolled, map_file, model_dir;

    std::string test_set_root_dir = "/home/thomas/Documents/icra16/keyframes/controlled_ba";

    po::options_description desc("Evaluation of partial model recognition results\n**Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("turntable_model_dir,t", po::value<std::string>(&turntable_model_dir)->default_value(turntable_model_dir), "")
            ("controlled_models_dir,c", po::value<std::string>(&controlled_models)->default_value(controlled_models), "")
            ("uncontrolled_models_dir,u", po::value<std::string>(&uncontrolled_models)->default_value(uncontrolled_models), "")
            ("info_file_controlled", po::value<std::string>(&info_file_controlled)->default_value(info_file_controlled), "")
            ("info_file_uncontrolled", po::value<std::string>(&info_file_uncontrolled)->default_value(info_file_uncontrolled), "")
            ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return false;
    }

    try
    {
        po::notify(vm);
    }
    catch(std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
        return false;
    }

    const size_t max_num_controlled_runs = 5;
    const size_t max_num_uncontrolled_runs = 3;
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_demeaned (new pcl::PointCloud<PointT>);


    std::vector<std::string> model_list;
    v4r::io::getFilesInDirectory(turntable_model_dir, model_list, "", ".*.pcd", false);
    std::sort(model_list.begin(), model_list.end());

    pcl::visualization::PCLVisualizer vis("object models");
    std::vector<int> viewports = visualization_framework (vis, model_list.size(), 5+3+1);

    std::vector<size_t> num_instances_per_model(model_list.size(), 0);
    for(size_t m_id=0; m_id<model_list.size(); m_id++)
    {
        const std::string fn = turntable_model_dir + "/" + model_list[m_id];
        pcl::io::loadPCDFile (fn, *cloud);

        for(size_t pt_id=0; pt_id<cloud->points.size(); pt_id++)
        {
            cloud->points[pt_id].z +=1.0f;  // otherwise the conversion to image won't work ( we will only see a very zoomed in version)
        }

        std::stringstream img_fn;
        img_fn << "/tmp/" << m_id << "_" << 0 << ".jpg";
//        cv::imwrite( img_fn.str(), v4r::ConvertUnorganizedPCLCloud2Image(*cloud, true));

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid (*cloud, centroid);
        pcl::demeanPointCloud (*cloud, centroid, *cloud_demeaned);
        vis.addPointCloud(cloud_demeaned, "cloud_" + m_id, viewports[m_id * (max_num_controlled_runs + max_num_uncontrolled_runs + 1)   + num_instances_per_model[m_id] ]);
    }

    ifstream info(info_file_controlled.c_str());
    std::string test_id, patrol_run_id, object_id;
    while (info >> test_id >> patrol_run_id >> object_id) {

        const std::string fn = controlled_models + "/" + test_id + ".pcd";

        if(!v4r::io::existsFile(fn))
            continue;

        pcl::io::loadPCDFile (fn, *cloud);

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid (*cloud, centroid);
        pcl::demeanPointCloud (*cloud, centroid, *cloud_demeaned);

        // check where we have to place that figure - to which viewport
        size_t vp_id = 0;
        for(size_t m_id=0; m_id<model_list.size(); m_id++)
        {
            if( hasEnding(object_id, model_list[m_id]) )
            {
                vp_id = m_id;
                num_instances_per_model[m_id] ++;
                break;
            }
        }

        vis.addPointCloud(cloud_demeaned, "cloud_" + test_id, viewports[vp_id * (max_num_controlled_runs + max_num_uncontrolled_runs + 1)   + num_instances_per_model[vp_id] ]);

        std::stringstream img_fn;
        img_fn << "/tmp/" << vp_id << "_" << num_instances_per_model[vp_id]  << ".jpg";
        cv::imwrite( img_fn.str(), v4r::ConvertUnorganizedPCLCloud2Image(*cloud, true));

    }
    info.close();


    // now read uncontrolled runs
    info.open(info_file_uncontrolled.c_str());
    num_instances_per_model.clear();
    num_instances_per_model.resize(model_list.size(), 0);
    while (info >> test_id >> patrol_run_id >> object_id) {

        const std::string fn = uncontrolled_models + "/" + test_id + ".pcd";

        if(!v4r::io::existsFile(fn))
            continue;

        pcl::io::loadPCDFile (fn, *cloud);

        pcl::PointCloud<PointT>::Ptr cloud_demeaned (new pcl::PointCloud<PointT>);
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid (*cloud, centroid);
        pcl::demeanPointCloud (*cloud, centroid, *cloud_demeaned);

        // check where we have to place that figure - to which viewport
        size_t vp_id = 0;
        for(size_t m_id=0; m_id<model_list.size(); m_id++)
        {
            if( hasEnding(object_id, model_list[m_id]) )
            {
                vp_id = m_id;
                num_instances_per_model[m_id] ++;
                break;
            }
        }


        std::stringstream img_fn;
        img_fn << "/tmp/" << vp_id << "_" << num_instances_per_model[vp_id] + max_num_controlled_runs << ".jpg";
        cv::imwrite( img_fn.str(), v4r::ConvertUnorganizedPCLCloud2Image(*cloud, true));

        vis.addPointCloud(cloud_demeaned, "cloud_uncontrolled_" + test_id, viewports[vp_id * (max_num_controlled_runs + max_num_uncontrolled_runs+1)  + num_instances_per_model[vp_id]+max_num_controlled_runs ]);
    }
    vis.spin();

    std::ofstream f("/tmp/latex_figure.txt");
    for(size_t r=0; r<(max_num_controlled_runs + max_num_uncontrolled_runs); r++)
    {
        for(size_t c=0; c<model_list.size(); c++)
        {
            f << "\\includegraphics[height=0.9cm]{pictures/models/" << r << "_" << c << ".jpg}&" << std::endl;
        }
        f << "\\\\" << std::endl;
    }
    f.close();
}
