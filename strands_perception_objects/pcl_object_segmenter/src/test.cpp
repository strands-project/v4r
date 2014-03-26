/*
 * main.cpp
 *
 *  Created on: Sep 7, 2013
 *      Author: aitor
 */

#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "std_msgs/String.h"
#include "segmentation_srv_definitions/segment.h"
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions.h>

class SOCDemo
{
  private:
    int kinect_trials_;
    int service_calls_;
    ros::NodeHandle n_;
    std::string camera_topic_;
    bool KINECT_OK_;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> vis_;
    int v1_,v2_;
    
    void
    checkCloudArrive (const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
      KINECT_OK_ = true;
    }

    void
    checkKinect ()
    {
      ros::Subscriber sub_pc = n_.subscribe (camera_topic_, 1, &SOCDemo::checkCloudArrive, this);
      ros::Rate loop_rate (1);
      kinect_trials_ = 0;
      while (!KINECT_OK_ && ros::ok ())
      {
        std::cout << "Checking kinect status..." << std::endl;
        ros::spinOnce ();
        loop_rate.sleep ();
        kinect_trials_++;
        if(kinect_trials_ >= 5)
        {
          std::cout << "Kinect is not working..." << std::endl;
          return;
        }
      }

      KINECT_OK_ = true;
      std::cout << "Kinect is up and running" << std::endl;
    }

    void
    callService (const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
      if( (service_calls_ % (30 * 5)) == 0)
      {
        std::cout << "going to call service..." << std::endl;
        ros::ServiceClient client = n_.serviceClient<segmentation_srv_definitions::segment>("object_segmenter");
        segmentation_srv_definitions::segment srv;
        srv.request.cloud = *msg;
        if (client.call(srv))
        {
          std::cout << "Number of clusters:" << static_cast<int>(srv.response.clusters_indices.size()) << std::endl;
          pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene (new pcl::PointCloud<pcl::PointXYZRGB>);
	      pcl::fromROSMsg (*msg, *scene); 
	      
	      pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_labels (new pcl::PointCloud<pcl::PointXYZRGB>);
	      pcl::copyPointCloud(*scene, *scene_labels);
	      
	      	std::vector<uint32_t> label_colors_;
            int max_label = srv.response.clusters_indices.size();
			label_colors_.reserve (max_label + 1);
			srand (static_cast<unsigned int> (time (0)));
			while (label_colors_.size () <= max_label )
			{
			   uint8_t r = static_cast<uint8_t>( (rand () % 256));
			   uint8_t g = static_cast<uint8_t>( (rand () % 256));
			   uint8_t b = static_cast<uint8_t>( (rand () % 256));
			   label_colors_.push_back (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
			}

            for(int i=0; i < srv.response.clusters_indices.size(); i++)
			{
                std::cout << "cluster size:" << srv.response.clusters_indices[i].data.size() << std::endl;
                for(size_t k=0; k < srv.response.clusters_indices[i].data.size(); k++)
				{
                    scene_labels->points[srv.response.clusters_indices[i].data[k]].rgb = label_colors_[i];
				}
			}
    	
	      vis_->addPointCloud(scene, "cloud", v1_);
	      
          pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler(scene_labels);
          vis_->addPointCloud<pcl::PointXYZRGB>(scene_labels, handler, "cloud_labels", v2_);
          
	      vis_->spin();
	      vis_->removeAllPointClouds();
/*          for(size_t i=0; i < srv.response.categories_found.size(); i++)
          {
            std::cout << "   => " << srv.response.categories_found[i] << std::endl;
          }*/
        }
        else
        {
        	std::cout << "Call did not succeed" << std::endl;
        }
      }
      service_calls_++;
    }

  public:
    SOCDemo()
    {
      KINECT_OK_ = false;
      camera_topic_ = "/camera/depth_registered/points";
      kinect_trials_ = 5;
    }

    bool initialize(int argc, char ** argv)
    {
      checkKinect();
      vis_.reset (new pcl::visualization::PCLVisualizer ("segmenter visualization"));
      vis_->createViewPort(0,0,0.5,1,v1_);
      vis_->createViewPort(0.5,0,1.0,1,v2_);
      return KINECT_OK_;
    }

    void run()
    {
      ros::Subscriber sub_pc = n_.subscribe (camera_topic_, 1, &SOCDemo::callService, this);
      ros::spin();
    }
};

int
main (int argc, char ** argv)
{
  ros::init (argc, argv, "segmentation_demo");

  SOCDemo m;
  m.initialize (argc, argv);
  m.run();
  return 0;
}
