/*
 * main.cpp
 *
 *  Created on: Feb 20, 2014
 *      Author: Thomas Fäulhammer
 */

#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions.h>
#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "std_msgs/String.h"
#include "std_msgs/Int32MultiArray.h"
#include "classifier_srv_definitions/segment_and_classify.h"
#include "classifier_srv_definitions/classify.h"
#include "segmentation_srv_definitions/segment.h"
#include "object_perception_msgs/classification.h"
#include "geometry_msgs/Point32.h"

class SOCDemo
{
private:
    typedef pcl::PointXYZ PointT;
    int kinect_trials_;
    int service_calls_;
    std::string topic_;
    bool KINECT_OK_;
    bool all_required_services_okay_;
    ros::NodeHandle *n_;
    bool visualize_output_;
    std::vector< std_msgs::Int32MultiArray> cluster_indices_ros_;
    std::vector< object_perception_msgs::classification> class_results_ros_;
    std::vector< geometry_msgs::Point32> cluster_centroids_ros_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr sceneXYZ_;
    std::vector<std::string> text_3d_;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> vis_;
    bool do_segmentation_;

    void
    checkCloudArrive (const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
        KINECT_OK_ = true;
    }

    void
    checkKinect ()
    {
        ros::Subscriber sub_pc = n_->subscribe (topic_, 1, &SOCDemo::checkCloudArrive, this);
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

    bool callSegService(const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
        ros::ServiceClient segmentation_client = n_->serviceClient<segmentation_srv_definitions::segment>("/object_segmenter_service/object_segmenter");
        segmentation_srv_definitions::segment seg_srv;
        seg_srv.request.cloud = *msg;

        if (segmentation_client.call(seg_srv))
        {
            std::cout << "Number of clusters:" << static_cast<int>(seg_srv.response.clusters_indices.size()) << std::endl;
            cluster_indices_ros_ = seg_srv.response.clusters_indices;
        }
        else
        {
            ROS_ERROR("Failed to call segmentation service.");
            return false;
        }
        return true;
    }

    bool callClassifierService(const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
        ros::ServiceClient classifierClient = n_->serviceClient<classifier_srv_definitions::classify>("/classifier_service/classify");
        classifier_srv_definitions::classify srv;
        srv.request.cloud = *msg;
        srv.request.clusters_indices = cluster_indices_ros_;
        if (classifierClient.call(srv))
        {
            class_results_ros_ = srv.response.class_results;
            cluster_indices_ros_ = srv.response.clusters_indices;
            cluster_centroids_ros_ = srv.response.centroid;
        }
        else
        {
            ROS_ERROR("Failed to call classifier service.");
            return false;
        }
        return true;
    }

    bool callSegAndClassifierService(const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
        ros::ServiceClient segAndClassifierClient = n_->serviceClient<classifier_srv_definitions::segment_and_classify>("/classifier_service/segment_and_classify");
        classifier_srv_definitions::segment_and_classify srv;
        srv.request.cloud = *msg;
        if (segAndClassifierClient.call(srv))
        {
            class_results_ros_ = srv.response.class_results;
            cluster_indices_ros_ = srv.response.clusters_indices;
        }
        else
        {
            ROS_ERROR("Failed to call segmentation_and_classifier service.");
            return false;
        }
        return true;
    }

    void
    callService (const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
        // if any service is not available, wait for 5 sec and check again
        if( all_required_services_okay_ || ( !all_required_services_okay_ && (service_calls_ % (30 * 5)) == 0))
        {
            std::cout << "going to call service..." << std::endl;

            if(do_segmentation_)
            {
                bool segServiceOkay = callSegService(msg);
                bool classifierServiceOkay = callClassifierService(msg);
                all_required_services_okay_ = segServiceOkay & classifierServiceOkay;
            }
            else
            {
                all_required_services_okay_ = callSegAndClassifierService(msg);
            }
            pcl::fromROSMsg(*msg, *scene_);
            pcl::copyPointCloud(*scene_, *sceneXYZ_);

            if (visualize_output_ && all_required_services_okay_)
            {
                visualize_output();
            }
        }
        service_calls_++;

    }
public:
    SOCDemo()
    {
        scene_.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        sceneXYZ_.reset(new pcl::PointCloud<pcl::PointXYZ>);
        KINECT_OK_ = false;
        topic_ = "/camera/depth_registered/points";
        kinect_trials_ = 5;
        do_segmentation_ = true;
        all_required_services_okay_ = false;
    }

    bool initialize(int argc, char ** argv)
    {
        ros::init (argc, argv, "classifier_demo");
        if (sizeof(int) != 4)
        {
            ROS_WARN("PC Architectur does not use 32bit for integer - check conflicts with pcl indices.");
        }
        n_ = new ros::NodeHandle ( "~" );
        n_->getParam ( "topic", topic_ );
        n_->getParam ( "visualize_output", visualize_output_ );
        checkKinect();
        return KINECT_OK_;
    }

    void run()
    {
        ros::Subscriber sub_pc = n_->subscribe (topic_, 1, &SOCDemo::callService, this);
        ros::spin();
        /*ros::Rate loop_rate (5);
      while (ros::ok () && (service_calls_ < 5)) //only calls 5 times
      {
        ros::spinOnce ();
        loop_rate.sleep ();
      }*/
    }

    void visualize_output()
    {
        int detected_classes = 0;
        if(!vis_)
        {
            vis_.reset(new pcl::visualization::PCLVisualizer("classifier visualization"));
        }
        vis_->addCoordinateSystem(0.2f);
        float text_scale = 0.010f;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pClassifiedPCl (new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::copyPointCloud(*sceneXYZ_, *pClassifiedPCl);
        vis_->removeAllPointClouds();
        //for(size_t kk=0; kk < text_3d_.size(); kk++)
        //{
            //vis_->removeShape(text_3d_[kk]);
        //}
        text_3d_.clear();
        vis_->removeAllShapes();
        vis_->addPointCloud(sceneXYZ_, "scene_cloud");
        /*
                //show table plane
                std::vector<int> plane_indices;
                for(size_t i=0; i < frame->points.size(); i++)
                {
                Eigen::Vector3f xyz_p = frame->points[i].getVector3fMap ();

                if (!pcl_isfinite (xyz_p[0]) || !pcl_isfinite (xyz_p[1]) || !pcl_isfinite (xyz_p[2]))
                  continue;

                float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

                if (val <= tolerance && val >= -tolerance)
                {
                  plane_indices.push_back(i);
                }
                }

                pcl::PointCloud<PointT>::Ptr plane (new pcl::PointCloud<PointT>);
                pcl::copyPointCloud(*frame, plane_indices, *plane);

                pcl::visualization::PointCloudColorHandlerCustom<PointT> random_handler (plane, 0, 255, 0);
                vis_->addPointCloud<PointT> (plane, random_handler, "table plane");
                vis_->spinOnce();
                */

        for(size_t i=0; i < cluster_indices_ros_.size(); i++)
        {
            float r = std::rand() % 255;
            float g = std::rand() % 255;
            float b = std::rand() % 255;
            for(size_t kk=0; kk < cluster_indices_ros_[i].data.size(); kk++)
            {
                pClassifiedPCl->at(cluster_indices_ros_[i].data[kk]).r = r;
                pClassifiedPCl->at(cluster_indices_ros_[i].data[kk]).g = g;
                pClassifiedPCl->at(cluster_indices_ros_[i].data[kk]).b = b;
            }
            std::stringstream cluster_name;
            if(class_results_ros_[i].class_type.size() > 0)
            {
                cluster_name << "#" << i << ": " << class_results_ros_[i].class_type[0].data;
                std::cout << "Cluster " << i << ": " << std::endl;
                for (size_t kk = 0; kk < class_results_ros_[i].class_type.size(); kk++)
                {
                    std::cout << class_results_ros_[i].class_type[kk].data <<
                                 " [" << class_results_ros_[i].confidence[kk] << "]" << std::endl;

                    std::stringstream prob_str;
                    prob_str.precision (2);
                    prob_str << class_results_ros_[i].class_type[kk].data << " [" << class_results_ros_[i].confidence[kk] << "]";

                    std::stringstream cluster_text;
                    cluster_text << "cluster_" << i << "_" << kk << "_text";
                    text_3d_.push_back(cluster_text.str());
                    vis_->addText(prob_str.str(), 10+150*detected_classes, 10+kk*25,
                                  17, r/255.f, g/255.f, b/255.f, cluster_text.str());
                }
                pcl::PointXYZ pos;
                pos.x = cluster_centroids_ros_[i].x;
                pos.y = cluster_centroids_ros_[i].y;
                pos.z = cluster_centroids_ros_[i].z;
                text_3d_.push_back(cluster_name.str());
                vis_->addText3D(cluster_name.str(), pos, text_scale, 0.8*r/255.f, 0.8*g/255.f, 0.8*b/255.f, cluster_name.str(), 0);
                detected_classes++;
            }
            std::cout << std::endl;
        }
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_handler (pClassifiedPCl);
        vis_->addPointCloud<pcl::PointXYZRGB> (pClassifiedPCl, rgb_handler, "classified_pcl");
        vis_->spin();
    }
};

int
main (int argc, char ** argv)
{
    SOCDemo m;
    m.initialize (argc, argv);
    m.run();
    return 0;
}
