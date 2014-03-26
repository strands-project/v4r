/*
 * shape_simple_classifier_node.cpp
 *
 *  Created on: Feb 24, 2014
 *      Author: Thomas Fäulhammer
 */

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include <pcl/common/common.h>
#include <pcl/console/parse.h>
#include <pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>
#include "segmentation_srv_definitions/segment.h"
#include "v4r/SurfaceSegmenter/segmentation.hpp"
#include "std_msgs/Int32MultiArray.h"

class RGBDSegmenterService
{
private:
  typedef pcl::PointXYZRGB PointT;
  double chop_at_z_;
  int v1_,v2_, v3_;
  ros::ServiceServer segment_;
  ros::NodeHandle *n_;
  boost::shared_ptr<segmentation::Segmenter> segmenter_;
  std::string model_filename_, scaling_filename_;

  bool
  segment (segmentation_srv_definitions::segment::Request & req, segmentation_srv_definitions::segment::Response & response)
  {
    pcl::PointCloud<PointT>::Ptr scene (new pcl::PointCloud<PointT>);
    pcl::fromROSMsg (req.cloud, *scene);

    if(chop_at_z_ > 0)
    {
        pcl::PassThrough<PointT> pass_;
        pass_.setFilterLimits (0.f, chop_at_z_);
        pass_.setFilterFieldName ("z");
        pass_.setInputCloud (scene);
        pass_.setKeepOrganized (true);
        pass_.filter (*scene);
    }
    
    std::cout << scene->points.size() << std::endl;
    segmenter_->setPointCloud(scene);
    segmenter_->segment();
    
    std::vector<std::vector<int> > clusters = segmenter_->getSegmentedObjectsIndices();
    
    std::cout << scene->points.size() << " " << clusters.size() << std::endl;
    for(size_t i=0; i < clusters.size(); i++)
    {
        std_msgs::Int32MultiArray indx;
    	for(size_t k=0; k < clusters[i].size(); k++)
    	{
            indx.data.push_back(clusters[i][k]);
        }
    	response.clusters_indices.push_back(indx);
    }
    
    return true;
  }
public:
  RGBDSegmenterService ()
  {
    //default values
    chop_at_z_ = 2.f;
  }

  void
  initialize (int argc, char ** argv)
  {
    ros::init (argc, argv, "object_segmenter_service");
    n_ = new ros::NodeHandle ( "~" );
    n_->getParam ( "model_filename", model_filename_ );
    n_->getParam ( "scaling_filename", scaling_filename_ );
    n_->getParam ( "chop_z", chop_at_z_ );

    if (model_filename_.compare ("") == 0)
    {
      PCL_ERROR ("Set -model_filename option in the command line, ABORTING");
      return;
    }
    
    if (scaling_filename_.compare ("") == 0)
    {
      PCL_ERROR ("Set -scaling_filename option in the command line, ABORTING");
      return;
    }
    
    segmenter_.reset(new segmentation::Segmenter);
    segmenter_->setModelFilename(model_filename_);
    segmenter_->setScaling(scaling_filename_);
    
    segment_ = n_->advertiseService ("object_segmenter", &RGBDSegmenterService::segment, this);
    std::cout << "Ready to get service calls..." << std::endl;
    ros::spin ();
  }
};

int
main (int argc, char ** argv)
{
  RGBDSegmenterService m;
  m.initialize (argc, argv);

  return 0;
}
