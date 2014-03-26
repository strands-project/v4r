#include "my_classifier.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>

int
main (int argc, char ** argv)
{
    pcl::PointCloud<PointT>::Ptr pCloud, pSegmentedCloud;
    pCloud.reset(new pcl::PointCloud<PointT>());
    pSegmentedCloud.reset(new pcl::PointCloud<PointT>());

    MyClassifier classifier;
    classifier.init (argc, argv);
    classifier.trainClassifier();
    classifier.classify();

    pcl::visualization::PCLVisualizer::Ptr vis;
    vis.reset(new pcl::visualization::PCLVisualizer("classifier visualization"));
    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb_handler (pSegmentedCloud);
    vis->addPointCloud<PointT> (pSegmentedCloud, rgb_handler, "classified_pcl");
    vis->spin();
    return 0;
}
