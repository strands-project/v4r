#pragma once

#include "inputModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/io/openni_grabber.h>

#include "output/pointCloudRenderer.h"

#include "pipeline.h"

namespace object_modeller
{
namespace reader
{

class CameraReader : public InModule<std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> >
{
protected:
    pcl::Grabber* grabber;
    boost::shared_ptr<output::Renderer> renderer;

    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> result;
    bool copyFrame;
    int nrSequences;

public:
    CameraReader(boost::shared_ptr<output::Renderer> renderer, std::string config_name="camreader") : InModule(config_name), renderer(renderer)
    {
        grabber = NULL;
        nrSequences = 1;
    }

    virtual int getNrOutputSequences()
    {
        return nrSequences;
    }

    virtual std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> process();

    void grabberCallback(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr& cloud);
    void image_callback(const boost::shared_ptr<openni_wrapper::Image>& image);

    std::string getName()
    {
        return "Camera Reader";
    }
};

}
}
