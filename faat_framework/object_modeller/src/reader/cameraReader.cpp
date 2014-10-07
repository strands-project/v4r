
#include "reader/cameraReader.h"

#include <pcl/io/pcd_io.h>

namespace object_modeller
{
namespace reader
{

void
CameraReader::grabberCallback(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr& cloud)
{
    if (copyFrame)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr deep_copy (new pcl::PointCloud<pcl::PointXYZRGB>() );
        pcl::copyPointCloud(*cloud, *deep_copy);

        result.push_back(deep_copy);
        copyFrame = false;

        renderer->addPointCloud(activeSequence, deep_copy);

        renderer->trigger(EventManager::UPDATE_RENDERER);

        std::cout << "frame added" << std::endl;
    }
}

void CameraReader::image_callback(const boost::shared_ptr<openni_wrapper::Image>& img)
{
    //std::cout << "imagecallback" << std::endl;

    unsigned char *rgb_buffer = new unsigned char[ img->getWidth() * img->getHeight() * 3];
    img->fillRGB(img->getWidth(), img->getHeight(), rgb_buffer);

    /*
    for (int i=0;i<img->getWidth() * img->getHeight();i++) {
        unsigned char temp = rgb_buffer[i*3];
        rgb_buffer[i*3] = rgb_buffer[i*3+2];
        rgb_buffer[i*3+2] = temp;
    }
    */

    object_modeller::output::ImageData *data = new object_modeller::output::ImageData();
    data->data = rgb_buffer;
    data->width = img->getWidth();
    data->height = img->getHeight();

    renderer->addImage(data);
    renderer->trigger(EventManager::UPDATE_IMAGE);
}

std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> CameraReader::process()
{
    result.clear();

    if (grabber == NULL)
    {
        grabber = new pcl::OpenNIGrabber();

        boost::function<void (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> f =
                    boost::bind(&CameraReader::grabberCallback, this, _1);
        grabber->registerCallback(f);

        boost::function < void (const boost::shared_ptr<openni_wrapper::Image>&)> f2 =
                    boost::bind (&CameraReader::image_callback, this, _1);
        grabber->registerCallback(f2);
    }

    grabber->start();

    renderer->addAvailableEvent(EventManager::ADD_SEQUENCE);
    renderer->addAvailableEvent(EventManager::GRAB_FRAME);
    renderer->addAvailableEvent(EventManager::CONTINUE);

    renderer->trigger(EventManager::UPDATE_RENDERCONTROLS);

    while (true) {
        EventManager::Event e = renderer->consumeBlocking();

        if (e == EventManager::GRAB_FRAME)
        {
            copyFrame = true;
        }
        if (e == EventManager::ADD_SEQUENCE)
        {
            nrSequences++;

            renderer->removeAvailableEvent(EventManager::ADD_SEQUENCE);
            renderer->removeAvailableEvent(EventManager::GRAB_FRAME);
            renderer->removeAvailableEvent(EventManager::CONTINUE);

            renderer->trigger(EventManager::UPDATE_RENDERCONTROLS);

            grabber->stop();

            //renderer->trigger(EventManager::UPDATE_RENDERER);

            return result;
        }
        if (e == EventManager::CONTINUE || e == EventManager::CLOSE)
        {
            renderer->removeAvailableEvent(EventManager::ADD_SEQUENCE);
            renderer->removeAvailableEvent(EventManager::GRAB_FRAME);
            renderer->removeAvailableEvent(EventManager::CONTINUE);

            renderer->trigger(EventManager::UPDATE_RENDERCONTROLS);

            grabber->stop();
            delete grabber;
            grabber = NULL;

            return result;
        }
    }
}

}
}
