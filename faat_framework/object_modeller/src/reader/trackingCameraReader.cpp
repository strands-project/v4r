
#include "reader/trackingCameraReader.h"

#include <pcl/io/pcd_io.h>

namespace object_modeller
{
namespace reader
{

void
TrackingCameraReader::grabberCallback(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr& cloud)
{
    if (m_trackerEnabled)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr deep_copy (new pcl::PointCloud<pcl::PointXYZRGB>() );
        pcl::copyPointCloud(*cloud, *deep_copy);

        if (last_keyframe == NULL)
        {
            std::cout << "set first frame" << std::endl;
            last_keyframe = deep_copy;
        }
        else
        {
            std::cout << "track frame" << std::endl;
            bool is_keyframe;
            bool track_ok = tracker->trackSingle(last_keyframe, deep_copy, is_keyframe);

            if (!track_ok)
            {
                renderer->trigger(EventManager::END_RECORDING);
            }

            if (is_keyframe)
            {
                std::cout << "keyframe added" << std::endl;
                last_keyframe = deep_copy;
                /*
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr deep_copy (new pcl::PointCloud<pcl::PointXYZRGB>() );
                pcl::copyPointCloud(*cloud, *deep_copy);

                result.push_back(deep_copy);
                copyFrame = false;

                std::cout << "frame added" << std::endl;
                */

                result.push_back(deep_copy);
                renderer->addPointCloud(activeSequence, deep_copy);

                renderer->trigger(EventManager::UPDATE_RENDERER);
            }

            last_keyframe = deep_copy;
        }
    }
}

void TrackingCameraReader::image_callback(const boost::shared_ptr<openni_wrapper::Image>& img)
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

std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> TrackingCameraReader::process()
{
    if (grabber == NULL)
    {
        grabber = new pcl::OpenNIGrabber();

        boost::function<void (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> f =
                    boost::bind(&TrackingCameraReader::grabberCallback, this, _1);
        grabber->registerCallback(f);

        boost::function < void (const boost::shared_ptr<openni_wrapper::Image>&)> f2 =
                    boost::bind (&TrackingCameraReader::image_callback, this, _1);
        grabber->registerCallback(f2);
    }

    grabber->start();

    renderer->addAvailableEvent(EventManager::START_RECORDING);
    renderer->trigger(EventManager::UPDATE_RENDERCONTROLS);

    while (true) {
        EventManager::Event e = renderer->consumeBlocking();

        if (e == EventManager::START_RECORDING)
        {
            renderer->removeAvailableEvent(EventManager::START_RECORDING);
            renderer->addAvailableEvent(EventManager::END_RECORDING);
            renderer->trigger(EventManager::UPDATE_RENDERCONTROLS);
            m_trackerEnabled = true;
        }

        if (e == EventManager::END_RECORDING)
        {
            renderer->addAvailableEvent(EventManager::CONTINUE);
            renderer->addAvailableEvent(EventManager::ADD_SEQUENCE);
            renderer->removeAvailableEvent(EventManager::END_RECORDING);
            renderer->trigger(EventManager::UPDATE_RENDERCONTROLS);
            m_trackerEnabled = false;
        }

        if (e == EventManager::ADD_SEQUENCE)
        {
            nrSequences++;

            renderer->removeAvailableEvent(EventManager::START_RECORDING);
            renderer->removeAvailableEvent(EventManager::ADD_SEQUENCE);
            renderer->removeAvailableEvent(EventManager::CONTINUE);

            renderer->trigger(EventManager::UPDATE_RENDERCONTROLS);

            grabber->stop();

            return result;
        }

        if (e == EventManager::CONTINUE)
        {
            renderer->removeAvailableEvent(EventManager::START_RECORDING);
            renderer->removeAvailableEvent(EventManager::ADD_SEQUENCE);
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
