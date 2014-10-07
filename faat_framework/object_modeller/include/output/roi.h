#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/visualization/pcl_visualizer.h>

namespace object_modeller
{
namespace output
{

class Roi
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;

    Eigen::Vector3f *dimension;
    Eigen::Vector3f *translation;
    Eigen::Quaternionf *rotation;

    Eigen::Vector3f originalPoint;
    Eigen::Vector3f originalTranslation;
    Eigen::Quaternionf originalRotation;

    // points
    pcl::PointXYZRGB *selectedPoint;
    pcl::PointXYZRGB *highlightPoint;

    pcl::PointXYZRGB *translationPoint;
    pcl::PointXYZRGB *scalePointX;
    pcl::PointXYZRGB *scalePointY;
    pcl::PointXYZRGB *scalePointZ;
    pcl::PointXYZRGB *rotPointX;
    pcl::PointXYZRGB *rotPointY;
    pcl::PointXYZRGB *rotPointZ;
public:
    Roi(Eigen::Vector3f *dimension, Eigen::Vector3f *translation, Eigen::Quaternionf *rotation);

    bool isPointSelected()
    {
        return selectedPoint != NULL;
    }

    void selectPoint(pcl::visualization::Camera camera, int screen_x, int screen_y);

    void highlightSelection()
    {
        removeHighlight();

        highlightPoint = selectedPoint;

        highlightPoint->r += 50;
        highlightPoint->g += 50;
        highlightPoint->b += 50;
    }

    bool removeHighlight()
    {
        if (highlightPoint != NULL)
        {
            highlightPoint->r -= 50;
            highlightPoint->g -= 50;
            highlightPoint->b -= 50;
            highlightPoint = NULL;
            return true;
        }

        return false;
    }

    void deselect()
    {
        selectedPoint = NULL;
    }

    void handleMouseMove(pcl::visualization::Camera camera, int screen_x, int screen_y);
    void handleTranslation(pcl::visualization::Camera camera, Eigen::Vector3f ray_start, Eigen::Vector3f ray_end);
    void handleScale(pcl::visualization::Camera camera, Eigen::Vector3f ray_start, Eigen::Vector3f ray_end, pcl::PointXYZRGB *scalePoint);
    void handleRotation(pcl::visualization::Camera camera, Eigen::Vector3f ray_start, Eigen::Vector3f ray_end, pcl::PointXYZRGB *rotPoint);

    void updateCube();

    void updateScalePoint(pcl::PointXYZRGB *scalePoint);
    void updateRotationPoint(pcl::PointXYZRGB *rotPoint);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr getCloud()
    {
        return cloud;
    }

    Eigen::Vector3f *getDimension()
    {
        return dimension;
    }

    Eigen::Vector3f *getTranslation()
    {
        return translation;
    }

    Eigen::Quaternionf *getRotation()
    {
        return rotation;
    }
};

}
}
