
#include "util/boxFilter.h"

#include <pcl/filters/crop_box.h>

namespace object_modeller
{
namespace util
{

BoxConfig::BoxConfig(BoxFilter *module) : VisualConfigBase(module)
{
    std::cout << "initalized box config" << std::endl;
    this->filter = module;
}

void BoxConfig::startConfig(output::Renderer::Ptr renderer)
{
    renderer->enableRoiMode(&filter->dim, &filter->translation, &filter->rotation);
}


BoxFilter::BoxFilter(std::string config_name) : InOutModule(config_name, new BoxConfig(this))
{
    std::cout << "box filter ctor" << std::endl;

    registerParameter("dimension", "Dimensions", &dim, Eigen::Vector3f(1.0f, 1.0f, 1.0f));
    registerParameter("translation", "Translation", &translation, Eigen::Vector3f(0.0f, 0.0f, 0.0f));
    registerParameter("rotation", "Rotation", &rotation, Eigen::Quaternionf::Identity());
}

std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> BoxFilter::process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointClouds)
{
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> result;

    std::cout << "input cloud size: " << pointClouds.size() << std::endl;

    for (size_t i = 0; i < pointClouds.size (); i++)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud (new pcl::PointCloud<pcl::PointXYZRGB>);

        Eigen::Vector4f max(dim.x() / 2.0, dim.y() / 2.0, dim.z() / 2.0, 0.0);
        Eigen::Vector4f min = -max;

        Eigen::Vector3f rot;
        Eigen::Affine3f aff = Eigen::Affine3f::Identity();
        aff.linear() = rotation.toRotationMatrix();
        pcl::getEulerAngles(aff, rot[0], rot[1], rot[2]);

        pcl::CropBox<pcl::PointXYZRGB> pass;
        pass.setMin(min);
        pass.setMax(max);
        pass.setRotation(rot);
        pass.setTranslation(translation);
        pass.setInputCloud (pointClouds[i]);
        pass.setKeepOrganized (true);
        pass.filter (*pointCloud);

        result.push_back(pointCloud);
    }

    std::cout << "result size: " << result.size() << std::endl;

    return result;
}

}
}
