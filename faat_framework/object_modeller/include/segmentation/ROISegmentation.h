
#include "ioModule.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace object_modeller
{
namespace segmentation
{

class ROISegmentation :
        public InOutModule<std::vector<std::vector<int> >,
                           std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> >
{

    Eigen::Vector4f min_, max_; //min and max values
    Eigen::Affine3f transformation_;

public:
    ROISegmentation(std::string config_name="ROISegmentation");

    std::vector<std::vector<int> > process(std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> input);

    virtual void applyConfig(Config &config);

    std::string getName()
    {
        return "ROISegmentation";
    }

    void setMinMax(Eigen::Vector4f & min, Eigen::Vector4f & max)
    {
        min_ = min;
        max_ = max;
    }

    void setTransformation(Eigen::Affine3f & transform)
    {
        transformation_ = transform;
    }

    void processSingle(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud, std::vector<int> & indices);

};

}
}
