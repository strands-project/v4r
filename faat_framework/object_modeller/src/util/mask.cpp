
#include "util/mask.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

namespace object_modeller
{
namespace util
{

template <class TPointType>
std::vector<typename pcl::PointCloud<TPointType>::Ptr> Mask<TPointType>::process(boost::tuples::tuple<
             std::vector<typename pcl::PointCloud<TPointType>::Ptr>,
             std::vector<std::vector<int> > > input)
{
    std::vector<typename pcl::PointCloud<TPointType>::Ptr> pointClouds = boost::tuples::get<0>(input);
    std::vector<std::vector<int> > indices = boost::tuples::get<1>(input);

    std::vector<typename pcl::PointCloud<TPointType>::Ptr> result;

    for (size_t i = 0; i < pointClouds.size (); i++)
    {
        typename pcl::PointCloud<TPointType>::Ptr pointCloud (new typename pcl::PointCloud<TPointType>);

        pcl::copyPointCloud(*pointClouds[i], indices[i], *pointCloud);

        result.push_back(pointCloud);
    }

    return result;
}

template class Mask<pcl::PointXYZRGB>;
template class Mask<pcl::Normal>;

}
}
