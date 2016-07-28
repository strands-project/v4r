#include <v4r/common/occlusion_reasoning.h>
#include <pcl/point_types.h>
#include <pcl/common/point_tests.h>
#include <pcl/impl/instantiate.hpp>
#include <glog/logging.h>

namespace v4r
{

template<typename PointTA, typename PointTB>
boost::dynamic_bitset<>
occlusion_reasoning (const pcl::PointCloud<PointTA> & organized_cloud,
                     const pcl::PointCloud<PointTB> & to_be_filtered,
                     const Camera::Ptr cam = Camera(),
                     float threshold = 0.01f,
                     bool is_occluded_out_fov = true)
{
    CHECK(organized_cloud.isOrganized());
    float cx = cam->getCx();
    float cy = cam->getCy();
    float f = cam->getFocalLength();
    size_t width = cam->getWidth();
    size_t height = cam->getHeight();

    boost::dynamic_bitset<> is_occluded (to_be_filtered.points.size(), 0);

    for (size_t i = 0; i < to_be_filtered.points.size (); i++)
    {
        if ( !pcl::isFinite(to_be_filtered.points[i]) )
            continue;

        const float x = to_be_filtered.points[i].x;
        const float y = to_be_filtered.points[i].y;
        const float z = to_be_filtered.points[i].z;
        const int u = static_cast<int> (f * x / z + cx);
        const int v = static_cast<int> (f * y / z + cy);

        // points out of the field of view in the first frame
        if ( (u >= static_cast<int> (width)) || (v >= static_cast<int> (height)) || (u < 0) || (v < 0) )
        {
            is_occluded[i] = is_occluded_out_fov;
            continue;
        }

        // Check for invalid depth
        if ( !pcl::isFinite (organized_cloud.at (u, v)) )
        {
            is_occluded.set(i);
            continue;
        }


        //Check if point depth (distance to camera) is greater than the (u,v)
        if ( ( z - organized_cloud.at(u, v).z ) > threshold )
            is_occluded.set(i);
    }
    return is_occluded;
}

#define PCL_INSTANTIATE_occlusion_reasoning(TA,TB) template V4R_EXPORTS boost::dynamic_bitset<> occlusion_reasoning<TA,TB>(const pcl::PointCloud<TA> &, const pcl::PointCloud<TB> &, const Camera::Ptr, float, bool);
PCL_INSTANTIATE_PRODUCT(occlusion_reasoning, (PCL_XYZ_POINT_TYPES)(PCL_XYZ_POINT_TYPES) )

}
