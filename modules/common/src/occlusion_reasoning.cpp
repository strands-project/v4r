#include <glog/logging.h>
#include <pcl/point_types.h>
#include <pcl/common/point_tests.h>
#include <pcl/impl/instantiate.hpp>
#include <pcl/visualization/pcl_visualizer.h>

#include <v4r/common/occlusion_reasoning.h>
#include <v4r/common/zbuffering.h>


namespace v4r
{
template<typename PointTA, typename PointTB>
boost::dynamic_bitset<>
OcclusionReasoner<PointTA, PointTB>::computeVisiblePoints()
{
    CHECK( occluder_cloud_ && cloud_to_be_filtered_ && ( (occluder_cloud_->isOrganized() && cloud_to_be_filtered_->isOrganized() ) || cam_));

    if( !occluder_cloud_->isOrganized() )
    {
        VLOG(1) << "Occluder not organized. Doing z-buffering";
        ZBuffering<PointTA> zbuf (cam_);
        typename pcl::PointCloud<PointTA>::Ptr organized_occlusion_cloud (new pcl::PointCloud<PointTA>);
        zbuf.renderPointCloud( *occluder_cloud_, *organized_occlusion_cloud);
        occluder_cloud_ = organized_occlusion_cloud;
    }

    Eigen::MatrixXi index_map;
    boost::dynamic_bitset<> mask(cloud_to_be_filtered_->points.size(), 0);

    if( !cloud_to_be_filtered_->isOrganized() )
    {
        VLOG(1) << "Cloud to be filtered is not organized. Doing z-buffering";
        ZBufferingParameter zBparam;
        zBparam.do_noise_filtering_ = false;
        zBparam.do_smoothing_ = false;
        zBparam.inlier_threshold_ = 0.015f;
        ZBuffering<PointTB> zbuf (cam_, zBparam);
        typename pcl::PointCloud<PointTB>::Ptr organized_cloud_to_be_filtered (new pcl::PointCloud<PointTB>);
        zbuf.renderPointCloud( *cloud_to_be_filtered_, *organized_cloud_to_be_filtered );
        cloud_to_be_filtered_ = organized_cloud_to_be_filtered;
//        pcl::visualization::PCLVisualizer vis;
//        vis.addPointCloud(organized_cloud_to_be_filtered);
//        vis.spin();
        index_map = zbuf.getIndexMap();

//        size_t check1=0;
//        for(int u=0; u<index_map.cols(); u++)
//            for (int v=0; v<index_map.rows(); v++)
//                if (index_map(v,u)>=0)
//                    check1++;

//        size_t check2=0;
//        for(PointTB p: cloud_to_be_filtered_->points)
//            if(pcl::isFinite(p) ) check2++;

//        std::cout << "check " << check1 << " " << check2;
    }

    CHECK (occluder_cloud_->width == cloud_to_be_filtered_->width && occluder_cloud_->height == cloud_to_be_filtered_->height)
            << "Occlusion cloud and the cloud that is filtered need to have the same image size!";

    px_is_visible_ = boost::dynamic_bitset<> (occluder_cloud_->points.size(), 0);

    for (size_t u=0; u<cloud_to_be_filtered_->width; u++)
    {
        for (size_t v=0; v<cloud_to_be_filtered_->height; v++)
        {
            const PointTB &pt = cloud_to_be_filtered_->at(u,v);

            if ( !pcl::isFinite(pt) )
                continue;

            const PointTA &pt_occ = occluder_cloud_->at(u,v);

            if( !pcl::isFinite(pt_occ) || ( pt.z - pt_occ.z ) < occlusion_threshold_m_ )
            {
                int idx;
                index_map.size() ? idx = index_map(v,u) : idx = v*cloud_to_be_filtered_->width + u;
                mask.set(idx);

                px_is_visible_.set( v*cloud_to_be_filtered_->width + u );
            }
        }
    }
//    std::cout << "Mask size: " << mask.count() << std::endl;
    return mask;
}

#define PCL_INSTANTIATE_OcclusionReasoner(TA,TB) template class V4R_EXPORTS OcclusionReasoner<TA,TB>;
PCL_INSTANTIATE_PRODUCT(OcclusionReasoner, (PCL_XYZ_POINT_TYPES)(PCL_XYZ_POINT_TYPES))

}
