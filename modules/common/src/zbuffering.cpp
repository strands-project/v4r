#include <v4r/common/zbuffering.h>
#include <v4r/common/miscellaneous.h>
#include <glog/logging.h>
#include <omp.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/impl/instantiate.hpp>

namespace v4r
{

template<typename PointT>
void
ZBuffering<PointT>::renderPointCloud(const pcl::PointCloud<PointT> &cloud, pcl::PointCloud<PointT> & rendered_view)
{
    float cx = cam_->getCx();
    float cy = cam_->getCy();
    float f = cam_->getFocalLength();
    size_t width = cam_->getWidth();
    size_t height = cam_->getHeight();

    rendered_view.points.resize( width * height );
    rendered_view.width = width;
    rendered_view.height = height;
    rendered_view.is_dense = false;

    index_map_ = -1 * Eigen::MatrixXi::Ones( height, width );

#pragma omp parallel for
    for (size_t i=0; i< width * height ; i++)    // initialize points to infinity
        rendered_view.points[i].x = rendered_view.points[i].y = rendered_view.points[i].z = std::numeric_limits<float>::quiet_NaN();

    std::vector<omp_lock_t> pt_locks (width * height);
    for(size_t i=0; i<pt_locks.size(); i++)
        omp_init_lock(&pt_locks[i]);

    cv::Mat registration_depth_mask = cam_->getCameraDepthRegistrationMask();

#pragma omp parallel for schedule (dynamic)
    for (int i=0; i< static_cast<int>(cloud.points.size()); i++)
    {
        const PointT &pt = cloud.points[i];
        int u = f * pt.x / pt.z + cx;
        int v = f * pt.y / pt.z + cy;

        if (u >= (int)width || v >= (int)height  || u < 0 || v < 0)
            continue;

        if( !registration_depth_mask.at<uchar>(v,u) )
            continue;

        int idx = v * width + u;

        omp_set_lock(&pt_locks[idx]);
        PointT &r_pt = rendered_view.points[idx];

        if ( !pcl_isfinite( r_pt.z ) || (pt.z < r_pt.z) )
        {
            r_pt = pt;
            index_map_(v,u) = i;
        }

        omp_unset_lock(&pt_locks[idx]);
    }

    for(size_t i=0; i<pt_locks.size(); i++)
        omp_destroy_lock(&pt_locks[i]);


    if (param_.do_smoothing_)
    {
        pcl::PointCloud<PointT> rendered_view_unsmooth = rendered_view;

        for (int u = param_.smoothing_radius_; u < ((int)width - param_.smoothing_radius_); u++)
        {
            for (int v = param_.smoothing_radius_; v < ((int)height - param_.smoothing_radius_); v++)
            {
                float min = std::numeric_limits<float>::max();
                int min_uu = u, min_vv = v;
                for (int uu = (u - param_.smoothing_radius_); uu <= (u + param_.smoothing_radius_); uu++)
                {
                    for (int vv = (v - param_.smoothing_radius_); vv <= (v + param_.smoothing_radius_); vv++)
                    {
                        if( uu<0 || vv<0 || uu>=(int)width || vv>=(int)height)    // this shouldn't happen anyway
                            continue;

                        PointT &p = rendered_view_unsmooth.at(uu,vv);
                        if ( !pcl_isfinite(min) || (pcl::isFinite(p) && ( p.z < min)) )
                        {
                            min = p.z;
                            min_uu = uu;
                            min_vv = vv;
                        }
                    }
                }

                rendered_view.at(u,v) = rendered_view_unsmooth.at(min_uu, min_vv);
                index_map_(v,u) = index_map_(min_vv, min_uu);   ///NOTE: Be careful, this is maybe not what you want to get!
            }
        }
    }

    if (param_.do_noise_filtering_)
    {
        pcl::PointCloud<PointT> rendered_view_filtered = rendered_view;

        for (int u = param_.smoothing_radius_; u < ((int)width - param_.smoothing_radius_); u++)
        {
            for (int v = param_.smoothing_radius_; v < ((int)height - param_.smoothing_radius_); v++)
            {
                PointT &p = rendered_view_filtered.at(u,v);
                bool is_noise = true;
                for (int uu = (u - param_.smoothing_radius_); uu <= (u + param_.smoothing_radius_) && is_noise; uu++)
                {
                    for (int vv = (v - param_.smoothing_radius_); vv <= (v + param_.smoothing_radius_); vv++)
                    {
                        if( uu<0 || vv<0 || uu>=(int)width || vv>=(int)height)    // this shouldn't happen anyway
                            continue;

                        PointT &p_tmp = rendered_view.at(uu,vv);
                        if ( std::abs(p.z - p_tmp.z) < param_.inlier_threshold_ )
                        {
                            is_noise=false;
                            break;
                        }
                    }
                }
                if(is_noise)
                {
                    p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
                    index_map_(v,u) = -1;
                }
           }
        }
        rendered_view = rendered_view_filtered;
    }
}

#define PCL_INSTANTIATE_ZBuffering(T) template class V4R_EXPORTS ZBuffering<T>;
PCL_INSTANTIATE(ZBuffering, PCL_XYZ_POINT_TYPES )

}


