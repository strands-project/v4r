#include <v4r/registration/metrics.h>
#include <v4r/registration/fast_icp_with_gc.h>
#include <pcl/filters/passthrough.h>

namespace v4r
{

template<typename PointT>
void
calcEdgeWeightAndRefineTf (const typename pcl::PointCloud<PointT>::ConstPtr &cloud_src,
                                                        const typename pcl::PointCloud<PointT>::ConstPtr &cloud_dst,
                                                        const Eigen::Matrix4f &transform,
                                                        float &registration_quality,
                                                        Eigen::Matrix4f &refined_transform)
{
    typename pcl::PointCloud<PointT>::Ptr cloud_src_wo_nan ( new pcl::PointCloud<PointT>());
    typename pcl::PointCloud<PointT>::Ptr cloud_dst_wo_nan ( new pcl::PointCloud<PointT>());

    pcl::PassThrough<PointT> pass;
    pass.setFilterLimits (0.f, 5.f);
    pass.setFilterFieldName ("z");
    pass.setInputCloud (cloud_src);
    pass.setKeepOrganized (true);
    pass.filter (*cloud_src_wo_nan);

    pcl::PassThrough<PointT> pass2;
    pass2.setFilterLimits (0.f, 5.f);
    pass2.setFilterFieldName ("z");
    pass2.setInputCloud (cloud_dst);
    pass2.setKeepOrganized (true);
    pass2.filter (*cloud_dst_wo_nan);

    float w_after_icp_ = std::numeric_limits<float>::max ();
    const float best_overlap_ = 0.75f;

    FastIterativeClosestPointWithGC<PointT> icp;
    icp.setMaxCorrespondenceDistance ( 0.02f );
    icp.setInputSource ( cloud_src_wo_nan );
    icp.setInputTarget ( cloud_dst_wo_nan );
    icp.setUseNormals (true);
    icp.useStandardCG (true);
    icp.setNoCG(true);
    icp.setOverlapPercentage (best_overlap_);
    icp.setKeepMaxHypotheses (5);
    icp.setMaximumIterations (10);
    icp.align (transform);
    w_after_icp_ = icp.getFinalTransformation ( refined_transform );

    if ( w_after_icp_ < 0 || !pcl_isfinite ( w_after_icp_ ) )
        w_after_icp_ = std::numeric_limits<float>::max ();
    else
        w_after_icp_ = best_overlap_ - w_after_icp_;

    //    transform = icp_trans; // refined transformation
    registration_quality = w_after_icp_;
}

template V4R_EXPORTS void
calcEdgeWeightAndRefineTf<pcl::PointXYZRGB> (const typename pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &,
                           const typename pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &,
                           const Eigen::Matrix4f &,
                           float &,
                           Eigen::Matrix4f &);

}
