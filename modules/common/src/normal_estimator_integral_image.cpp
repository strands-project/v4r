#include <v4r/common/normal_estimator_integral_image.h>

#include <pcl/features/integral_image_normal.h>
#include <pcl/impl/instantiate.hpp>

#include <glog/logging.h>

namespace v4r
{

template<typename PointT>
pcl::PointCloud<pcl::Normal>::Ptr
NormalEstimatorIntegralImage<PointT>::compute()
{
    CHECK(input_ && input_->isOrganized());

    normal_.reset(new pcl::PointCloud<pcl::Normal>);
    normal_->points.resize(input_->height * input_->width);
    normal_->height = input_->height;
    normal_->width = input_->width;

    pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
    ne.setNormalEstimationMethod ( ne.COVARIANCE_MATRIX );
    ne.setMaxDepthChangeFactor( param_.max_depth_change_factor_ );
    ne.setNormalSmoothingSize( param_.smoothing_size_ );
    ne.setDepthDependentSmoothing( param_.use_depth_depended_smoothing_ );
    ne.setInputCloud( input_ );
    ne.compute( *normal_) ;

    return normal_;
}


#define PCL_INSTANTIATE_NormalEstimatorIntegralImage(T) template class V4R_EXPORTS NormalEstimatorIntegralImage<T>;
PCL_INSTANTIATE(NormalEstimatorIntegralImage, PCL_XYZ_POINT_TYPES )

}
