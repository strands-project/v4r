#include "v4r/registration/MultiSessionModelling.h"

template<class PointT>
size_t
v4r::Registration::PartialModelRegistrationBase<PointT>::getTotalNumberOfClouds()
{
    return msm_->getTotalNumberOfClouds();
}

template<class PointT>
typename pcl::PointCloud<PointT>::Ptr
v4r::Registration::PartialModelRegistrationBase<PointT>::getCloud(size_t i)
{
    return msm_->getCloud(i);
}

template<class PointT>
std::vector<int> &
v4r::Registration::PartialModelRegistrationBase<PointT>::getIndices(size_t i)
{
    return msm_->getIndices(i);
}

template<class PointT>
Eigen::Matrix4f
v4r::Registration::PartialModelRegistrationBase<PointT>::getPose(size_t i)
{
    return msm_->getPose(i);
}

template<class PointT>
pcl::PointCloud<pcl::Normal>::Ptr
v4r::Registration::PartialModelRegistrationBase<PointT>::getNormal(size_t i)
{
    return msm_->getNormal(i);
}

template class V4R_EXPORTS v4r::Registration::PartialModelRegistrationBase<pcl::PointXYZRGB>;

