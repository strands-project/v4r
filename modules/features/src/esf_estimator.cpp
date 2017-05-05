#include <v4r/features/esf_estimator.h>
#include <pcl/features/esf.h>

namespace v4r
{
template<typename PointT>
bool
ESFEstimation<PointT>::compute (Eigen::MatrixXf &signature)
{
    CHECK(cloud_ && !cloud_->points.empty());
    typename pcl::ESFEstimation<PointT, pcl::ESFSignature640> esf;
    pcl::PointCloud<pcl::ESFSignature640> ESF_signature;

    if(!indices_.empty())   /// NOTE: setIndices does not seem to work for ESF
    {
        typename pcl::PointCloud<PointT>::Ptr cloud_roi (new pcl::PointCloud<PointT>);
        pcl::copyPointCloud(*cloud_, indices_, *cloud_roi);
        esf.setInputCloud(cloud_roi);
    }
    else
        esf.setInputCloud (cloud_);

    esf.compute (ESF_signature);
    signature.resize(ESF_signature.points.size(), feature_dimensions_);

    for(size_t pt=0; pt<ESF_signature.points.size(); pt++)
    {
        for(size_t i=0; i<feature_dimensions_; i++)
            signature(pt, i) = ESF_signature.points[pt].histogram[i];
    }

    indices_.clear();

    return true;
}
}

template class V4R_EXPORTS v4r::ESFEstimation<pcl::PointXYZ>;
template class V4R_EXPORTS v4r::ESFEstimation<pcl::PointXYZRGB>;
