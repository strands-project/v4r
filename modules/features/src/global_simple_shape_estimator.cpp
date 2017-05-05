#include <v4r/features/global_simple_shape_estimator.h>
#include <v4r/common/miscellaneous.h>

namespace v4r
{
template<typename PointT>
bool
SimpleShapeEstimator<PointT>::compute (Eigen::MatrixXf &signature)
{
    CHECK(cloud_ && !cloud_->points.empty());

    Eigen::Vector4f centroid;
    Eigen::Vector3f elongationsXYZ;
    Eigen::Matrix4f eigenBasis;

    v4r::computePointCloudProperties( *cloud_, centroid, elongationsXYZ, eigenBasis, indices_);

    signature = Eigen::MatrixXf(1, feature_dimensions_);
    signature.row(0) = elongationsXYZ;
    indices_.clear();

    return true;
}

template class V4R_EXPORTS SimpleShapeEstimator<pcl::PointXYZ>;
template class V4R_EXPORTS SimpleShapeEstimator<pcl::PointXYZRGB>;
}

