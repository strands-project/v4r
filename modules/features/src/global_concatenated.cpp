#include <v4r/features/global_concatenated.h>
#include <v4r/features/esf_estimator.h>
#include <v4r/features/global_simple_shape_estimator.h>

namespace v4r
{

template<typename PointT>
bool
GlobalConcatEstimator<PointT>::compute (Eigen::MatrixXf &signature)
{
    Eigen::MatrixXf signature_esf, signature_simple_shape;

    ESFEstimation<PointT> esf;
    esf.setInputCloud(cloud_);
    esf.setIndices(indices_);
    esf.compute(signature_esf);

    SimpleShapeEstimator<PointT> sse;
    sse.setInputCloud(cloud_);
    sse.setIndices(indices_);
    sse.compute(signature_simple_shape);

    CHECK(signature_esf.rows() == signature_simple_shape.rows() );

    signature = Eigen::MatrixXf ( signature_esf.rows(), signature_esf.cols()+signature_simple_shape.cols());
    signature << signature_esf, signature_simple_shape;
    indices_.clear();

    return true;
}

template class V4R_EXPORTS GlobalConcatEstimator<pcl::PointXYZ>;
template class V4R_EXPORTS GlobalConcatEstimator<pcl::PointXYZRGB>;
}

