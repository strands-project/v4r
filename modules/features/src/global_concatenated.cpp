#include <v4r/features/global_concatenated.h>
#include <v4r/features/esf_estimator.h>
#include <v4r/features/global_simple_shape_estimator.h>
#include <v4r/features/global_color_estimator.h>

namespace v4r
{

template<typename PointT>
bool
GlobalConcatEstimator<PointT>::compute (Eigen::MatrixXf &signature)
{
    Eigen::MatrixXf signature_esf, signature_simple_shape, signature_color;

    if(param_.feature_type & FeatureType::ESF)
    {
        ESFEstimation<PointT> esf;
        esf.setInputCloud(cloud_);
        esf.setIndices(indices_);
        esf.compute(signature_esf);
    }

    if(param_.feature_type & FeatureType::SIMPLE_SHAPE)
    {
        SimpleShapeEstimator<PointT> sse;
        sse.setInputCloud(cloud_);
        sse.setIndices(indices_);
        sse.compute(signature_simple_shape);
    }

    if(param_.feature_type & FeatureType::GLOBAL_COLOR)
    {
        GlobalColorEstimator<PointT> color_e;
        color_e.setInputCloud(cloud_);
        color_e.setIndices(indices_);
        color_e.compute(signature_color);
    }


//    CHECK(signature_esf.rows() == signature_simple_shape.rows()  && signature_simple_shape.rows() == signature_color.rows() );

    signature = Eigen::MatrixXf ( signature_esf.rows(), signature_esf.cols()+signature_simple_shape.cols()+signature_color.cols());
    signature << signature_esf, signature_simple_shape, signature_color;

    indices_.clear();

    return true;
}

template class V4R_EXPORTS GlobalConcatEstimator<pcl::PointXYZRGB>;
}

