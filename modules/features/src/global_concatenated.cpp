#include <v4r/features/global_concatenated.h>
#include <v4r/features/esf_estimator.h>
#include <v4r/features/global_simple_shape_estimator.h>
#include <v4r/features/global_color_estimator.h>
#include <v4r/features/ourcvfh_estimator.h>

namespace v4r
{

template<typename PointT>
bool
GlobalConcatEstimator<PointT>::needNormals() const
{
    if(param_.feature_type & FeatureType::ESF)
    {
        ESFEstimation<PointT> esf;
        if (esf.needNormals())
            return true;
    }

    if(param_.feature_type & FeatureType::SIMPLE_SHAPE)
    {
        SimpleShapeEstimator<PointT> sse;
        if (sse.needNormals())
            return true;
    }

    if(param_.feature_type & FeatureType::GLOBAL_COLOR)
    {
        GlobalColorEstimator<PointT> color_e;
        if(color_e.needNormals())
            return true;
    }

    if(param_.feature_type & FeatureType::OURCVFH)
    {
        OURCVFHEstimator<PointT> ourcvfh;
        if(ourcvfh.needNormals())
            return true;
    }

    return false;
}

template<typename PointT>
bool
GlobalConcatEstimator<PointT>::compute (Eigen::MatrixXf &signature)
{
    Eigen::MatrixXf signature_esf, signature_simple_shape, signature_color, signature_ourcvfh;

    int num_signatures = 0;

    if(param_.feature_type & FeatureType::ESF)
    {
        ESFEstimation<PointT> esf;
        esf.setInputCloud(cloud_);
        esf.setNormals(normals_);
        esf.setIndices(indices_);
        esf.compute(signature_esf);

        CHECK(num_signatures == 0 || num_signatures == signature_esf.rows() ) << "Cannot concatenate features with a diferent number of feature vectors.";

        num_signatures = signature_esf.rows();
    }

    if(param_.feature_type & FeatureType::SIMPLE_SHAPE)
    {
        SimpleShapeEstimator<PointT> sse;
        sse.setInputCloud(cloud_);
        sse.setNormals(normals_);
        sse.setIndices(indices_);
        sse.compute(signature_simple_shape);

        CHECK(num_signatures == 0 || num_signatures == signature_simple_shape.rows() ) << "Cannot concatenate features with a diferent number of feature vectors.";

        num_signatures = signature_simple_shape.rows();
    }

    if(param_.feature_type & FeatureType::GLOBAL_COLOR)
    {
        GlobalColorEstimator<PointT> color_e;
        color_e.setInputCloud(cloud_);
        color_e.setNormals(normals_);
        color_e.setIndices(indices_);
        color_e.compute(signature_color);

        CHECK(num_signatures == 0 || num_signatures == signature_color.rows() ) << "Cannot concatenate features with a diferent number of feature vectors.";

        num_signatures = signature_color.rows();
    }

    if(param_.feature_type & FeatureType::OURCVFH)
    {
        OURCVFHEstimator<PointT> ourcvfh;
        ourcvfh.setInputCloud(cloud_);
        ourcvfh.setNormals(normals_);
        ourcvfh.setIndices(indices_);
        ourcvfh.compute(signature_ourcvfh);
        transforms_ = ourcvfh.getTransforms();

        CHECK(num_signatures == 0 || num_signatures == signature_ourcvfh.rows() ) << "Cannot concatenate features with a diferent number of feature vectors.";

        num_signatures = signature_ourcvfh.rows();
    }


//    CHECK(signature_esf.rows() == signature_simple_shape.rows()  && signature_simple_shape.rows() == signature_color.rows() );

    signature = Eigen::MatrixXf ( num_signatures,
                                  signature_esf.cols()+signature_simple_shape.cols()+signature_color.cols()+signature_ourcvfh.cols());
    signature << signature_esf, signature_simple_shape, signature_color, signature_ourcvfh;

    indices_.clear();

    return true;
}

template class V4R_EXPORTS GlobalConcatEstimator<pcl::PointXYZRGB>;
}

