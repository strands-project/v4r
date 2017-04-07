#include <v4r/features/global_concatenated.h>

namespace v4r
{
template<typename PointT>
GlobalConcatEstimator<PointT>::GlobalConcatEstimator(
        std::vector<std::string> &boost_command_line_arguments,
        const GlobalConcatEstimatorParameter &p
        )
    :
      need_normals_(false),
      param_(p)
{
    descr_name_ = "global";
    feature_dimensions_ = 0;

    if(param_.feature_type & FeatureType::ESF)
    {
        esf_estimator_.reset( new ESFEstimation<PointT> );
        descr_name_ += "_" + esf_estimator_->getFeatureDescriptorName();
        feature_dimensions_ += esf_estimator_->getFeatureDimensions();
        need_normals_ |= esf_estimator_->needNormals();
    }
    if(param_.feature_type & FeatureType::SIMPLE_SHAPE)
    {
        simple_shape_estimator_.reset(new SimpleShapeEstimator<PointT>);
        descr_name_ += "_" + simple_shape_estimator_->getFeatureDescriptorName();
        feature_dimensions_ += simple_shape_estimator_->getFeatureDimensions();
        need_normals_ |= simple_shape_estimator_->needNormals();
    }
    if(param_.feature_type & FeatureType::GLOBAL_COLOR)
    {
        GlobalColorEstimatorParameter color_param;
        color_param.init(boost_command_line_arguments);
        color_estimator_.reset( new GlobalColorEstimator<PointT>(color_param) );
        descr_name_ += "_" + color_estimator_->getFeatureDescriptorName();
        feature_dimensions_ += color_estimator_->getFeatureDimensions();
        need_normals_ |= color_estimator_->needNormals();
    }
    if(param_.feature_type & FeatureType::OURCVFH)
    {
        ourcvfh_estimator_.reset (new OURCVFHEstimator<PointT>);
        descr_name_ += "_" + ourcvfh_estimator_->getFeatureDescriptorName();
        feature_dimensions_ += ourcvfh_estimator_->getFeatureDimensions();
        need_normals_ |= ourcvfh_estimator_->needNormals();
    }
#ifdef HAVE_CAFFE
    if(param_.feature_type & FeatureType::ALEXNET)
    {
        CNN_Feat_ExtractorParameter cnnparam;
        cnnparam.init(boost_command_line_arguments);
        cnn_feat_estimator_.reset(new CNN_Feat_Extractor<PointT>(cnnparam) );
        descr_name_ += "_" + cnn_feat_estimator_->getFeatureDescriptorName();
        feature_dimensions_ += cnn_feat_estimator_->getFeatureDimensions();
        need_normals_ |= cnn_feat_estimator_->needNormals();
    }
#endif

    VLOG(1) << "Initialized global concatenated pipeline with " << descr_name_ << " resulting in " << feature_dimensions_ << " feature dimensions.";

    descr_type_ = param_.feature_type;
}

template<typename PointT>
bool
GlobalConcatEstimator<PointT>::compute (Eigen::MatrixXf &signature)
{
    Eigen::MatrixXf signature_esf, signature_simple_shape, signature_color, signature_ourcvfh, signature_cnn;

    int num_signatures = 0;

    if(esf_estimator_)
    {
        esf_estimator_->setInputCloud(cloud_);
        esf_estimator_->setNormals(normals_);
        esf_estimator_->setIndices(indices_);
        esf_estimator_->compute(signature_esf);

        CHECK(num_signatures == 0 || num_signatures == signature_esf.rows() ) << "Cannot concatenate features with a diferent number of feature vectors.";

        num_signatures = signature_esf.rows();
    }

    if(simple_shape_estimator_)
    {
        simple_shape_estimator_->setInputCloud(cloud_);
        simple_shape_estimator_->setNormals(normals_);
        simple_shape_estimator_->setIndices(indices_);
        simple_shape_estimator_->compute(signature_simple_shape);

        CHECK(num_signatures == 0 || num_signatures == signature_simple_shape.rows() ) << "Cannot concatenate features with a diferent number of feature vectors.";

        num_signatures = signature_simple_shape.rows();
    }

    if(color_estimator_)
    {
        color_estimator_->setInputCloud(cloud_);
        color_estimator_->setNormals(normals_);
        color_estimator_->setIndices(indices_);
        color_estimator_->compute(signature_color);

        CHECK(num_signatures == 0 || num_signatures == signature_color.rows() ) << "Cannot concatenate features with a diferent number of feature vectors.";

        num_signatures = signature_color.rows();
    }

    if(ourcvfh_estimator_)
    {
        ourcvfh_estimator_->setInputCloud(cloud_);
        ourcvfh_estimator_->setNormals(normals_);
        ourcvfh_estimator_->setIndices(indices_);
        ourcvfh_estimator_->compute(signature_ourcvfh);
        transforms_ = ourcvfh_estimator_->getTransforms();

        CHECK(num_signatures == 0 || num_signatures == signature_ourcvfh.rows() ) << "Cannot concatenate features with a diferent number of feature vectors.";

        num_signatures = signature_ourcvfh.rows();
    }

#ifdef HAVE_CAFFE
    if(cnn_feat_estimator_)
    {
        cnn_feat_estimator_->setInputCloud(cloud_);
        cnn_feat_estimator_->setNormals(normals_);
        cnn_feat_estimator_->setIndices(indices_);
        cnn_feat_estimator_->compute(signature_cnn);

        CHECK(num_signatures == 0 || num_signatures == signature_cnn.rows() ) << "Cannot concatenate features with a diferent number of feature vectors.";

        num_signatures = signature_cnn.rows();
    }
#endif


//    CHECK(signature_esf.rows() == signature_simple_shape.rows()  && signature_simple_shape.rows() == signature_color.rows() );

    signature = Eigen::MatrixXf ( num_signatures,
                                  signature_esf.cols()+signature_simple_shape.cols()+signature_color.cols()+signature_ourcvfh.cols()+signature_cnn.cols());
    signature << signature_esf, signature_simple_shape, signature_color, signature_ourcvfh, signature_cnn;

    indices_.clear();

    return true;
}

template class V4R_EXPORTS GlobalConcatEstimator<pcl::PointXYZRGB>;
}

