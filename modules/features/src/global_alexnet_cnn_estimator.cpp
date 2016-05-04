#include <v4r/features/global_alexnet_cnn_estimator.h>

#include <stdio.h>  // for snprintf
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <google/protobuf/text_format.h>

#include <caffe/blob.hpp>
#include <caffe/common.hpp>
//#include <caffe/proto/caffe.pb.h>
#include <caffe/util/db.hpp>
#include <caffe/util/io.hpp>
#include <caffe/layer.hpp>
#include <caffe/layers/memory_data_layer.hpp>

#include <v4r/common/pcl_opencv.h>

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using std::string;
namespace db = caffe::db;

namespace v4r
{

template<typename PointT, typename Dtype>
bool
CNN_Feat_Extractor<PointT, Dtype>::compute (Eigen::MatrixXf &signature)
{
    CHECK( cloud_ && cloud_->isOrganized() );

    if(!init_)
    {
        if (strcmp(param_.device_name_.c_str(), "GPU") == 0) {
          Caffe::SetDevice(param_.device_id_);
          Caffe::set_mode(Caffe::GPU);
        }
        else
          Caffe::set_mode(Caffe::CPU);

        feature_extraction_net_.reset(new Net<Dtype>(param_.feature_extraction_proto_, caffe::TEST));
        feature_extraction_net_->CopyTrainedLayersFrom(param_.pretrained_binary_proto_);

        size_t num_features = param_.extract_feature_blob_names_.size();

        for (size_t i = 0; i < num_features; i++) {
          CHECK(feature_extraction_net_->has_blob(param_.extract_feature_blob_names_[i]))
              << "Unknown feature blob name " << param_.extract_feature_blob_names_[i]
              << " in the network " << param_.feature_extraction_proto_;
        }

        init_ = true;
    }

    CHECK(cloud_ && !cloud_->points.empty() && !indices_.empty());

    cv::Mat img = v4r::ConvertPCLCloud2FixedSizeImage(*cloud_, indices_, param_.image_height_,
                                                      param_.image_width_, 10, cv::Scalar(255,255,255), true);

//    cv::namedWindow("test");
//    cv::imshow("test", img);
//    cv::waitKey();

    caffe::Datum datum;
    caffe::CVMatToDatum(img, &datum);
    std::vector<caffe::Datum> datum_vector;
    datum_vector.push_back(datum);

    const boost::shared_ptr<caffe::MemoryDataLayer<Dtype> > layer = boost::static_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net_->layer_by_name("data"));
    layer->AddDatumVector(datum_vector);

    const std::vector<caffe::Blob<Dtype>*>& result = feature_extraction_net_->ForwardPrefilled();
    const boost::shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net_->blob_by_name(param_.extract_feature_blob_names_[0]);
    signature.resize( 1, feature_blob->count());

    for(size_t i=0; i<feature_blob->count(); i++)
        signature(0, i) = feature_blob->cpu_data()[i];

    indices_.clear();

    return true;
}
    template class V4R_EXPORTS CNN_Feat_Extractor<pcl::PointXYZRGB>;
}
