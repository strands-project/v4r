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
void
CNN_Feat_Extractor<PointT, Dtype>::WrapInputLayer(std::vector<cv::Mat>* input_channels)
{
    caffe::Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

template<typename PointT, typename Dtype>
void
CNN_Feat_Extractor<PointT, Dtype>::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";
}


/* Load the mean file in binaryproto format. */
template<typename PointT, typename Dtype>
void
CNN_Feat_Extractor<PointT, Dtype>::SetMean(const std::string& mean_file)
{
    caffe::BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
            << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; i++)
    {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

template<typename PointT, typename Dtype>
int
CNN_Feat_Extractor<PointT, Dtype>::init(){

  if (strcmp(param_.device_name_.c_str(), "GPU") == 0) {
    Caffe::SetDevice(param_.device_id_);
    Caffe::set_mode(Caffe::GPU);
  }
  else
    Caffe::set_mode(Caffe::CPU);

  net_.reset(new Net<Dtype>(param_.feature_extraction_proto_, caffe::TEST));
  net_->CopyTrainedLayersFrom(param_.pretrained_binary_proto_);

  caffe::Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(param_.input_mean_file_);
  return 0;
}


template<typename PointT, typename Dtype>
bool
CNN_Feat_Extractor<PointT, Dtype>::compute (const cv::Mat &img, Eigen::MatrixXf &signature)
{
    if(!init_)
    {
        init();
        init_ = true;
    }

    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    net_->Forward();

    /* Copy the output layer to a std::vector */
//    Blob<float>* output_layer = net_->output_blobs()[0];
    boost::shared_ptr<Blob<float> > output_layer = net_->blob_by_name(param_.output_layer_name_);
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    std::vector<float> sign_f = std::vector<float>(begin, end);

    signature = Eigen::MatrixXf (1, sign_f.size());

    for(size_t i=0; i<sign_f.size(); i++)
        signature(0,i) = sign_f[i];

    indices_.clear();
    cloud_.reset();
    return true;
}

template<typename PointT, typename Dtype>
bool
CNN_Feat_Extractor<PointT, Dtype>::compute (Eigen::MatrixXf &signature)
{
    CHECK( cloud_ && cloud_->isOrganized() && !cloud_->points.empty() && !indices_.empty());
    cv::Mat img = v4r::ConvertPCLCloud2FixedSizeImage(*cloud_, indices_, param_.image_height_,
                                                      param_.image_width_, 10, cv::Scalar(255,255,255), true);
    compute(img, signature);
    indices_.clear();
    return true;
}
    template class V4R_EXPORTS CNN_Feat_Extractor<pcl::PointXYZRGB>;
}
