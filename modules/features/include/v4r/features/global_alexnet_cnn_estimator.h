/******************************************************************************
 * Copyright (c) 2015 Thomas Faeulhammer
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/

#ifndef V4R_CNN_FEATURE_EXTRACTOR_H__
#define V4R_CNN_FEATURE_EXTRACTOR_H__

#include <caffe/net.hpp>
#include <opencv/cv.h>
#include <v4r/features/global_estimator.h>
#include <v4r/common/pcl_opencv.h>

#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;

namespace v4r
{

/**
 * @brief Feature extraction from a Convolutional Neural Network based on Berkeley's Caffe Framework.
 * Extracts the image from a RGB-D point cloud by the bounding box indicated from the object indices
 * @author Thomas Faeulhammer
 * @date Nov, 2015
 */
template<typename PointT, typename Dtype = float>
class V4R_EXPORTS CNN_Feat_Extractor : public GlobalEstimator<PointT>
{
public:
    using GlobalEstimator<PointT>::indices_;
    using GlobalEstimator<PointT>::cloud_;
    using GlobalEstimator<PointT>::descr_name_;
    using GlobalEstimator<PointT>::descr_type_;
    using GlobalEstimator<PointT>::feature_dimensions_;

    class V4R_EXPORTS Parameter
    {
    public:
        size_t image_height_;
        size_t image_width_;
        int device_id_;
        std::string device_name_;
        Parameter(
                size_t image_height = 256,
                size_t image_width = 256,
                int device_id = 0,
                std::string device_name = "CPU"
                )
            :
                image_height_ (image_height),
                image_width_ (image_width),
                device_id_ (device_id),
                device_name_ (device_name)
        {}
    } param_;

private:
    typedef typename pcl::PointCloud<PointT>::Ptr PointInTPtr;
    boost::shared_ptr<caffe::Net<Dtype> > feature_extraction_net_;
    std::vector<std::string> extract_feature_blob_names_;
    std::string feature_extraction_proto_, pretrained_binary_proto_;
    bool init_;

public:
    CNN_Feat_Extractor(const Parameter &p = Parameter()) : param_(p), init_ (false)
    {
        descr_name_ = "alexnet";
        descr_type_ = FeatureType::ALEXNET;
        feature_dimensions_ = 4096;
    }

    CNN_Feat_Extractor(int argc, char **argv) : init_ (false), param_ (Parameter())
    {
        descr_name_ = "alexnet";
        descr_type_ = FeatureType::ALEXNET;
        feature_dimensions_ = 4096;

        po::options_description desc("CNN parameters\n=====================");
        desc.add_options()
                ("help,h", "produce help message")
                ("feature_extraction_proto", po::value<std::string>(&feature_extraction_proto_)->required(), "")
                ("pretrained_net", po::value<std::string>(&pretrained_binary_proto_)->required(), "")
                ("extract_feature_blob_names", po::value<std::vector<std::string> >(&extract_feature_blob_names_)->multitoken()->required(), "")
                ("device_name", po::value<std::string>(&param_.device_name_)->default_value(param_.device_name_), "")
                ("device_id", po::value<int>(&param_.device_id_)->default_value(param_.device_id_), "")
                ("cnn_image_height", po::value<size_t>(&param_.image_height_)->default_value(param_.image_height_), "")
                ("cnn_image_width", po::value<size_t>(&param_.image_width_)->default_value(param_.image_width_), "")
                ;
        po::variables_map vm;
        po::parsed_options parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
        po::store(parsed, vm);
        if (vm.count("help")) { std::cout << desc << std::endl; }
        try { po::notify(vm); }
        catch(std::exception& e) {  std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl; }
    }

    int init();

    void setFeatureExtractionProto(const std::string &val)
    {
        feature_extraction_proto_ = val;
    }

    void setExtractFeatureBlobNames(const std::vector<std::string> &val)
    {
        extract_feature_blob_names_ = val;
    }

    void setPretrainedBinaryProto(const std::string &val)
    {
        pretrained_binary_proto_ = val;
    }

    bool compute(Eigen::MatrixXf &signature);

    bool needNormals() const
    {
        return false;
    }


    typedef boost::shared_ptr< ::v4r::CNN_Feat_Extractor<PointT, Dtype> > Ptr;
    typedef boost::shared_ptr< ::v4r::CNN_Feat_Extractor<PointT, Dtype> const> ConstPtr;
};

}
#endif