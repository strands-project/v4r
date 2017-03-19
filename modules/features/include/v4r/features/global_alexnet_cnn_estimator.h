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

#pragma once

#include <caffe/net.hpp>
#include <opencv/cv.h>
#include <v4r/features/global_estimator.h>

#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;

namespace v4r
{

class V4R_EXPORTS CNN_Feat_ExtractorParameter
{
public:
    size_t image_height_;
    size_t image_width_;
    int device_id_;
    std::string device_name_;
    std::string output_layer_name_; ///< name of the layer of the CNN that is used for feature extraction
    std::string feature_extraction_proto_, pretrained_binary_proto_, input_mean_file_;
    cv::Vec3b background_color_; ///< background color of pixels not representing the object (only used if remove background is enabled)
    bool remove_background_; ///< if true, removes background for pixels not representing the object
    int margin_; ///< margin in pixel added to each side of the object's bounding box
    CNN_Feat_ExtractorParameter() :
            image_height_ (256),
            image_width_ (256),
            device_id_ (0),
            device_name_ ("CPU"),
            output_layer_name_ ("fc7"),
            background_color_ ( cv::Vec3b(255,255,255) ),
            remove_background_ (false),
            margin_ (10)
    {}


    /**
     * @brief init parameters
     * @param command_line_arguments (according to Boost program options library)
     * @return unused parameters (given parameters that were not used in this initialization call)
     */
    std::vector<std::string>
    init(int argc, char **argv)
    {
            std::vector<std::string> arguments(argv + 1, argv + argc);
            return init(arguments);
    }

    /**
     * @brief init parameters
     * @param command_line_arguments (according to Boost program options library)
     * @return unused parameters (given parameters that were not used in this initialization call)
     */
    std::vector<std::string>
    init(const std::vector<std::string> &command_line_arguments)
    {
        po::options_description desc("CNN parameters\n=====================");
        desc.add_options()
                ("help,h", "produce help message")
                ("cnn_net", po::value<std::string>(&feature_extraction_proto_)->required(), "Definition of CNN (.prototxt)")
                ("cnn_pretrained_net", po::value<std::string>(&pretrained_binary_proto_)->required(), "Trained weights (.caffemodel)")
                ("cnn_input_mean_file", po::value<std::string>(&input_mean_file_)->required(), "mean pixel values (.binaryproto)")
                ("cnn_device_name", po::value<std::string>(&device_name_)->default_value(device_name_), "")
                ("cnn_output_layer_name", po::value<std::string>(&output_layer_name_)->default_value(output_layer_name_), "")
                ("cnn_device_id", po::value<int>(&device_id_)->default_value(device_id_), "")
                ("cnn_image_height", po::value<size_t>(&image_height_)->default_value(image_height_), "")
                ("cnn_remove_background", po::value<bool>(&remove_background_)->default_value(remove_background_), "if true, removes background for pixels not representing the object")
                ("cnn_margin", po::value<int>(&margin_)->default_value(margin_), "margin in pixel added to each side of the object's bounding box")
                ;
        po::variables_map vm;
        po::parsed_options parsed = po::command_line_parser(command_line_arguments).options(desc).allow_unregistered().run();
        std::vector<std::string> to_pass_further = po::collect_unrecognized(parsed.options, po::include_positional);
        po::store(parsed, vm);
        if (vm.count("help")) { std::cout << desc << std::endl; to_pass_further.push_back("-h"); }
        try { po::notify(vm); }
        catch(std::exception& e) {  std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl; }
        return to_pass_further;
    }
};

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

    CNN_Feat_ExtractorParameter param_;

private:
    boost::shared_ptr<caffe::Net<Dtype> > net_;
    bool init_;
    cv::Mat mean_;
    cv::Size input_geometry_;
    int num_channels_;

    void SetMean(const std::string& mean_file);
    void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
    void WrapInputLayer(std::vector<cv::Mat>* input_channels);
    int init();

public:
    CNN_Feat_Extractor(const CNN_Feat_ExtractorParameter &p = CNN_Feat_ExtractorParameter(),
                       const std::string &descr_name = "alexnet",
                       size_t descr_type = FeatureType::ALEXNET,
                       size_t feature_dimensions = 4096) :
        GlobalEstimator<PointT>(descr_name, descr_type, feature_dimensions),
        param_(p),
        init_ (false)
    { }

    bool compute(Eigen::MatrixXf &signature);

    bool compute(const cv::Mat &img, Eigen::MatrixXf &signature);

    bool needNormals() const { return false; }

    typedef boost::shared_ptr< CNN_Feat_Extractor<PointT, Dtype> > Ptr;
    typedef boost::shared_ptr< CNN_Feat_Extractor<PointT, Dtype> const> ConstPtr;
};

}
