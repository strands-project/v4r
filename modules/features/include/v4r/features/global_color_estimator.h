/******************************************************************************
 * Copyright (c) 2017 Thomas Faeulhammer
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

#include <v4r/common/color_transforms.h>
//#include <v4r/common/rgb2cielab.h>
#include <v4r/core/macros.h>
#include <v4r/features/global_estimator.h>
#include <v4r/features/types.h>

#include <glog/logging.h>

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/serialization.hpp>

#include <fstream>

namespace po = boost::program_options;

namespace v4r
{

class V4R_EXPORTS GlobalColorEstimatorParameter
{
 public:
    size_t num_bins; ///< number of bins for each color chanel to create color histogram
    float std_dev_multiplier_; ///< multiplication factor of the standard deviation of the color channel for minimum and maximum range of color histogram
    GlobalColorEstimatorParameter() :
        num_bins (15),
        std_dev_multiplier_(3.f)
    { }

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
        po::options_description desc("Global Color Feature Estimator Parameter\n=====================\n");
        desc.add_options()
                ("help,h", "produce help message")
                ("global_color_num_bins", po::value<size_t>(&num_bins)->default_value(num_bins), "number of bins for each color chanel to create color histogram.")
                ("global_color_std_dev_multiplier", po::value<float>(&std_dev_multiplier_)->default_value(std_dev_multiplier_), "multiplication factor of the standard deviation of the color channel for minimum and maximum range of color histogram.")
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
 * @brief The GlobalColorEstimator class implements a simple global description
 * in terms of the color of the input cloud
 */
template<typename PointT>
class V4R_EXPORTS GlobalColorEstimator : public GlobalEstimator<PointT>
{
private:
    using GlobalEstimator<PointT>::indices_;
    using GlobalEstimator<PointT>::cloud_;
    using GlobalEstimator<PointT>::descr_name_;
    using GlobalEstimator<PointT>::descr_type_;
    using GlobalEstimator<PointT>::feature_dimensions_;

    GlobalColorEstimatorParameter param_;

//    RGB2CIELAB::Ptr color_transf_;

public:
    GlobalColorEstimator(const GlobalColorEstimatorParameter &p = GlobalColorEstimatorParameter())
        : GlobalEstimator<PointT>("global_color", FeatureType::GLOBAL_COLOR),
          param_(p)
    {
        feature_dimensions_ = 3 * param_.num_bins;
    }

    bool compute (Eigen::MatrixXf &signature);

    bool needNormals() const { return false; }

    typedef boost::shared_ptr< GlobalColorEstimator<PointT> > Ptr;
    typedef boost::shared_ptr< GlobalColorEstimator<PointT> const> ConstPtr;
};
}
