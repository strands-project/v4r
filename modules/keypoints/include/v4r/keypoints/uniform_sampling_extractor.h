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

#include <pcl/common/io.h>
#include <v4r/keypoints/keypoint_extractor.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace v4r
{
class V4R_EXPORTS UniformSamplingExtractorParameter
{
public:
    float sampling_density_; ///< sampling distance in meter
    UniformSamplingExtractorParameter(
            float sampling_density = 0.02f
            ) :
    sampling_density_ (sampling_density)
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
        po::options_description desc("Uniform Sampling Extractor Parameter\n=====================\n");
        desc.add_options()
                ("help,h", "produce help message")
                ("uniform_sampling_density", po::value<float>(&sampling_density_)->default_value(sampling_density_), "sampling density in meter")
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

template<typename PointT>
class V4R_EXPORTS UniformSamplingExtractor : public KeypointExtractor<PointT>
{
private:
    typedef typename pcl::PointCloud<PointT>::Ptr PointInTPtr;
    using KeypointExtractor<PointT>::input_;
    using KeypointExtractor<PointT>::indices_;
    using KeypointExtractor<PointT>::keypoints_;
    using KeypointExtractor<PointT>::keypoint_indices_;

   UniformSamplingExtractorParameter param_;
public:
    UniformSamplingExtractor( const UniformSamplingExtractorParameter &p = UniformSamplingExtractorParameter() ) :
        param_ (p)
    { }

    void compute ();

    int getKeypointExtractorType() const { return KeypointType::UniformSampling; }
    std::string getKeypointExtractorName() const { return "uniform_sampling"; }

    typename pcl::PointCloud<PointT>::Ptr
    getKeypoints()
    {
        keypoints_.reset(new pcl::PointCloud<PointT>);
        pcl::copyPointCloud(*input_, keypoint_indices_, *keypoints_);
        return keypoints_;
    }

    typedef boost::shared_ptr< UniformSamplingExtractor<PointT> > Ptr;
    typedef boost::shared_ptr< UniformSamplingExtractor<PointT> const> ConstPtr;
};
}
