/******************************************************************************
 * Copyright (c) 2016 Thomas Faeulhammer
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

class V4R_EXPORTS Harris3DKeypointExtractorParameter
{
    public:
    float threshold_;
    float search_radius_; ///< radius the sphere radius used as the maximum distance to consider a point a neighbor
    bool refine_;

    Harris3DKeypointExtractorParameter() :
        threshold_ (1e-4),
        search_radius_ (0.02f),
        refine_ (true)
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
        po::options_description desc("Harris 3D Keypoint Extractor Parameter\n=====================\n");
        desc.add_options()
                ("help,h", "produce help message")
                ("harris3d_kp_threshold", po::value<float>(&threshold_)->default_value(threshold_), "threshold")
                ("harris3d_search_radius", po::value<float>(&search_radius_)->default_value(search_radius_), "radius the sphere radius used as the maximum distance to consider a point a neighbor")
                ("harris3d_refine", po::value<bool>(&refine_)->default_value(refine_), "refine")
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
class V4R_EXPORTS Harris3DKeypointExtractor : public KeypointExtractor<PointT>
{
private:
    typedef typename pcl::PointCloud<PointT>::Ptr PointInTPtr;
    using KeypointExtractor<PointT>::input_;
    using KeypointExtractor<PointT>::normals_;
    using KeypointExtractor<PointT>::indices_;
    using KeypointExtractor<PointT>::keypoints_;
    using KeypointExtractor<PointT>::keypoint_indices_;

    Harris3DKeypointExtractorParameter param_;

public:
    Harris3DKeypointExtractor(const Harris3DKeypointExtractorParameter &p = Harris3DKeypointExtractorParameter()) :
        param_ (p)
    { }

    void compute ();

    bool needNormals() const
    {
        return true;
    }

    int getKeypointExtractorType() const { return KeypointType::HARRIS3D; }

    std::string getKeypointExtractorName() const { return "harris3d"; }

    typename pcl::PointCloud<PointT>::Ptr
    getKeypoints()
    {
        keypoints_.reset(new pcl::PointCloud<PointT>);
        pcl::copyPointCloud(*input_, keypoint_indices_, *keypoints_);
        return keypoints_;
    }

    typedef boost::shared_ptr< Harris3DKeypointExtractor<PointT> > Ptr;
    typedef boost::shared_ptr< Harris3DKeypointExtractor<PointT> const> ConstPtr;
};

}

