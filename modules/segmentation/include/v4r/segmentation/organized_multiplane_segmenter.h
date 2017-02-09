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

/**
*
*      @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
*      @date April, 2016
*      @brief organized multiplane segmentation (taken from PCL)
*/

#pragma once

#include <v4r/core/macros.h>
#include <v4r/segmentation/segmenter.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace v4r
{

class V4R_EXPORTS OrganizedMultiplaneSegmenterParameter
{
public:
    int min_cluster_size_, num_plane_inliers_;
    double sensor_noise_max_,
           angular_threshold_deg_;
    OrganizedMultiplaneSegmenterParameter (
               int min_cluster_size=500,
               int num_plane_inliers=1000,
               double sensor_noise_max = 0.01f,
               double angular_threshold_deg = 10.f)
        :
          min_cluster_size_ (min_cluster_size),
          num_plane_inliers_ (num_plane_inliers),
          sensor_noise_max_ (sensor_noise_max),
          angular_threshold_deg_ (angular_threshold_deg)
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
        po::options_description desc("Organized Multiplane Segmentation Parameter\n=====================\n");
        desc.add_options()
                ("help,h", "produce help message")
                ("org_multi_seg_min_cluster_size", po::value<int>(&min_cluster_size_)->default_value(min_cluster_size_), "")
                ("org_multi_seg_num_plane_inliers", po::value<int>(&num_plane_inliers_)->default_value(num_plane_inliers_), "")
                ("org_multi_seg_sensor_noise_max", po::value<double>(&sensor_noise_max_)->default_value(sensor_noise_max_), "")
                ("org_multi_seg_angular_threshold_deg", po::value<double>(&angular_threshold_deg_)->default_value(angular_threshold_deg_), "")
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

template <typename PointT>
class V4R_EXPORTS OrganizedMultiplaneSegmenter : public Segmenter<PointT>
{
    using Segmenter<PointT>::indices_;
    using Segmenter<PointT>::normals_;
    using Segmenter<PointT>::clusters_;
    using Segmenter<PointT>::scene_;
    using Segmenter<PointT>::dominant_plane_;

    OrganizedMultiplaneSegmenterParameter param_;

public:
    OrganizedMultiplaneSegmenter(const OrganizedMultiplaneSegmenterParameter &p = OrganizedMultiplaneSegmenterParameter() ) : param_(p)
    { }

    bool getRequiresNormals() { return true; }

    void segment();

    typedef boost::shared_ptr< OrganizedMultiplaneSegmenter<PointT> > Ptr;
    typedef boost::shared_ptr< OrganizedMultiplaneSegmenter<PointT> const> ConstPtr;
};

}
