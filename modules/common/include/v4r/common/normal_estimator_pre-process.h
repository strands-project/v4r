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

#include <v4r/common/normal_estimator.h>
#include <v4r/core/macros.h>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace v4r
{

class V4R_EXPORTS NormalEstimatorPreProcessParameter
{
public:
    bool compute_mesh_resolution_;
    bool do_voxel_grid_;
    bool remove_outliers_;

    //this values are used when CMR=false
    float grid_resolution_;
    float normal_radius_;

    //this are used when CMR=true
    float factor_normals_;
    float factor_voxel_grid_;
    float min_n_radius_;
    bool force_unorganized_;

    bool only_on_indices_;
    pcl::PointIndices indices_;

    NormalEstimatorPreProcessParameter (
            ) :
        compute_mesh_resolution_(false),
        do_voxel_grid_ (false),
        remove_outliers_ (false),
        grid_resolution_(0.01f),
        normal_radius_(0.02f),
        factor_normals_(1),
        factor_voxel_grid_(1),
        min_n_radius_ (16),
        force_unorganized_(false),
        only_on_indices_ (false)
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
        std::cerr << "parameter init function for pre-process normals not implemented! " << std::endl;
        po::options_description desc("Surface Normal Estimator Parameter\n=====================\n");
        desc.add_options()
                ("help,h", "produce help message")
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
class V4R_EXPORTS NormalEstimatorPreProcess : public NormalEstimator<PointT>
{
public:
    using NormalEstimator<PointT>::input_;
    using NormalEstimator<PointT>::indices_;
    using NormalEstimator<PointT>::normal_;

private:
    NormalEstimatorPreProcessParameter param_;

    typename pcl::PointCloud<PointT>::Ptr processed_;

public:
    NormalEstimatorPreProcess(
            const NormalEstimatorPreProcessParameter &p = NormalEstimatorPreProcessParameter()
            )
        : param_(p)
    {
    }

    ~NormalEstimatorPreProcess(){}

    pcl::PointCloud<pcl::Normal>::Ptr
    compute ();

    int
    getNormalEstimatorType() const
    {
        return NormalEstimatorType::PCL_INTEGRAL_NORMAL;
    }

    typedef boost::shared_ptr< NormalEstimatorPreProcess> Ptr;
    typedef boost::shared_ptr< NormalEstimatorPreProcess const> ConstPtr;
};

}

