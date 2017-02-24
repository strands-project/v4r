/******************************************************************************
 * Copyright (c) 2012 Andreas Richtsfeld
 * Copyright (c) 2012 Hannes Prankl
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

#include <Eigen/Dense>
#include <v4r/core/macros.h>
#include <v4r/common/normal_estimator.h>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace v4r
{

/**
 * @brief The ZAdaptiveNormalsParameter class represent the parameter for ZAdpativeNormals estimation
 */
class V4R_EXPORTS ZAdaptiveNormalsParameter
{
public:
    float radius_;            ///< euclidean inlier radius
    int kernel_;               ///< kernel radius [px]
    bool adaptive_;            ///< Activate z-adaptive normals calcualation
    float kappa_;              ///< gradient
    float d_;                  ///< constant
    std::vector<int> kernel_radius_;   ///< Kernel radius for each 0.5 meter intervall (e.g. if 8 elements, then 0-4m)
    ZAdaptiveNormalsParameter(
            float radius=0.02,
            int kernel=5,
            bool adaptive=false,
            float kappa=0.005125,
            float d = 0.0,
            std::vector<int> kernel_radius = {3,3,3,3,4,5,6,7}
            )
        : radius_(radius),
          kernel_(kernel),
          adaptive_(adaptive),
          kappa_(kappa),
          d_(d),
          kernel_radius_ (kernel_radius)
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
        po::options_description desc("Surface Normal Estimator Parameter\n=====================\n");
        desc.add_options()
                ("help,h", "produce help message")
                ("normals_radius", po::value<float>(&radius_)->default_value(radius_), "euclidean inlier radius for normal computation.")
                ("normals_kernel_size", po::value<int>(&kernel_)->default_value(kernel_), "kernel radius in pixel.")
                ("normals_z_adaptive", po::value<bool>(&adaptive_)->default_value(adaptive_), "if true, adapts kernel radius with distance of point to camera.")
                ("normals_kappa", po::value<float>(&kappa_)->default_value(kappa_), "gradient.")
                ("normals_d", po::value<float>(&d_)->default_value(d_), "constant.")
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
class V4R_EXPORTS ZAdaptiveNormalsPCL : public NormalEstimator<PointT>
{
public:
    using NormalEstimator<PointT>::input_;
    using NormalEstimator<PointT>::indices_;
    using NormalEstimator<PointT>::normal_;

private:
    ZAdaptiveNormalsParameter param_;

    float sqr_radius;

    void computeCovarianceMatrix ( const std::vector<int> &indices, const Eigen::Vector3f &mean, Eigen::Matrix3f &cov);
    void getIndices(size_t u, size_t v, int kernel, std::vector<int> &indices);
    float computeNormal(std::vector<int> &indices,  Eigen::Matrix3d &eigen_vectors);

    int getIdx(short x, short y) const
    {
        return y*input_->width+x;
    }

public:
    ZAdaptiveNormalsPCL(const ZAdaptiveNormalsParameter &p = ZAdaptiveNormalsParameter())
        : param_(p)
    {
        sqr_radius = p.radius_*p.radius_;
    }

    ~ZAdaptiveNormalsPCL(){}

    pcl::PointCloud<pcl::Normal>::Ptr
    compute ();

    int
    getNormalEstimatorType() const
    {
        return NormalEstimatorType::Z_ADAPTIVE;
    }

    typedef boost::shared_ptr< ZAdaptiveNormalsPCL> Ptr;
    typedef boost::shared_ptr< ZAdaptiveNormalsPCL const> ConstPtr;
};

}

