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

#include <v4r/keypoints/keypoint_extractor.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace v4r
{

class V4R_EXPORTS IssKeypointExtractorParameter // see PCL documentation for further details
{
    public:
    double salient_radius_; ///< Set the radius of the spherical neighborhood used to compute the scatter matrix.
    double non_max_radius_; ///< Set the radius for the application of the non maxima supression algorithm.
    double normal_radius_; ///< Set the radius used for the estimation of the surface normals of the input cloud. If the radius is too large, the temporal performances of the detector may degrade significantly. Only used if parameter with_border_estimation equal true.
    double border_radius_; ///< Set the radius used for the estimation of the boundary points. If the radius is too large, the temporal performances of the detector may degrade significantly. Only used if parameter with_border_estimation equal true.
    double gamma_21_; ///< Set the upper bound on the ratio between the second and the first eigenvalue.
    double gamma_32_; ///< Set the upper bound on the ratio between the third and the second eigenvalue.
    int min_neighbors_ ; ///< Set the minimum number of neighbors that has to be found while applying the non maxima suppression algorithm.
    bool with_border_estimation_;
    int threads_;
    float angle_thresh_deg_;

    IssKeypointExtractorParameter() :
        salient_radius_ (0.02f),//(6*0.005f),
        non_max_radius_ (4*0.005f),
        normal_radius_ (4*0.005f),
        border_radius_ (1*0.005f),
        gamma_21_ (0.8),//(0.975),
        gamma_32_ (0.8),//(0.975),
        min_neighbors_ (5),
        with_border_estimation_ (false),
        threads_ (4),
        angle_thresh_deg_(60.f)
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
        po::options_description desc("ISS Keypoint Extractor Parameter\n=====================\n");
        desc.add_options()
                ("help,h", "produce help message")
                ("iss_salient_radius", po::value<double>(&salient_radius_)->default_value(salient_radius_), "Set the radius of the spherical neighborhood used to compute the scatter matrix.")
                ("iss_non_max_radius", po::value<double>(&non_max_radius_)->default_value(non_max_radius_), "Set the radius for the application of the non maxima supression algorithm.")
                ("iss_normal_radius", po::value<double>(&normal_radius_)->default_value(normal_radius_), " Set the radius used for the estimation of the surface normals of the input cloud. If the radius is too large, the temporal performances of the detector may degrade significantly. Only used if parameter with_border_estimation equal true.")
                ("iss_border_radius", po::value<double>(&border_radius_)->default_value(border_radius_), "Set the radius used for the estimation of the boundary points. If the radius is too large, the temporal performances of the detector may degrade significantly. Only used if parameter with_border_estimation equal true.")
                ("iss_gamma_21", po::value<double>(&gamma_21_)->default_value(gamma_21_), "Set the upper bound on the ratio between the second and the first eigenvalue")
                ("iss_gamma_32", po::value<double>(&gamma_32_)->default_value(gamma_32_), "Set the upper bound on the ratio between the third and the second eigenvalue.")
                ("iss_min_neighbors", po::value<int>(&min_neighbors_)->default_value(min_neighbors_), "Set the minimum number of neighbors that has to be found while applying the non maxima suppression algorithm")
                ("iss_with_border_estimation", po::value<bool>(&with_border_estimation_)->default_value(with_border_estimation_), "")
                ("iss_threads", po::value<int>(&threads_)->default_value(threads_), "number of threads")
                ("iss_angle_thresh_deg", po::value<float>(&angle_thresh_deg_)->default_value(angle_thresh_deg_), "Set the decision boundary (angle threshold) that marks points as boundary or regular.")
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
class V4R_EXPORTS IssKeypointExtractor : public KeypointExtractor<PointT>
{
private:
    typedef typename pcl::PointCloud<PointT>::Ptr PointInTPtr;
    using KeypointExtractor<PointT>::input_;
    using KeypointExtractor<PointT>::normals_;
    using KeypointExtractor<PointT>::indices_;
    using KeypointExtractor<PointT>::keypoints_;
    using KeypointExtractor<PointT>::keypoint_indices_;

    IssKeypointExtractorParameter param_;

public:

    IssKeypointExtractor( const IssKeypointExtractorParameter &p = IssKeypointExtractorParameter() ) :
        param_ (p)
    {}

    void
    compute ();

    bool
    needNormals() const
    {
        return true;
    }

    int getKeypointExtractorType() const { return KeypointType::ISS; }

    std::string getKeypointExtractorName() const { return "iss"; }

    typedef boost::shared_ptr< IssKeypointExtractor<PointT> > Ptr;
    typedef boost::shared_ptr< IssKeypointExtractor<PointT> const> ConstPtr;
};
}
