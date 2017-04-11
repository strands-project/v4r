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

#include <v4r/core/macros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace v4r
{

class V4R_EXPORTS PlaneExtractorParameter
{
public:
    size_t max_iterations_; ///< maximum number of iterations the sample consensus method will run
    size_t min_num_plane_inliers_; ///< minimum number of plane inliers
    double distance_threshold_; ///< tolerance in meters for difference in perpendicular distance (d component of plane equation) to the plane between neighboring points, to be considered part of the same plane
    double angular_threshold_deg_; ///< tolerance in gradients for difference in normal direction between neighboring points, to be considered part of the same plane
    bool check_if_higher_plane_exists_; ///< if true, checks if there is a plane (with at least min_num_plane_inliers_) parallel to the one with maximum inliers and takes this one instead
    bool compute_all_planes_; ///< if true, computes all planes (also if method does not compute all of them intrinsically)
    PlaneExtractorParameter (
               size_t max_iterations=100,
               size_t num_plane_inliers=1000,
               double sensor_noise_max = 0.01f,
               double angular_threshold_deg = 10.f,
               bool check_if_higher_plane_exists = true,
               bool compute_all_planes = true
            )
        :
          max_iterations_ (max_iterations),
          min_num_plane_inliers_ (num_plane_inliers),
          distance_threshold_ (sensor_noise_max),
          angular_threshold_deg_ (angular_threshold_deg),
          check_if_higher_plane_exists_ (check_if_higher_plane_exists),
          compute_all_planes_ (compute_all_planes)
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
        po::options_description desc("Plane Extractor Parameter\n=====================\n");
        desc.add_options()
                ("help,h", "produce help message")
                ("plane_extractor_max_iterations", po::value<size_t>(&max_iterations_)->default_value(max_iterations_), "maximum number of iterations the sample consensus method will run")
                ("plane_extractor_min_num_plane_inliers", po::value<size_t>(&min_num_plane_inliers_)->default_value(min_num_plane_inliers_), "minimum number of plane inliers")
                ("plane_extractor_distance_threshold", po::value<double>(&distance_threshold_)->default_value(distance_threshold_), "tolerance in meters for difference in perpendicular distance (d component of plane equation) to the plane between neighboring points, to be considered part of the same plane")
                ("plane_extractor_angular_threshold_deg", po::value<double>(&angular_threshold_deg_)->default_value(angular_threshold_deg_), "tolerance in gradients for difference in normal direction between neighboring points, to be considered part of the same plane.")
                ("plane_extractor_check_if_higher_plane_exists", po::value<bool>(&check_if_higher_plane_exists_)->default_value(check_if_higher_plane_exists_), "if true, checks if there is a plane (with at least min_num_plane_inliers_) parallel to the one with maximum inliers and takes this one instead.")
                ("plane_extractor_compute_all_planes", po::value<bool>(&compute_all_planes_)->default_value(compute_all_planes_), "if true, computes all planes (also if method does not compute all of them intrinsically)")
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
 * @brief The PlaneExtractor class is an abstract class which extracts planar surfaces from a point cloud
 */
template<typename PointT>
class V4R_EXPORTS PlaneExtractor
{
protected:
    typename pcl::PointCloud<PointT>::ConstPtr cloud_; ///< input cloud
    pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud_; ///< surface normals associated to input cloud
    std::vector< Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > all_planes_; ///< all extracted planes (if segmentation algorithm supports it)
    PlaneExtractorParameter param_;
    std::vector<std::vector<int> > plane_inliers_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PlaneExtractor ( const PlaneExtractorParameter &p = PlaneExtractorParameter() )
        : param_(p)
    {}

    /**
     * @brief compute extract planes
     */
    virtual void compute() = 0;

    /**
     * @brief getRequiresNormals
     * @return true if method requires normal cloud to be set
     */
    virtual bool getRequiresNormals() const = 0;

    /**
     * @brief sets the cloud which ought to be segmented
     * @param cloud
     */
    void
    setInputCloud ( const typename pcl::PointCloud<PointT>::ConstPtr &cloud )
    {
        cloud_ = cloud;
    }

    /**
     * @brief sets the normals of the cloud which ought to be segmented
     * @param normals
     */
    void
    setNormalsCloud ( const pcl::PointCloud<pcl::Normal>::ConstPtr &normals )
    {
        normal_cloud_ = normals;
    }

    /**
     * @brief getPlanes
     * @return all extracted planes
     */
    std::vector< Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> >
    getPlanes() const
    {
        return all_planes_;
    }

    std::vector<std::vector<int> >
    getPlaneInliers() const
    {
        return plane_inliers_;
    }

    typedef boost::shared_ptr< PlaneExtractor<PointT> > Ptr;
    typedef boost::shared_ptr< PlaneExtractor<PointT> const> ConstPtr;
};

}
