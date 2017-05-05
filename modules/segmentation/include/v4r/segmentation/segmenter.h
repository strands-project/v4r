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
*      @brief base class for segmentation
*/

#pragma once

#include <v4r/core/macros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;

namespace v4r
{

class V4R_EXPORTS SegmenterParameter
{
public:
    size_t min_cluster_size_; ///< minimum number of points in a cluster
    size_t max_cluster_size_; ///< minimum number of points in a cluster
    double distance_threshold_; ///< tolerance in meters for difference in perpendicular distance (d component of plane equation) to the plane between neighboring points, to be considered part of the same plane
    double angular_threshold_deg_; ///< tolerance in gradients for difference in normal direction between neighboring points, to be considered part of the same plane
    int wsize_;
    float cluster_tolerance_;
    SegmenterParameter (
               size_t min_cluster_size=500,
               size_t max_cluster_size=std::numeric_limits<int>::max(),
               double distance_threshold = 0.01f, //0.035f
               double angular_threshold_deg = 10.f,
               int wsize = 5,
               float cluster_tolerance = 0.05f
            )
        :
          min_cluster_size_ (min_cluster_size),
          max_cluster_size_ (max_cluster_size),
          distance_threshold_ (distance_threshold),
          angular_threshold_deg_ (angular_threshold_deg),
          wsize_ (wsize),
          cluster_tolerance_ (cluster_tolerance)
    {}

    virtual ~SegmenterParameter(){}

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
        po::options_description desc("Segmentation Parameter\n=====================\n");
        desc.add_options()
                ("help,h", "produce help message")
                ("seg_min_cluster_size", po::value<size_t>(&min_cluster_size_)->default_value(min_cluster_size_), "minimum number of points in a cluster")
                ("seg_max_cluster_size", po::value<size_t>(&max_cluster_size_)->default_value(max_cluster_size_), "")
                ("seg_distance_threshold", po::value<double>(&distance_threshold_)->default_value(distance_threshold_), "tolerance in meters for difference in perpendicular distance (d component of plane equation) to the plane between neighboring points, to be considered part of the same plane")
                ("seg_angular_threshold_deg", po::value<double>(&angular_threshold_deg_)->default_value(angular_threshold_deg_), "tolerance in gradients for difference in normal direction between neighboring points, to be considered part of the same plane.")
                ("seg_wsize", po::value<int>(&wsize_)->default_value(wsize_), "")
                ("seg_object_cluster_tolerance", po::value<float>(&cluster_tolerance_)->default_value(cluster_tolerance_), "")
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
class V4R_EXPORTS Segmenter
{
protected:
    typename pcl::PointCloud<PointT>::ConstPtr scene_; ///< point cloud to be segmented
    pcl::PointCloud<pcl::Normal>::ConstPtr normals_; ///< normals of the cloud to be segmented
    std::vector< std::vector<int> > clusters_; ///< segmented clusters. Each cluster represents a bunch of indices of the input cloud

public:
    virtual ~Segmenter(){}

    Segmenter()
    {}

    /**
     * @brief sets the cloud which ought to be segmented
     * @param cloud
     */
    void
    setInputCloud ( const typename pcl::PointCloud<PointT>::ConstPtr &cloud )
    {
        scene_ = cloud;
    }

    /**
     * @brief sets the normals of the cloud which ought to be segmented
     * @param normals
     */
    void
    setNormalsCloud ( const pcl::PointCloud<pcl::Normal>::ConstPtr &normals )
    {
        normals_ = normals;
    }

    /**
     * @brief get segmented indices
     * @param indices
     */
    void
    getSegmentIndices ( std::vector<std::vector<int> > & indices ) const
    {
        indices = clusters_;
    }

    virtual bool
    getRequiresNormals() = 0;

    /**
     * @brief segment
     */
    virtual void
    segment() = 0;

    typedef boost::shared_ptr< Segmenter<PointT> > Ptr;
    typedef boost::shared_ptr< Segmenter<PointT> const> ConstPtr;
};

}
