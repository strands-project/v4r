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

/**
*
*      @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
*      @date April, 2016
*      @brief Euclidean segmentation (taken from PCL)
*/

#pragma once

#include <v4r/core/macros.h>
#include <v4r/common/pcl_utils.h>
#include <v4r/segmentation/segmenter.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;

namespace v4r
{

template <typename PointT>
class V4R_EXPORTS EuclideanSegmenter : public Segmenter<PointT>
{
    using Segmenter<PointT>::indices_;
    using Segmenter<PointT>::normals_;
    using Segmenter<PointT>::clusters_;
    using Segmenter<PointT>::scene_;
    using Segmenter<PointT>::dominant_plane_;
    using Segmenter<PointT>::all_planes_;
    using Segmenter<PointT>::visualize_;


    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_xyz_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered;
    std::vector<bool> filter_mask_;

public:
    typedef boost::shared_ptr< EuclideanSegmenter<PointT> > Ptr;
    typedef boost::shared_ptr< EuclideanSegmenter<PointT> const> ConstPtr;

    class Parameter
    {
    public:
        int min_cluster_size_;
        int max_cluster_size_;
        int num_plane_inliers_;
        float cluster_tolerance_;
        float sensor_noise_max_;
        float chop_z_;
        Parameter (
                int min_cluster_size=500,
                int max_cluster_size=std::numeric_limits<int>::max(),
                int num_plane_inliers=1000,
                float cluster_tolerance = 0.02,
                float sensor_noise_max = 0.01f,
                float chop_z = std::numeric_limits<float>::max()
                )
            :
              min_cluster_size_ (min_cluster_size),
              max_cluster_size_ (max_cluster_size),
              num_plane_inliers_ (num_plane_inliers),
              cluster_tolerance_ (cluster_tolerance),
              sensor_noise_max_ (sensor_noise_max),
              chop_z_ ( chop_z )
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
            po::options_description desc("Euclidean Segmentation\n=====================");
            desc.add_options()
                    ("help,h", "produce help message")
                    ("seg_min_cluster_size", po::value<int>(&min_cluster_size_)->default_value(min_cluster_size_), "")
                    ("seg_max_cluster_size", po::value<int>(&max_cluster_size_)->default_value(max_cluster_size_), "")
                    ("seg_num_plane_inliers", po::value<int>(&num_plane_inliers_)->default_value(num_plane_inliers_), "")
                    ("seg_cluster_tolerance", po::value<float>(&cluster_tolerance_)->default_value(cluster_tolerance_), "")
                    ("seg_chop_z", po::value<float>(&chop_z_)->default_value(chop_z_), "")
                    ("seg_sensor_noise_max", po::value<float>(&sensor_noise_max_)->default_value(sensor_noise_max_), "")
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
    }param_;

    EuclideanSegmenter(const Parameter &p = Parameter() ) : param_(p)  { visualize_ = false; }

    bool getRequiresNormals() { return false; }

    void
    computeTablePlanes();

    void
    segment();
};

}
