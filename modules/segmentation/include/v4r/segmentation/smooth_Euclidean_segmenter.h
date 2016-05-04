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
*      @brief smooth Euclidean segmentation
*/

#ifndef V4R_SMOOTH_EUCLIDEAN_SEGMENTER_H__
#define V4R_SMOOTH_EUCLIDEAN_SEGMENTER_H__

#include <v4r/core/macros.h>
#include <v4r/segmentation/segmenter.h>
#include <pcl/octree/octree.h>

#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;

namespace v4r
{

template <typename PointT>
class V4R_EXPORTS SmoothEuclideanSegmenter : public Segmenter<PointT>
{
    using Segmenter<PointT>::indices_;
    using Segmenter<PointT>::normals_;
    using Segmenter<PointT>::clusters_;
    using Segmenter<PointT>::scene_;
    using Segmenter<PointT>::dominant_plane_;
    using Segmenter<PointT>::visualize_;

    typename pcl::octree::OctreePointCloudSearch<PointT>::Ptr octree_;

public:
    class Parameter
    {
    public:
        float eps_angle_threshold_deg_;
        float curvature_threshold_;
        float cluster_tolerance_;
        int min_points_;
        bool z_adaptive_;   /// @brief if true, scales the smooth segmentation parameters linear with distance (constant till 1m at the given parameters)
        float octree_resolution_;
        bool force_unorganized_; /// @brief if true, searches for neighboring points using the search tree and not pixel neighbors (even though input cloud is organized)
        bool compute_planar_patches_only_;  /// @brief if true, only compute planar surface patches
        float planar_inlier_dist_;  /// @brief maximum allowed distance of a point to the plane

        Parameter (
                float eps_angle_threshold_deg = 5.f, //0.25f
                float curvature_threshold = 0.04f,
                float cluster_tolerance = 0.01f, //0.015f;
                int min_points = 100, // 20
                bool z_adaptive = true,
                float octree_resolution = 0.01f,
                bool force_unorganized = false,
                bool compute_planar_patches_only = false,
                float planaer_inlier_dist = 0.02f
                )
            :
              eps_angle_threshold_deg_ (eps_angle_threshold_deg),
              curvature_threshold_ (curvature_threshold),
              cluster_tolerance_ (cluster_tolerance),
              min_points_ (min_points),
              z_adaptive_ ( z_adaptive ),
              octree_resolution_ ( octree_resolution ),
              force_unorganized_ ( force_unorganized ),
              compute_planar_patches_only_ (compute_planar_patches_only),
              planar_inlier_dist_ (planaer_inlier_dist)
        {
        }


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
            po::options_description desc("Smooth Region Growing Segmentation Parameters\n=====================");
            desc.add_options()
                    ("help,h", "produce help message")
                    ("min_cluster_size", po::value<int>(&min_points_)->default_value(min_points_), "")
                    ("sensor_noise_max", po::value<float>(&cluster_tolerance_)->default_value(cluster_tolerance_), "")
                    //                ("chop_z_segmentation", po::value<double>(&chop_z_)->default_value(chop_z_), "")
                    ("eps_angle_threshold", po::value<float>(&eps_angle_threshold_deg_)->default_value(eps_angle_threshold_deg_), "smooth clustering parameter for the angle threshold")
                    ("curvature_threshold", po::value<float>(&curvature_threshold_)->default_value(curvature_threshold_), "smooth clustering parameter for curvate")
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

    SmoothEuclideanSegmenter(const Parameter &p = Parameter() ) : param_(p)  { visualize_ = false; }

    bool getRequiresNormals() { return true; }

    void
    setSearchMethod(const typename pcl::octree::OctreePointCloudSearch<PointT>::Ptr &octree)
    {
        octree_ = octree;
    }

    void
    segment();

    typedef boost::shared_ptr< SmoothEuclideanSegmenter<PointT> > Ptr;
    typedef boost::shared_ptr< SmoothEuclideanSegmenter<PointT> const> ConstPtr;
};

}

#endif
