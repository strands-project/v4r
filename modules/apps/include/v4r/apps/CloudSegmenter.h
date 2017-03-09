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

#include <v4r/core/macros.h>
#include <v4r/common/normals.h>
#include <v4r/segmentation/all_headers.h>

#pragma once

namespace v4r
{

namespace apps
{

class V4R_EXPORTS CloudSegmenterParameter
{
public:
    float chop_z_; ///< cut-off distance in meter. Points further away than this threshold will be neglected
    float plane_inlier_threshold_; ///< threshold in meter for a point to be counted as inlier when computing the amount of inliers for a plane
    size_t min_plane_inliers_; ///< minimum number of inlier points for a plane to be valid
    bool skip_segmentation_;    ///< if true, skips segmentation
    bool skip_plane_extraction_;    ///< if true, skips plane extraction
    bool remove_planes_;   ///< if true, removes plane from input cloud only. If false, removes plane and everything below it (i.e. further away from the camera)
    bool remove_selected_plane_;   ///< if true, removes the selected plane (either dominant or the one parallel and higher)
    bool remove_points_below_selected_plane_;  ///< if true, removes any point not above selected plane (plane gets either selected by dominant (=plane with most inliers) or by the highest plane parallel to this dominant one)
    bool use_highest_plane_;  ///< if true, selects the highest plane parallel to the dominant plane
    float cosinus_angle_for_planes_to_be_parallel_;  ///< the minimum cosinus angle of the surface normals of two planes such that the two planes are considered parallel (only used if check for higher plane is enabled)
    float min_distance_to_plane_; ///< minimum distance in meter a point needs to have to be considered above

public:
    CloudSegmenterParameter()
        :
          chop_z_ (std::numeric_limits<float>::max()),
          plane_inlier_threshold_(0.02f),
          min_plane_inliers_(2000),
          skip_segmentation_(false),
          skip_plane_extraction_(false),
          remove_planes_(false),
          remove_selected_plane_(true),
          remove_points_below_selected_plane_(true),
          use_highest_plane_ (false),
          cosinus_angle_for_planes_to_be_parallel_(0.95f),
          min_distance_to_plane_(0.f)
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
        po::options_description desc("Cloud Segmentation Parameter\n=====================\n");
        desc.add_options()
                ("help,h", "produce help message")
                ("plane_inlier_threshold", po::value<float>(&plane_inlier_threshold_)->default_value(plane_inlier_threshold_), "inlier threshold for plane")
                ("chop_z,z", po::value<float>(&chop_z_)->default_value(chop_z_), "cut-off threshold in meter")
                ("min_plane_inliers", po::value<size_t>(&min_plane_inliers_)->default_value(min_plane_inliers_), "minimum number of inlier points for a plane to be valid")
                ("skip_segmentation", po::value<bool>(&skip_segmentation_)->default_value(skip_segmentation_), " if true, skips segmentation")
                ("remove_planes", po::value<bool>(&remove_planes_)->default_value(remove_planes_), "if true, removes plane from input cloud only. If false, removes plane and everything below it (i.e. further away from the camera)")
                ("remove_selected_plane", po::value<bool>(&remove_selected_plane_)->default_value(remove_selected_plane_), "if true, removes the selected plane (either dominant or the one parallel and higher)")
                ("remove_points_below_selected_plane", po::value<bool>(&remove_points_below_selected_plane_)->default_value(remove_points_below_selected_plane_), "if true, removes only the plane with the largest number of plane inliers")
                ("use_highest_plane", po::value<bool>(&use_highest_plane_)->default_value(use_highest_plane_), "if true, removes all points which are not above the highest plane parallel to the dominant plane")
                ("cosinus_angle_for_planes_to_be_parallel", po::value<float>(&cosinus_angle_for_planes_to_be_parallel_)->default_value(cosinus_angle_for_planes_to_be_parallel_), "the minimum cosinus angle of the surface normals of two planes such that the two planes are considered parallel (only used if check for higher plane is enabled)")
                ("min_distance_to_plane", po::value<float>(&min_distance_to_plane_)->default_value(min_distance_to_plane_), "minimum distance in meter a point needs to have to be considered above")
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
 * @brief The CloudSegmenter class segments an input cloud by first doing plane removal and then running the segmentation algorithm
 * @author Thomas Faeulhammer
 */

template<typename PointT>
class V4R_EXPORTS CloudSegmenter
{
private:
    typename v4r::PlaneExtractor<PointT>::Ptr plane_extractor_;
    typename v4r::Segmenter<PointT>::Ptr segmenter_;
    typename v4r::NormalEstimator<PointT>::Ptr normal_estimator_;
    std::vector<std::vector<int> > found_clusters_;
    pcl::PointCloud<pcl::Normal>::ConstPtr normals_;
    std::vector< Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > planes_;
    std::vector<std::vector<int> > plane_inliers_;
    typename pcl::PointCloud<PointT>::Ptr processed_cloud_;

    CloudSegmenterParameter param_;

public:
    CloudSegmenter(const CloudSegmenterParameter &p = CloudSegmenterParameter() ) :
        param_(p)
    { }

    /**
     * @brief initialize initialize Point Cloud Segmenter (sets up plane extraction, segmentation and potential normal estimator)
     * @param arguments
     */
    void initialize(std::vector<std::string> &command_line_arguments);

    /**
     * @brief recognize recognize objects in point cloud
     * @param cloud (organized) point cloud
     * @return
     */
    void
    segment(const typename pcl::PointCloud<PointT>::ConstPtr &cloud);

    /**
     * @brief getClusters
     * @param cluster_indices
     */
    std::vector<std::vector<int> >
    getClusters() const
    {
       return found_clusters_;
    }

    /**
     * @brief getPlanes
     * @return extracted planar surfaces
     */
    std::vector< Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> >
    getPlanes( ) const
    {
        return planes_;
    }

    /**
     * @brief setNormals sets the normals associated to the input cloud (if not set, they will be computed inside if neccessary)
     * @param normals
     */
    void
    setNormals(const typename pcl::PointCloud<pcl::Normal>::ConstPtr &normals)
    {
        normals_ = normals;
    }

    /**
     * @brief getPlaneInliers indices of the cloud that belong to a plane with given inlier threshold parameter (or as specified by the plane extraction method)
     * @return plane inliers
     */
    std::vector<std::vector<int> >
    getPlaneInliers() const
    {
        return plane_inliers_;
    }

    /**
     * @brief getProcessedCloud
     * @return processed/filtered cloud
     */
    typename pcl::PointCloud<PointT>::Ptr
    getProcessedCloud() const
    {
        return processed_cloud_;
    }


    typedef boost::shared_ptr< CloudSegmenter> Ptr;
    typedef boost::shared_ptr< CloudSegmenter const> ConstPtr;
};

}

}
