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


#ifndef V4R_ISS_KEYPOINT_EXTRACTOR__
#define V4R_ISS_KEYPOINT_EXTRACTOR__

#include <v4r/keypoints/keypoint_extractor.h>

namespace v4r
{
template<typename PointT>
class V4R_EXPORTS IssKeypointExtractor : public KeypointExtractor<PointT>
{
private:
    typedef typename pcl::PointCloud<PointT>::Ptr PointInTPtr;
    using KeypointExtractor<PointT>::input_;
    using KeypointExtractor<PointT>::indices_;
    using KeypointExtractor<PointT>::keypoint_indices_;
    using KeypointExtractor<PointT>::keypoint_extractor_type_;
    using KeypointExtractor<PointT>::keypoint_extractor_name_;

public:
    class Parameter // see PCL documentation for further details
    {
        public:

        double salient_radius_; /// @brief Set the radius of the spherical neighborhood used to compute the scatter matrix.
        double non_max_radius_; /// @brief Set the radius for the application of the non maxima supression algorithm.
        double normal_radius_; /// @brief Set the radius used for the estimation of the surface normals of the input cloud. If the radius is too large, the temporal performances of the detector may degrade significantly. Only used if parameter with_border_estimation equal true.
        double border_radius_; /// @brief Set the radius used for the estimation of the boundary points. If the radius is too large, the temporal performances of the detector may degrade significantly. Only used if parameter with_border_estimation equal true.
        double gamma_21_; /// @brief Set the upper bound on the ratio between the second and the first eigenvalue.
        double gamma_32_; /// @brief Set the upper bound on the ratio between the third and the second eigenvalue.
        double min_neighbors_ ; /// @brief Set the minimum number of neighbors that has to be found while applying the non maxima suppression algorithm.
        bool with_border_estimation_;
        int threads_;

        Parameter(
                double salient_radius = 6*0.005f,   // 6 * model resolution (according to http://www.pointclouds.org/blog/gsoc12/gballin/iss.php)
                double non_max_radius = 4*0.005f,
                double normal_radius = 4*0.005f,
                double border_radius = 1*0.005f,
                double gamma_21 = 0.975,
                double gamma_32 = 0.975,
                double min_neighbors = 5,
                bool with_border_estimation = true,
                int threads = 0
                ) :
            salient_radius_ (salient_radius),
            non_max_radius_ (non_max_radius),
            normal_radius_ (normal_radius),
            border_radius_ (border_radius),
            gamma_21_ (gamma_21),
            gamma_32_ (gamma_32),
            min_neighbors_ (min_neighbors),
            with_border_estimation_ (with_border_estimation),
            threads_ (threads)
        {}
    }param_;

    IssKeypointExtractor(const Parameter &p = Parameter()) : param_ (p)
    {

        keypoint_extractor_type_ = KeypointType::ISS;
        keypoint_extractor_name_ = "iss";
    }

    void
    compute (pcl::PointCloud<PointT> & keypoints);

    typedef boost::shared_ptr< IssKeypointExtractor<PointT> > Ptr;
    typedef boost::shared_ptr< IssKeypointExtractor<PointT> const> ConstPtr;
};
}

#endif
