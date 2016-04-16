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
*      @brief multiplane segmentation (taken from PCL)
*/

#ifndef V4R_MULTIPLANE_SEGMENTER_H__
#define V4R_MULTIPLANE_SEGMENTER_H__

#include <v4r/core/macros.h>
#include <v4r/segmentation/segmenter.h>
#include <v4r/segmentation/multiplane_segmentation.h>
#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;
namespace v4r
{

template <typename PointT>
class V4R_EXPORTS MultiplaneSegmenter: public Segmenter<PointT>
{
    using Segmenter<PointT>::indices_;
    using Segmenter<PointT>::normals_;
    using Segmenter<PointT>::clusters_;
    using Segmenter<PointT>::scene_;
    using Segmenter<PointT>::table_plane_;

public:
    class Parameter
    {
    public:
        int min_cluster_size_, num_plane_inliers_;
        double sensor_noise_max_,
               angular_threshold_deg_;
        Parameter (int min_cluster_size=500,
                   int num_plane_inliers=1000,
                   double sensor_noise_max = 0.01f,
                   double angular_threshold_deg = 10.f)
            :
              min_cluster_size_ (min_cluster_size),
              num_plane_inliers_ (num_plane_inliers),
              sensor_noise_max_ (sensor_noise_max),
              angular_threshold_deg_ (angular_threshold_deg)
        {
        }
    }param_;

    MultiplaneSegmenter(const Parameter &p = Parameter() ) : param_(p)
    {
    }

    MultiplaneSegmenter(int argc, char **argv)
    {
        po::options_description desc("Multi-Plane Segmentation\n=====================");
        desc.add_options()
                ("help,h", "produce help message")
                ("min_cluster_size", po::value<int>(&param_.min_cluster_size_)->default_value(param_.min_cluster_size_), "")
                ("num_plane_inliers", po::value<int>(&param_.num_plane_inliers_)->default_value(param_.num_plane_inliers_), "")
                ("sensor_noise_max", po::value<double>(&param_.sensor_noise_max_)->default_value(param_.sensor_noise_max_), "")
                ("angular_threshold_deg", po::value<double>(&param_.angular_threshold_deg_)->default_value(param_.angular_threshold_deg_), "")
                ;
        po::variables_map vm;
        po::parsed_options parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
        po::store(parsed, vm);
        if (vm.count("help")) { std::cout << desc << std::endl; }
        try { po::notify(vm); }
        catch(std::exception& e) {  std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl; }
    }

    void
    segment();

    typedef boost::shared_ptr< MultiplaneSegmenter<PointT> > Ptr;
    typedef boost::shared_ptr< MultiplaneSegmenter<PointT> const> ConstPtr;
};

}

#endif
