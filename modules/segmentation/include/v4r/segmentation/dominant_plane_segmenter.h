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
*      @brief dominant plane segmentation (taken from PCL)
*/

#ifndef V4R_DOMINANT_PLANE_SEGMENTER_H__
#define V4R_DOMINANT_PLANE_SEGMENTER_H__

#include <v4r/core/macros.h>
#include <v4r/segmentation/segmenter.h>
#include <pcl/apps/dominant_plane_segmentation.h>

#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;

namespace v4r
{

template <typename PointT>
class V4R_EXPORTS DominantPlaneSegmenter : public Segmenter<PointT>
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
        int min_cluster_size_;
        double object_min_height_;
        double object_max_height_;
        double chop_z_;
        float min_distance_between_clusters_;
        int w_size_px_;
        float downsampling_size_;
        Parameter (int min_cluster_size=500,
                   double object_min_height = 0.01f,
                   double object_max_height = 0.7f,
                   double chop_at_z = 3.f,
                   float min_distance_between_clusters = 0.03f,
                   int w_size_px = 5,
                   float downsampling_size = 0.005f
                )
            :
              min_cluster_size_ (min_cluster_size),
              object_min_height_ (object_min_height),
              object_max_height_ (object_max_height),
              chop_z_ (chop_at_z),
              min_distance_between_clusters_ (min_distance_between_clusters),
              w_size_px_ (w_size_px),
              downsampling_size_ (downsampling_size)
        {

        }
    }param_;

    DominantPlaneSegmenter(const Parameter &p = Parameter() ) : param_(p)  { }

    DominantPlaneSegmenter(int argc, char **argv)
    {
        po::options_description desc("Dominant Plane Segmentation\n=====================");
        desc.add_options()
                ("help,h", "produce help message")
                ("min_cluster_size", po::value<int>(&param_.min_cluster_size_)->default_value(param_.min_cluster_size_), "")
                ("sensor_noise_max", po::value<double>(&param_.object_min_height_)->default_value(param_.object_min_height_), "")
                ("chop_z_segmentation", po::value<double>(&param_.chop_z_)->default_value(param_.chop_z_), "")
                ("min_distance_between_clusters", po::value<float>(&param_.min_distance_between_clusters_)->default_value(param_.min_distance_between_clusters_), "")
                ("w_size_px", po::value<int>(&param_.w_size_px_)->default_value(param_.w_size_px_), "")
                ("downsampling_size", po::value<float>(&param_.downsampling_size_)->default_value(param_.downsampling_size_), "")
                ;
        po::variables_map vm;
        po::parsed_options parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
        po::store(parsed, vm);
        if (vm.count("help")) { std::cout << desc << std::endl; }
        try { po::notify(vm); }
        catch(std::exception& e) {  std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl; }
    }

    void
    segment()
    {
        clusters_.clear();
        pcl::apps::DominantPlaneSegmentation<PointT> dps;
        dps.setInputCloud (scene_);
        dps.setMaxZBounds (param_.chop_z_);
        dps.setObjectMinHeight (param_.object_min_height_);
        dps.setObjectMaxHeight (param_.object_max_height_);
        dps.setMinClusterSize (param_.min_cluster_size_);
        dps.setWSize (param_.w_size_px_);
        dps.setDistanceBetweenClusters (param_.min_distance_between_clusters_);
        std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;
        dps.setDownsamplingSize ( param_.downsampling_size_ );
        dps.compute_fast (clusters);
        dps.getIndicesClusters (clusters_);
        dps.getTableCoefficients (table_plane_);
    }


    typedef boost::shared_ptr< DominantPlaneSegmenter<PointT> > Ptr;
    typedef boost::shared_ptr< DominantPlaneSegmenter<PointT> const> ConstPtr;
};

}

#endif
