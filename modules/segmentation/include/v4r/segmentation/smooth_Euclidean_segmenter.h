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

public:
    class Parameter
    {
    public:
        float eps_angle_threshold_;
        float curvature_threshold_;
        float cluster_tolerance_;
        int min_points_;
        bool z_adaptive_;   /// @brief if true, scales the smooth segmentation parameters linear with distance (constant till 1m at the given parameters)
        float octree_resolution_;

        Parameter (
                float eps_angle_threshold = 0.1f, //0.25f
                float curvature_threshold = 0.04f,
                float cluster_tolerance = 0.01f, //0.015f;
                int min_points = 100, // 20
                bool z_adaptive = true,
                float octree_resolution = 0.01f
                )
            :
              eps_angle_threshold_ (eps_angle_threshold),
              curvature_threshold_ (curvature_threshold),
              cluster_tolerance_ (cluster_tolerance),
              min_points_ (min_points),
              z_adaptive_ ( z_adaptive ),
              octree_resolution_ ( octree_resolution )
        {
        }
    }param_;

    SmoothEuclideanSegmenter(const Parameter &p = Parameter() ) : param_(p)  { visualize_ = false; }

    SmoothEuclideanSegmenter(int argc, char **argv)
    {
        visualize_ = false;
        po::options_description desc("Dominant Plane Segmentation\n=====================");
        desc.add_options()
                ("help,h", "produce help message")
                ("min_cluster_size", po::value<int>(&param_.min_points_)->default_value(param_.min_points_), "")
                ("sensor_noise_max", po::value<float>(&param_.cluster_tolerance_)->default_value(param_.cluster_tolerance_), "")
//                ("chop_z_segmentation", po::value<double>(&param_.chop_z_)->default_value(param_.chop_z_), "")
                ("eps_angle_threshold", po::value<float>(&param_.eps_angle_threshold_)->default_value(param_.eps_angle_threshold_), "smooth clustering parameter for the angle threshold")
                ("curvature_threshold", po::value<float>(&param_.curvature_threshold_)->default_value(param_.curvature_threshold_), "smooth clustering parameter for curvate")
                ("visualize_segments", po::bool_switch(&visualize_), "If set, visualizes segmented clusters.")
                ;
        po::variables_map vm;
        po::parsed_options parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
        po::store(parsed, vm);
        if (vm.count("help")) { std::cout << desc << std::endl; }
        try { po::notify(vm); }
        catch(std::exception& e) {  std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl; }
    }


    bool getRequiresNormals() { return true; }

    void
    segment()
    {
        size_t max_pts_per_cluster = std::numeric_limits<int>::max();

        clusters_.clear();
        CHECK (scene_->points.size() == normals_->points.size ());

        // create an octree for search
        pcl::octree::OctreePointCloudSearch<PointT> octree (param_.octree_resolution_ );
        octree.setInputCloud(scene_);
        octree.addPointsFromInputCloud();

       // Create a bool vector of processed point indices, and initialize it to false
       std::vector<bool> processed (scene_->points.size (), false);
       std::vector<int> nn_indices;
       std::vector<float> nn_distances;
       // Process all points in the indices vector
       for (size_t i = 0; i < scene_->points.size (); ++i)
       {
         if (processed[i] || !pcl::isFinite(scene_->points[i]))
           continue;

         std::vector<size_t> seed_queue;
         size_t sq_idx = 0;
         seed_queue.push_back (i);

         processed[i] = true;

         while (sq_idx < seed_queue.size ())
         {

             size_t sidx = seed_queue[sq_idx];
           if (normals_->points[ sidx ].curvature > param_.curvature_threshold_)
           {
             sq_idx++;
             continue;
           }

           // Search for sq_idx - scale radius with distance of point (due to noise)
           float radius = param_.cluster_tolerance_;
           float curvature_threshold = param_.curvature_threshold_;
           float eps_angle_threshold = param_.eps_angle_threshold_;

           if ( param_.z_adaptive_ )
           {
               radius = param_.cluster_tolerance_ * ( 1 + (std::max(scene_->points[sidx].z, 1.f) - 1.f));
               curvature_threshold = param_.curvature_threshold_ * ( 1 + (std::max(scene_->points[sidx].z, 1.f) - 1.f));
               eps_angle_threshold = param_.eps_angle_threshold_ * ( 1 + (std::max(scene_->points[sidx].z, 1.f) - 1.f));
           }

           if (!octree.radiusSearch (sidx, radius, nn_indices, nn_distances))
           {
             sq_idx++;
             continue;
           }

           for (size_t j = 1; j < nn_indices.size (); ++j) // nn_indices[0] should be sq_idx
           {
             if (processed[nn_indices[j]]) // Has this point been processed before ?
               continue;

             if (normals_->points[nn_indices[j]].curvature > curvature_threshold)
               continue;

             //processed[nn_indices[j]] = true;
             // [-1;1]

             double dot_p = normals_->points[ sidx ].normal[0] * normals_->points[nn_indices[j]].normal[0]
                 + normals_->points[ sidx ].normal[1] * normals_->points[nn_indices[j]].normal[1] + normals_->points[sidx].normal[2]
                 * normals_->points[ nn_indices[j] ].normal[2];

             if (fabs (acos (dot_p)) < eps_angle_threshold)
             {
               processed[nn_indices[j]] = true;
               seed_queue.push_back (nn_indices[j]);
             }
           }

           sq_idx++;
         }

         // If this queue is satisfactory, add to the clusters
         if (seed_queue.size () >= param_.min_points_ && seed_queue.size () <= max_pts_per_cluster)
         {
             pcl::PointIndices r;
           r.indices.resize (seed_queue.size ());
           for (size_t j = 0; j < seed_queue.size (); ++j)
             r.indices[j] = seed_queue[j];

           std::sort (r.indices.begin (), r.indices.end ());
           r.indices.erase (std::unique (r.indices.begin (), r.indices.end ()), r.indices.end ());
           clusters_.push_back (r); // We could avoid a copy by working directly in the vector
         }
       }

       if (visualize_)
           this->visualize();
    }


    typedef boost::shared_ptr< SmoothEuclideanSegmenter<PointT> > Ptr;
    typedef boost::shared_ptr< SmoothEuclideanSegmenter<PointT> const> ConstPtr;
};

}

#endif
