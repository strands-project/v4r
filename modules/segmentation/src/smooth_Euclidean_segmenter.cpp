#include <v4r/segmentation/smooth_Euclidean_segmenter.h>

namespace v4r
{

template<typename PointT>
void
SmoothEuclideanSegmenter<PointT>::segment()
{
    size_t max_pts_per_cluster = std::numeric_limits<int>::max();

    clusters_.clear();
    CHECK (scene_->points.size() == normals_->points.size ());

    if(!octree_ || octree_->getInputCloud() != scene_) {// create an octree for search
        octree_.reset( new pcl::octree::OctreePointCloudSearch<PointT> (param_.octree_resolution_ ) );
        octree_->setInputCloud(scene_);
        octree_->addPointsFromInputCloud();
    }

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
            const PointT &query_pt = scene_->points[ sidx ];
            const pcl::Normal &query_n = normals_->points[ sidx ];

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
                radius = param_.cluster_tolerance_ * ( 1 + (std::max(query_pt.z, 1.f) - 1.f));
                curvature_threshold = param_.curvature_threshold_ * ( 1 + (std::max(query_pt.z, 1.f) - 1.f));
                eps_angle_threshold = param_.eps_angle_threshold_ * ( 1 + (std::max(query_pt.z, 1.f) - 1.f));
            }

            if (!octree_->radiusSearch (query_pt, radius, nn_indices, nn_distances))
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

                Eigen::Vector3f n1 = query_n.getNormalVector3fMap();
                Eigen::Vector3f n2 = normals_->points[ nn_indices[j] ].getNormalVector3fMap();


                double dot_p = n1.dot(n2);

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

template class V4R_EXPORTS SmoothEuclideanSegmenter<pcl::PointXYZRGB>;
}
