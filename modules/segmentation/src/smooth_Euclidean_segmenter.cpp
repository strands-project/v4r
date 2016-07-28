#include <pcl/common/angles.h>
#include <v4r/common/miscellaneous.h>
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

    float eps_angle_threshold_rad = pcl::deg2rad(param_.eps_angle_threshold_deg_);

    // Process all points in the indices vector
    for (size_t i = 0; i < scene_->points.size (); ++i)
    {
        if (processed[i] || !pcl::isFinite(scene_->points[i]))
            continue;

        std::vector<size_t> seed_queue;
        size_t sq_idx = 0;
        seed_queue.push_back (i);

        processed[i] = true;

        // this is used if planar surface extraction only is enabled
        Eigen::Vector3f avg_normal = normals_->points[i].getNormalVector3fMap();
        Eigen::Vector3f avg_plane_pt = scene_->points[i].getVector3fMap();

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
            float eps_angle_threshold = eps_angle_threshold_rad;

            if ( param_.z_adaptive_ )
            {
                radius = param_.cluster_tolerance_ * ( 1 + (std::max(query_pt.z, 1.f) - 1.f));
                curvature_threshold = param_.curvature_threshold_ * ( 1 + (std::max(query_pt.z, 1.f) - 1.f));
                eps_angle_threshold = eps_angle_threshold_rad * ( 1 + (std::max(query_pt.z, 1.f) - 1.f));
            }

            if (!scene_->isOrganized() && !param_.force_unorganized_)
            {
                if(!octree_->radiusSearch (query_pt, radius, nn_indices, nn_distances))
                {
                    sq_idx++;
                    continue;
                }
            }
            else    // check pixel neighbors
            {
                int width = scene_->width;
                int height = scene_->height;
                int u = sidx%width;
                int v = sidx/width;

                nn_indices.resize(8);
                nn_distances.resize(8);
                size_t kept=0;
                for(int shift_u=-1; shift_u<=1; shift_u++)
                {
                    int uu = u + shift_u;
                    if ( uu < 0 || uu>=width)
                        continue;

                    for(int shift_v=-1; shift_v<=1; shift_v++)
                    {
                        int vv = v + shift_v;
                        if ( vv < 0 || vv>=height)
                            continue;

                        int nn_idx = vv*width + uu;
                        float dist = ( scene_->points[sidx].getVector3fMap() - scene_->points[nn_idx].getVector3fMap() ).norm();
                        if (dist < radius)
                        {
                            nn_indices[kept] = nn_idx;
                            nn_distances[kept] = dist;
                            kept++;
                        }
                    }
                }
                nn_indices.resize(kept);
                nn_distances.resize(kept);
            }

            for (size_t j = 0; j < nn_indices.size (); j++)
            {
                if ( processed[nn_indices[j]] ) // Has this point been processed before ?
                    continue;

                if (normals_->points[nn_indices[j]].curvature > curvature_threshold)
                    continue;

                Eigen::Vector3f n1;
                if(param_.compute_planar_patches_only_)
                    n1 = avg_normal;
                else
                    n1 = query_n.getNormalVector3fMap();

                pcl::Normal nn = normals_->points[ nn_indices[j] ];
                const Eigen::Vector3f &n2 = nn.getNormalVector3fMap();

                double dot_p = n1.dot(n2);

                if (fabs (dot_p) > cos(eps_angle_threshold))
                {
                    if(param_.compute_planar_patches_only_)
                    {
                        const Eigen::Vector3f &nn_pt = scene_->points[ nn_indices[j] ].getVector3fMap();
                        float dist = fabs(avg_normal.dot(nn_pt - avg_plane_pt));

                        if(dist > param_.planar_inlier_dist_)
                            continue;

                        runningAverage( avg_normal, seed_queue.size(), n2 );
                        avg_normal.normalize();
                        runningAverage( avg_plane_pt, seed_queue.size(), nn_pt );
                    }

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
