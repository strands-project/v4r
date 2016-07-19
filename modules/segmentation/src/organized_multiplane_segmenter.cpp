#include <v4r/segmentation/organized_multiplane_segmenter.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>

namespace v4r
{

template<typename PointT>
void
OrganizedMultiplaneSegmenter<PointT>::segment()
{
    clusters_.clear();
    pcl::OrganizedMultiPlaneSegmentation<PointT, pcl::Normal, pcl::Label> mps;
    mps.setMinInliers (param_.num_plane_inliers_);
    mps.setAngularThreshold (param_.angular_threshold_deg_ * M_PI/180.f);
    mps.setDistanceThreshold (param_.sensor_noise_max_);
    mps.setInputNormals (normals_);
    mps.setInputCloud (scene_);

    std::vector < pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions;
    std::vector < pcl::ModelCoefficients > model_coeff;
    std::vector < pcl::PointIndices > inlier_indices;
    pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
    std::vector < pcl::PointIndices > label_indices;
    std::vector < pcl::PointIndices > boundary_indices;

    typename pcl::PlaneRefinementComparator<PointT, pcl::Normal, pcl::Label>::Ptr ref_comp (
                new pcl::PlaneRefinementComparator<PointT, pcl::Normal, pcl::Label> ());
    ref_comp->setDistanceThreshold (param_.sensor_noise_max_, false);
    ref_comp->setAngularThreshold (2 * M_PI/180.f);
    mps.setRefinementComparator (ref_comp);
    mps.segmentAndRefine (regions, model_coeff, inlier_indices, labels, label_indices, boundary_indices);

    std::cout << "Number of planes found:" << model_coeff.size () << std::endl;
    if ( !model_coeff.size() )
        return;

    size_t table_plane_selected = 0;
    int max_inliers_found = -1;
    std::vector<size_t> plane_inliers_counts;
    plane_inliers_counts.resize (model_coeff.size ());

    for (size_t i = 0; i < model_coeff.size (); i++)
    {
        Eigen::Vector4f table_plane = Eigen::Vector4f (model_coeff[i].values[0], model_coeff[i].values[1],
                model_coeff[i].values[2], model_coeff[i].values[3]);

        std::cout << "Number of inliers for this plane:" << inlier_indices[i].indices.size () << std::endl;
        size_t remaining_points = 0;
        typename pcl::PointCloud<PointT>::Ptr plane_points (new pcl::PointCloud<PointT> (*scene_));
        for (size_t j = 0; j < plane_points->points.size (); j++)
        {
            const Eigen::Vector3f xyz_p = plane_points->points[j].getVector3fMap ();

            if ( !pcl::isFinite( plane_points->points[j] ) )
                continue;

            float val = xyz_p[0] * table_plane[0] + xyz_p[1] * table_plane[1] + xyz_p[2] * table_plane[2] + table_plane[3];

            if (std::abs (val) > param_.sensor_noise_max_)
            {
                plane_points->points[j].x = std::numeric_limits<float>::quiet_NaN ();
                plane_points->points[j].y = std::numeric_limits<float>::quiet_NaN ();
                plane_points->points[j].z = std::numeric_limits<float>::quiet_NaN ();
            }
            else
                remaining_points++;
        }

        plane_inliers_counts[i] = remaining_points;

        if ( (int)remaining_points > max_inliers_found )
        {
            table_plane_selected = i;
            max_inliers_found = remaining_points;
        }
    }

    size_t itt = table_plane_selected;
    dominant_plane_ = Eigen::Vector4f (model_coeff[itt].values[0], model_coeff[itt].values[1], model_coeff[itt].values[2], model_coeff[itt].values[3]);
    Eigen::Vector3f normal_table = Eigen::Vector3f (model_coeff[itt].values[0], model_coeff[itt].values[1], model_coeff[itt].values[2]);

    size_t inliers_count_best = plane_inliers_counts[itt];

    //check that the other planes with similar normal are not higher than the table_plane_selected
    for (size_t i = 0; i < model_coeff.size (); i++)
    {
        Eigen::Vector4f model = Eigen::Vector4f (model_coeff[i].values[0], model_coeff[i].values[1], model_coeff[i].values[2],
                model_coeff[i].values[3]);

        Eigen::Vector3f normal = Eigen::Vector3f (model_coeff[i].values[0], model_coeff[i].values[1], model_coeff[i].values[2]);

        int inliers_count = plane_inliers_counts[i];

        std::cout << "Dot product is:" << normal.dot (normal_table) << std::endl;
        if ((normal.dot (normal_table) > 0.95) && (inliers_count_best * 0.5 <= inliers_count))
        {
            //check if this plane is higher, projecting a point on the normal direction
            std::cout << "Check if plane is higher, then change table plane" << std::endl;
            std::cout << model[3] << " " << dominant_plane_[3] << std::endl;
            if (model[3] < dominant_plane_[3])
            {
                PCL_WARN ("Changing table plane...");
                table_plane_selected = i;
                dominant_plane_ = model;
                normal_table = normal;
                inliers_count_best = inliers_count;
            }
        }
    }

    dominant_plane_ = Eigen::Vector4f (model_coeff[table_plane_selected].values[0], model_coeff[table_plane_selected].values[1],
            model_coeff[table_plane_selected].values[2], model_coeff[table_plane_selected].values[3]);

    typename pcl::EuclideanClusterComparator<PointT, pcl::Normal, pcl::Label>::Ptr
            euclidean_cluster_comparator_ (new pcl::EuclideanClusterComparator< PointT, pcl::Normal,pcl::Label> ());

    //create two labels, 1 one for points belonging to or under the plane, 1 for points above the plane
    label_indices.resize (2);

    for (size_t j = 0; j < scene_->points.size (); j++)
    {
        Eigen::Vector3f xyz_p = scene_->points[j].getVector3fMap ();

        if (! pcl::isFinite(scene_->points[j]) )
            continue;

        float val = xyz_p[0] * dominant_plane_[0] + xyz_p[1] * dominant_plane_[1] + xyz_p[2] * dominant_plane_[2] + dominant_plane_[3];

        if (val >= param_.sensor_noise_max_)
        {
            /*plane_points->points[j].x = std::numeric_limits<float>::quiet_NaN ();
     plane_points->points[j].y = std::numeric_limits<float>::quiet_NaN ();
     plane_points->points[j].z = std::numeric_limits<float>::quiet_NaN ();*/
            labels->points[j].label = 1;
            label_indices[0].indices.push_back (j);
        }
        else
        {
            labels->points[j].label = 0;
            label_indices[1].indices.push_back (j);
        }
    }

    std::vector<bool> plane_labels;
    plane_labels.resize (label_indices.size (), false);
    plane_labels[0] = true;

    euclidean_cluster_comparator_->setInputCloud (scene_);
    euclidean_cluster_comparator_->setLabels (labels);
    euclidean_cluster_comparator_->setExcludeLabels (plane_labels);
    euclidean_cluster_comparator_->setDistanceThreshold (0.035f, true);

    pcl::PointCloud < pcl::Label > euclidean_labels;
    std::vector < pcl::PointIndices > euclidean_label_indices;
    pcl::OrganizedConnectedComponentSegmentation<PointT, pcl::Label> euclidean_segmentation (euclidean_cluster_comparator_);
    euclidean_segmentation.setInputCloud (scene_);
    euclidean_segmentation.segment (euclidean_labels, euclidean_label_indices);

    for (size_t i = 0; i < euclidean_label_indices.size (); i++)
    {
        if ( (int)euclidean_label_indices[i].indices.size () >= param_.min_cluster_size_)
        {
            clusters_.push_back (euclidean_label_indices[i]);
        }
    }
}

template class V4R_EXPORTS OrganizedMultiplaneSegmenter<pcl::PointXYZRGB>;
}
