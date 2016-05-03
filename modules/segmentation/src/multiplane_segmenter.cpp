#include <v4r/segmentation/multiplane_segmenter.h>
#include <v4r/segmentation/multiplane_segmentation.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>

namespace v4r
{

template <typename PointT>
void
MultiplaneSegmenter<PointT>::computeTablePlanes()
{
    all_planes_.clear();
    pcl::OrganizedMultiPlaneSegmentation<PointT, pcl::Normal, pcl::Label> mps;
    mps.setMinInliers (param_.num_plane_inliers_);
    mps.setAngularThreshold ( pcl::deg2rad(param_.angular_threshold_deg_) );
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
    ref_comp->setAngularThreshold ( pcl::deg2rad(2.f) );
    mps.setRefinementComparator (ref_comp);
    mps.segmentAndRefine (regions, model_coeff, inlier_indices, labels, label_indices, boundary_indices);

    std::cout << "Number of planes found:" << model_coeff.size () << std::endl;

    all_planes_.resize ( model_coeff.size() );
    for (size_t i = 0; i < model_coeff.size (); i++)
    {
        // flip normal of plane towards viewpoint
        Eigen::Vector3f vp;
        vp(0)=vp(1)=0.f; vp(2) = 1;

        Eigen::Vector4f plane_tmp = Eigen::Vector4f(model_coeff[i].values[0], model_coeff[i].values[1], model_coeff[i].values[2], model_coeff[i].values[3]);
        Eigen::Vector3f table_vec = plane_tmp.head(3);
        if(vp.dot(table_vec)>0)
            plane_tmp *= -1.f;

        all_planes_[i].reset( new PlaneModel<PointT>);
        all_planes_[i]->coefficients_ = plane_tmp;
        all_planes_[i]->inliers_ = inlier_indices[i].indices;
        all_planes_[i]->cloud_ = scene_;


        typename pcl::PointCloud<PointT>::Ptr plane_cloud (new pcl::PointCloud<PointT>);
        typename pcl::PointCloud<PointT>::Ptr above_plane_cloud (new pcl::PointCloud<PointT>);
        pcl::copyPointCloud(*scene_, inlier_indices[i], *plane_cloud);

        double z_min = 0., z_max = 0.30; // we want the points above the plane, no farther than zmax cm from the surface
        typename pcl::PointCloud<PointT>::Ptr hull_points = all_planes_[i]->getConvexHullCloud();

        pcl::PointIndices cloud_indices;
        pcl::ExtractPolygonalPrismData<PointT> prism;
        prism.setInputCloud (scene_);
        prism.setInputPlanarHull (hull_points);
        prism.setHeightLimits (z_min, z_max);
        prism.segment (cloud_indices);

        pcl::copyPointCloud(*scene_, cloud_indices, *above_plane_cloud);

        pcl::visualization::PCLVisualizer::Ptr vis;
        int vp1, vp2;
        if(!vis)
        {
            vis.reset (new pcl::visualization::PCLVisualizer("plane22 visualization"));
            vis->createViewPort(0,0,0.5,1,vp1);
            vis->createViewPort(0.5,0,1,1,vp2);
        }
        vis->removeAllPointClouds();
        vis->removeAllShapes();
        vis->addPointCloud(scene_, "cloud", vp1);

        vis->addPointCloud(above_plane_cloud, "convex_hull", vp2);
        vis->spin();

        all_planes_[i]->visualize();
    }
}

template <typename PointT>
void
MultiplaneSegmenter<PointT>::segment()
{
    clusters_.clear();
    computeTablePlanes();


    typename pcl::EuclideanClusterComparator<PointT, pcl::Normal, pcl::Label>::Ptr
            euclidean_cluster_comparator_ (new pcl::EuclideanClusterComparator< PointT, pcl::Normal,pcl::Label> ());

    pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
    labels->points.resize( scene_->points.size() );
    std::vector < pcl::PointIndices > label_indices(2);
    std::vector<bool> plane_labels;
    plane_labels.resize (label_indices.size (), false);

    if ( !all_planes_.empty() )
    {
        size_t table_plane_selected = 0;
        size_t max_inliers_found = 0;
        std::vector<size_t> plane_inliers_counts (all_planes_.size (), 0);

        for (size_t i = 0; i < all_planes_.size (); i++)
        {
            const Eigen::Vector3f plane_normal = all_planes_[i]->coefficients_.head(3);
            for (size_t j = 0; j < scene_->points.size (); j++)
            {
                if ( !pcl::isFinite( scene_->points[j] ) )
                    continue;

                const Eigen::Vector3f &xyz_p = scene_->points[j].getVector3fMap ();

                if (std::abs ( xyz_p.dot(plane_normal) ) < param_.sensor_noise_max_)
                    plane_inliers_counts[i]++;
            }

            if ( plane_inliers_counts[i] > max_inliers_found )
            {
                table_plane_selected = i;
                max_inliers_found = plane_inliers_counts[i];
            }
        }

        size_t itt = table_plane_selected;
        dominant_plane_ = all_planes_[itt]->coefficients_;
        Eigen::Vector3f normal_table = all_planes_[itt]->coefficients_.head(3);
        size_t inliers_count_best = plane_inliers_counts[itt];

        //check that the other planes with similar normal are not higher than the table_plane_selected
        for (size_t i = 0; i < all_planes_.size (); i++)
        {
            Eigen::Vector4f model = all_planes_[i]->coefficients_;
            Eigen::Vector3f normal_tmp = all_planes_[i]->coefficients_.head(3);

            int inliers_count = plane_inliers_counts[i];

            if ((normal_tmp.dot (normal_table) > 0.95) && (inliers_count_best * 0.5 <= inliers_count))
            {
                //check if this plane is higher, projecting a point on the normal direction
                std::cout << model[3] << " " << dominant_plane_[3] << std::endl;
                if (model[3] < dominant_plane_[3])
                {
                    PCL_WARN ("Changing table plane...");
                    dominant_plane_ = all_planes_[i]->coefficients_;
                    normal_table = normal_tmp;
                    inliers_count_best = inliers_count;
                }
            }
        }

        dominant_plane_ = all_planes_[table_plane_selected]->coefficients_;

        //create two labels, 1 one for points belonging to or under the plane, 1 for points above the plane

        for (size_t j = 0; j < scene_->points.size (); j++)
        {
            const Eigen::Vector3f &xyz_p = scene_->points[j].getVector3fMap ();

            if (! pcl::isFinite(scene_->points[j]) )
                continue;

            float val = xyz_p[0] * dominant_plane_[0] + xyz_p[1] * dominant_plane_[1] + xyz_p[2] * dominant_plane_[2] + dominant_plane_[3];

            if (val >= param_.sensor_noise_max_)
            {
                labels->points[j].label = 1;
                label_indices[0].indices.push_back (j);
            }
            else
            {
                labels->points[j].label = 0;
                label_indices[1].indices.push_back (j);
            }
        }

        plane_labels[0] = true;
    }
    else
    {
        for (size_t j = 0; j < scene_->points.size (); j++)
            labels->points[j].label = 1;
    }

    euclidean_cluster_comparator_->setInputCloud (scene_);
    euclidean_cluster_comparator_->setLabels (labels);
    euclidean_cluster_comparator_->setExcludeLabels (plane_labels);
    euclidean_cluster_comparator_->setDistanceThreshold ( param_.min_distance_between_clusters_, true);

    pcl::PointCloud < pcl::Label > euclidean_labels;
    std::vector < pcl::PointIndices > euclidean_label_indices;
    pcl::OrganizedConnectedComponentSegmentation<PointT, pcl::Label> euclidean_segmentation (euclidean_cluster_comparator_);
    euclidean_segmentation.setInputCloud (scene_);
    euclidean_segmentation.segment (euclidean_labels, euclidean_label_indices);

    for (size_t i = 0; i < euclidean_label_indices.size (); i++)
    {
        if ( euclidean_label_indices[i].indices.size () >= param_.min_cluster_size_)
        {
            clusters_.push_back (euclidean_label_indices[i]);
        }
    }
}

template class V4R_EXPORTS MultiplaneSegmenter<pcl::PointXYZRGB>;
}
