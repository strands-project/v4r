#include <v4r/common/normal_estimator_pre-process.h>

#include <v4r/common/miscellaneous.h>
#include <pcl/point_types.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>

#include <pcl/impl/instantiate.hpp>
#include <glog/logging.h>

namespace v4r
{

template<typename PointT>
pcl::PointCloud<pcl::Normal>::Ptr
NormalEstimatorPreProcess<PointT>::compute()
{
    normal_.reset(new pcl::PointCloud<pcl::Normal>);
    processed_.reset(new pcl::PointCloud<PointT>);

    float mesh_resolution;

    if ( param_.compute_mesh_resolution_ )
        mesh_resolution = computeMeshResolution<PointT> (input_);

    if (param_.do_voxel_grid_)
    {
        //pcl::ScopeTime t ("Voxel grid...");
        float voxel_grid_size = param_.grid_resolution_;

        if ( param_.compute_mesh_resolution_)
            voxel_grid_size = mesh_resolution * param_.factor_voxel_grid_;

        pcl::VoxelGrid<PointT> grid;
        grid.setInputCloud (input_);
        grid.setLeafSize (voxel_grid_size, voxel_grid_size, voxel_grid_size);
        grid.setDownsampleAllData (true);
        grid.filter (*processed_);
    }
    else
        pcl::copyPointCloud(*input_, *processed_);

    if ( processed_->points.empty() )
    {
        PCL_WARN("NORMAL estimator: Cloud has no points after voxel grid, wont be able to compute normals!\n");
        return normal_;
    }

    if (param_.remove_outliers_)
    {
        typename pcl::PointCloud<PointT>::Ptr out2 (new pcl::PointCloud<PointT> ());
        float radius = param_.normal_radius_;
        if (param_.compute_mesh_resolution_)
        {
            radius = mesh_resolution * param_.factor_normals_;
            if (param_.do_voxel_grid_)
                radius *= param_.factor_voxel_grid_;
        }

        //in synthetic views the render grazes some parts of the objects
        //thus creating a very sparse set of points that causes the normals to be very noisy
        //remove these points
        pcl::RadiusOutlierRemoval<PointT> sor;
        sor.setInputCloud (processed_);
        sor.setRadiusSearch (radius);
        sor.setMinNeighborsInRadius (param_.min_n_radius_);
        sor.filter (*out2);
        processed_ = out2;

    }

    if (processed_->points.empty())
    {
        PCL_WARN("NORMAL estimator: Cloud has no points after removing outliers...!\n");
        return normal_;
    }

    float radius = param_.normal_radius_;
    if (param_.compute_mesh_resolution_)
    {
        radius = mesh_resolution * param_.factor_normals_;
        if (param_.do_voxel_grid_)
            radius *= param_.factor_voxel_grid_;
    }

    if (processed_->isOrganized () && !param_.force_unorganized_)
    {
        typedef typename pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> NormalEstimator_;
        NormalEstimator_ n3d;
        n3d.setNormalEstimationMethod (n3d.COVARIANCE_MATRIX);
        //n3d.setNormalEstimationMethod (n3d.AVERAGE_3D_GRADIENT);
        n3d.setInputCloud (processed_);
        n3d.setRadiusSearch (radius);
        n3d.setKSearch (0);
        //n3d.setMaxDepthChangeFactor(0.02f);
        //n3d.setNormalSmoothingSize(15.0f);
        n3d.compute (*normal_);
    }
    else
    {

        if( param_.only_on_indices_)
        {
            typename pcl::PointCloud<PointT>::Ptr indices_cloud(new pcl::PointCloud<PointT>);
            pcl::copyPointCloud(*processed_, indices_, *indices_cloud);
            pcl::NormalEstimationOMP<PointT, pcl::Normal> n3d;
            n3d.setRadiusSearch (radius);
            n3d.setInputCloud (indices_cloud);
            n3d.setSearchSurface(processed_);
            n3d.compute (*normal_);
            processed_ = indices_cloud;
        }
        else
        { //check nans before computing normals

            size_t kept = 0;
            for (size_t i = 0; i < processed_->points.size (); i++)
            {
                if ( !pcl::isFinite (processed_->points[i]) )
                    continue;

                processed_->points[kept] = processed_->points[i];
                kept++;
            }

            processed_->points.resize (kept);
            processed_->width = kept;
            processed_->height = 1;

            typename pcl::search::KdTree<PointT>::Ptr normals_tree (new pcl::search::KdTree<PointT>);
            normals_tree->setInputCloud (processed_);
            typename pcl::NormalEstimationOMP<PointT, pcl::Normal> n3d;
            n3d.setRadiusSearch (radius);
            n3d.setSearchMethod (normals_tree);
            n3d.setInputCloud (processed_);
            n3d.compute (*normal_);
        }
    }

    //check nans...
    if ( !processed_->isOrganized () )
    {
        size_t kept = 0;
        for (size_t i = 0; i < normal_->points.size (); ++i)
        {
            if ( !pcl::isFinite (normal_->points[i]) )
                continue;

            normal_->points[kept] = normal_->points[i];
            processed_->points[kept] = processed_->points[i];
            kept++;
        }

        normal_->points.resize (kept);
        normal_->width = kept;
        normal_->height = 1;

        processed_->points.resize (kept);
        processed_->width = kept;
        processed_->height = 1;
    }

    return normal_;
}


#define PCL_INSTANTIATE_NormalEstimatorPreProcess(T) template class V4R_EXPORTS NormalEstimatorPreProcess<T>;
PCL_INSTANTIATE(NormalEstimatorPreProcess, PCL_XYZ_POINT_TYPES )

}
