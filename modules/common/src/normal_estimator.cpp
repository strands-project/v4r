#include <v4r/common/normal_estimator.h>
#include <v4r/common/miscellaneous.h>
#include <pcl/point_types.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/common/time.h>
#include <pcl/common/io.h>
#include <pcl/impl/instantiate.hpp>

namespace v4r
{

template<typename PointT, typename PointOutT>
void
PreProcessorAndNormalEstimator<PointT, PointOutT>::estimate (const typename pcl::PointCloud<PointT>::ConstPtr & in, PointInTPtr & out, pcl::PointCloud<pcl::Normal>::Ptr & normals)
{
    float mesh_resolution;

    if (compute_mesh_resolution_)
        mesh_resolution = computeMeshResolution<PointT> (in);

    if (do_voxel_grid_)
    {
        //pcl::ScopeTime t ("Voxel grid...");
        float voxel_grid_size = grid_resolution_;

        if (compute_mesh_resolution_)
            voxel_grid_size = mesh_resolution * factor_voxel_grid_;

        pcl::VoxelGrid<PointT> grid_;
        grid_.setInputCloud (in);
        grid_.setLeafSize (voxel_grid_size, voxel_grid_size, voxel_grid_size);
        grid_.setDownsampleAllData (true);
        grid_.filter (*out);
    }
    else
        pcl::copyPointCloud(*in, *out);

    if ( out->points.empty() )
    {
        PCL_WARN("NORMAL estimator: Cloud has no points after voxel grid, wont be able to compute normals!\n");
        return;
    }

    if (remove_outliers_)
    {
        PointInTPtr out2 (new pcl::PointCloud<PointT> ());
        float radius = normal_radius_;
        if (compute_mesh_resolution_)
        {
            radius = mesh_resolution * factor_normals_;
            if (do_voxel_grid_)
                radius *= factor_voxel_grid_;
        }

        //in synthetic views the render grazes some parts of the objects
        //thus creating a very sparse set of points that causes the normals to be very noisy
        //remove these points
        pcl::RadiusOutlierRemoval<PointT> sor;
        sor.setInputCloud (out);
        sor.setRadiusSearch (radius);
        sor.setMinNeighborsInRadius (min_n_radius_);
        sor.filter (*out2);
        out = out2;

    }

    if (out->points.empty())
    {
        PCL_WARN("NORMAL estimator: Cloud has no points after removing outliers...!\n");
        return;
    }

    float radius = normal_radius_;
    if (compute_mesh_resolution_)
    {
        radius = mesh_resolution * factor_normals_;
        if (do_voxel_grid_)
            radius *= factor_voxel_grid_;
    }

    if (out->isOrganized () && !force_unorganized_)
    {
        typedef typename pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> NormalEstimator_;
        NormalEstimator_ n3d;
        n3d.setNormalEstimationMethod (n3d.COVARIANCE_MATRIX);
        //n3d.setNormalEstimationMethod (n3d.AVERAGE_3D_GRADIENT);
        n3d.setInputCloud (out);
        n3d.setRadiusSearch (radius);
        n3d.setKSearch (0);
        //n3d.setMaxDepthChangeFactor(0.02f);
        //n3d.setNormalSmoothingSize(15.0f);
        n3d.compute (*normals);
    }
    else
    {

        if(only_on_indices_)
        {
            PointInTPtr indices_cloud(new pcl::PointCloud<PointT>);
            pcl::copyPointCloud(*out, indices_, *indices_cloud);
            typedef typename pcl::NormalEstimationOMP<PointT, pcl::Normal> NormalEstimator_;
            NormalEstimator_ n3d;
            n3d.setRadiusSearch (radius);
            n3d.setInputCloud (indices_cloud);
            n3d.setSearchSurface(out);
            n3d.compute (*normals);
            out = indices_cloud;
        }
        else
        { //check nans before computing normals

            size_t kept = 0;
            for (size_t i = 0; i < out->points.size (); i++)
            {
                if ( !pcl::isFinite (out->points[i]) )
                    continue;

                out->points[kept] = out->points[i];
                kept++;
            }

            out->points.resize (kept);
            out->width = kept;
            out->height = 1;

            typedef typename pcl::NormalEstimationOMP<PointT, pcl::Normal> NormalEstimator_;
            NormalEstimator_ n3d;
            typename pcl::search::KdTree<PointT>::Ptr normals_tree (new pcl::search::KdTree<PointT>);
            normals_tree->setInputCloud (out);
            n3d.setRadiusSearch (radius);
            n3d.setSearchMethod (normals_tree);
            n3d.setInputCloud (out);
            n3d.compute (*normals);
        }
    }

    //check nans...
    if ( !out->isOrganized () )
    {
        int j = 0;
        for (size_t i = 0; i < normals->points.size (); ++i)
        {
            if (!pcl_isfinite (normals->points[i].normal_x) || !pcl_isfinite (normals->points[i].normal_y)
                    || !pcl_isfinite (normals->points[i].normal_z))
                continue;

            normals->points[j] = normals->points[i];
            out->points[j] = out->points[i];
            j++;
        }

        normals->points.resize (j);
        normals->width = j;
        normals->height = 1;

        out->points.resize (j);
        out->width = j;
        out->height = 1;
    }
}

#define PCL_INSTANTIATE_PreProcessorAndNormalEstimator(A,B) template class V4R_EXPORTS PreProcessorAndNormalEstimator<A,B>;
PCL_INSTANTIATE_PRODUCT(PreProcessorAndNormalEstimator, (PCL_XYZ_POINT_TYPES)((pcl::Normal)) )

}
