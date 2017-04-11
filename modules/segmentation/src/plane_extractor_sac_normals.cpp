#include <v4r/segmentation/plane_extractor_sac_normals.h>

#include <pcl/impl/instantiate.hpp>
#include <glog/logging.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/kdtree/kdtree.h>


#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/console/parse.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/pcl_search.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/segmentation/extract_clusters.h>

namespace v4r
{

template<typename PointT>
void
SACNormalsPlaneExtractor<PointT>::compute()
{
    CHECK ( cloud_  ) << "Input cloud is not organized!";

    all_planes_.clear();

    // ---[ PassThroughFilter
    typename pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT> ());
    pcl::PassThrough<PointT> pass;
    pass.setFilterLimits (0, max_z_bounds_);
    pass.setFilterFieldName ("z");
    pass.setInputCloud (cloud_);
    pass.filter (*cloud_filtered);

    if ( cloud_filtered->points.size () < k_)
    {
      PCL_WARN ("[DominantPlaneSegmentation] Filtering returned %lu points! Aborting.",
          cloud_filtered->points.size ());
      return;
    }

    // Downsample the point cloud
    typename pcl::PointCloud<PointT>::Ptr cloud_downsampled (new pcl::PointCloud<PointT> ());
    pcl::VoxelGrid<PointT> grid;
    grid.setLeafSize (downsample_leaf_, downsample_leaf_, downsample_leaf_);
    grid.setDownsampleAllData (false);
    grid.setInputCloud (cloud_filtered);
    grid.filter (*cloud_downsampled);

    // ---[ Estimate the point normals
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal> ());
    pcl::NormalEstimation<PointT, pcl::Normal> n3d;
    typename pcl::search::KdTree<PointT>::Ptr normals_tree_ (new pcl::search::KdTree<PointT>);
    n3d.setKSearch ( (int) k_);
    n3d.setSearchMethod (normals_tree_);
    n3d.setInputCloud (cloud_downsampled);
    n3d.compute (*cloud_normals);

    // ---[ Perform segmentation
    pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
    seg.setDistanceThreshold (sac_distance_threshold_);
    seg.setMaxIterations (2000);
    seg.setNormalDistanceWeight (0.1);
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setProbability (0.99);
    seg.setInputCloud (cloud_downsampled);
    seg.setInputNormals (cloud_normals);
    pcl::PointIndices table_inliers;
    pcl::ModelCoefficients coefficients;
    seg.segment ( table_inliers, coefficients);

    Eigen::Vector4f plane = Eigen::Vector4f(coefficients.values[0], coefficients.values[1],
                    coefficients.values[2], coefficients.values[3]);

    all_planes_.resize(1);
    all_planes_[0] = plane;
}

#define PCL_INSTANTIATE_SACNormalsPlaneExtractor(T) template class V4R_EXPORTS SACNormalsPlaneExtractor<T>;
PCL_INSTANTIATE(SACNormalsPlaneExtractor, PCL_XYZ_POINT_TYPES )

}
