#include <v4r/common/normals.h>
#include <v4r/common/ZAdaptiveNormals.h>
#include <v4r/common/normal_estimator.h>
#include <v4r/common/convertCloud.h>
#include <v4r/common/convertNormals.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>
#include <glog/logging.h>

namespace v4r
{
template<typename PointT>
V4R_EXPORTS
void computeNormals(const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
                    pcl::PointCloud<pcl::Normal>::Ptr &normals,
                    int method, float radius)
{
    CHECK(normals);

    if(method == 0)
    {
        pcl::NormalEstimation<PointT, pcl::Normal> n3d;
        n3d.setRadiusSearch (radius);
        n3d.setInputCloud (cloud);
        n3d.compute (*normals);
    }
    else if(method == 1)
    {
        pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
        ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
        ne.setMaxDepthChangeFactor(0.02f);
        ne.setNormalSmoothingSize(15.f);//20.0f);
        ne.setDepthDependentSmoothing(false);//param.normals_depth_dependent_smoothing);
        ne.setInputCloud(cloud);
        ne.setViewPoint(0,0,0);
        ne.compute(*normals);
    }
    else if(method == 2)
    {
        pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
        ne.setRadiusSearch ( radius );
        ne.setInputCloud ( cloud );
        ne.compute ( *normals );
    }
    else if(method == 3)
    {
        ZAdaptiveNormals::Parameter n_param;
        n_param.adaptive = true;
        ZAdaptiveNormals nest(n_param);

        DataMatrix2D<Eigen::Vector3f>::Ptr kp_cloud( new DataMatrix2D<Eigen::Vector3f>() );
        DataMatrix2D<Eigen::Vector3f>::Ptr kp_normals_tmp( new DataMatrix2D<Eigen::Vector3f>() );
        convertCloud(*cloud, *kp_cloud);
        nest.compute(*kp_cloud, *kp_normals_tmp);
        convertNormals(*kp_normals_tmp, *normals);
    }
    else if(method==4)
    {
        boost::shared_ptr<PreProcessorAndNormalEstimator<PointT, pcl::Normal> > normal_estimator;
        normal_estimator.reset (new PreProcessorAndNormalEstimator<PointT, pcl::Normal>);
        normal_estimator->setCMR (false);
        normal_estimator->setDoVoxelGrid (false);
        normal_estimator->setRemoveOutliers (false);
        normal_estimator->setValuesForCMRFalse (0.003f, 0.02f);
        normal_estimator->setForceUnorganized(true);

        typename pcl::PointCloud<PointT>::Ptr processed (new pcl::PointCloud<PointT>);
        normal_estimator->estimate (cloud, processed, normals);
    }
    else
    {
        throw std::runtime_error("Chosen normal computation method not implemented!");
    }

    // Normalize normals to unit length
    for ( size_t normal_pt_id = 0; normal_pt_id < normals->points.size(); normal_pt_id++)
    {
        Eigen::Vector3f n1 = normals->points[normal_pt_id].getNormalVector3fMap();
        n1.normalize();
        normals->points[normal_pt_id].normal_x = n1(0);
        normals->points[normal_pt_id].normal_y = n1(1);
        normals->points[normal_pt_id].normal_z = n1(2);
    }
}

template V4R_EXPORTS void computeNormals<pcl::PointXYZRGB>(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &, pcl::PointCloud<pcl::Normal>::Ptr &, int, float);
template V4R_EXPORTS void computeNormals<pcl::PointXYZ>(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &, pcl::PointCloud<pcl::Normal>::Ptr &, int, float);
}
