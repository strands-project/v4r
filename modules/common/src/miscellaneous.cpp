#include "v4r/common/miscellaneous.h"

#include <v4r/common/convertCloud.h>
#include <v4r/common/convertNormals.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/common/normal_estimator.h>
#include <v4r/common/impl/miscellaneous.hpp>
#include <v4r/common/ZAdaptiveNormals.h>
#include <pcl/visualization/cloud_viewer.h>

namespace v4r
{
template<typename PointT>
void computeNormals(const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
                    pcl::PointCloud<pcl::Normal>::Ptr &normals,
                    int method)
{
    normals.reset(new pcl::PointCloud<pcl::Normal>());

    if(method == 0)
    {
        pcl::NormalEstimation<PointT, pcl::Normal> n3d;
        n3d.setRadiusSearch (0.01f);
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
        ne.setRadiusSearch ( 0.02f );
        ne.setInputCloud ( cloud );
        ne.compute ( *normals );
    }
    else if(method == 3)
    {
        v4r::ZAdaptiveNormals::Parameter n_param;
        n_param.adaptive = true;
        v4r::ZAdaptiveNormals nest(n_param);

        v4r::DataMatrix2D<Eigen::Vector3f>::Ptr kp_cloud( new v4r::DataMatrix2D<Eigen::Vector3f>() );
        v4r::DataMatrix2D<Eigen::Vector3f>::Ptr kp_normals_tmp( new v4r::DataMatrix2D<Eigen::Vector3f>() );
        v4r::convertCloud(*cloud, *kp_cloud);
        nest.compute(*kp_cloud, *kp_normals_tmp);
        v4r::convertNormals(*kp_normals_tmp, *normals);
    }
    else if(method==4)
    {
        boost::shared_ptr<v4r::PreProcessorAndNormalEstimator<PointT, pcl::Normal> > normal_estimator;
        normal_estimator.reset (new v4r::PreProcessorAndNormalEstimator<PointT, pcl::Normal>);
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

template V4R_EXPORTS void convertToFLANN<pcl::Histogram<128>, flann::L1<float> > (const pcl::PointCloud<pcl::Histogram<128> >::ConstPtr & cloud, boost::shared_ptr< flann::Index<flann::L1<float> > > &flann_index); // explicit instantiation.
template V4R_EXPORTS void convertToFLANN<pcl::Histogram<128>, flann::L2<float> > (const pcl::PointCloud<pcl::Histogram<128> >::ConstPtr & cloud, boost::shared_ptr< flann::Index<flann::L2<float> > > &flann_index); // explicit instantiation.
template V4R_EXPORTS void nearestKSearch<flann::L1<float> > ( boost::shared_ptr< flann::Index< flann::L1<float> > > &index, float * descr, int descr_size, int k, flann::Matrix<int> &indices,
flann::Matrix<float> &distances );
template void V4R_EXPORTS nearestKSearch<flann::L2<float> > ( boost::shared_ptr< flann::Index< flann::L2<float> > > &index, float * descr, int descr_size, int k, flann::Matrix<int> &indices,
flann::Matrix<float> &distances );

template V4R_EXPORTS void setCloudPose<pcl::PointXYZ>(const Eigen::Matrix4f &tf, pcl::PointCloud<pcl::PointXYZ> &cloud);
template V4R_EXPORTS void setCloudPose<pcl::PointXYZRGB>(const Eigen::Matrix4f &tf, pcl::PointCloud<pcl::PointXYZRGB> &cloud);
template V4R_EXPORTS void setCloudPose<pcl::PointXYZRGBNormal>(const Eigen::Matrix4f &tf, pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud);
template V4R_EXPORTS void setCloudPose<pcl::PointXYZRGBA>(const Eigen::Matrix4f &tf, pcl::PointCloud<pcl::PointXYZRGBA> &cloud);

template V4R_EXPORTS void
computeNormals<pcl::PointXYZRGB>(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud,
                    pcl::PointCloud<pcl::Normal>::Ptr &normals,
                    int method);

template V4R_EXPORTS void
computeNormals<pcl::PointXYZ>(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
                    pcl::PointCloud<pcl::Normal>::Ptr &normals,
                    int method);

template V4R_EXPORTS void
getIndicesFromCloud<pcl::PointXYZRGB, int>(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &full_input_cloud,
                                           const pcl::PointCloud<pcl::PointXYZRGB> &search_points,
                                           std::vector<int> &indices, float resolution);

template V4R_EXPORTS void
getIndicesFromCloud<pcl::PointXYZ, int>(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &full_input_cloud,
                                           const pcl::PointCloud<pcl::PointXYZ> &search_points,
                                           std::vector<int> &indices, float resolution);

template V4R_EXPORTS void
getIndicesFromCloud<pcl::PointXYZRGB, size_t>(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &full_input_cloud,
                                           const pcl::PointCloud<pcl::PointXYZRGB> &search_points,
                                           std::vector<size_t> &indices, float resolution);

template V4R_EXPORTS void
getIndicesFromCloud<pcl::PointXYZ, size_t>(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &full_input_cloud,
                                           const pcl::PointCloud<pcl::PointXYZ> &search_points,
                                           std::vector<size_t> &indices, float resolution);


template V4R_EXPORTS
std::vector<size_t>
createIndicesFromMask(const std::vector<bool> &mask, bool invert);

template V4R_EXPORTS
std::vector<int>
createIndicesFromMask(const std::vector<bool> &mask, bool invert);


}



template V4R_EXPORTS  void
pcl::copyPointCloud<pcl::PointXYZ> (const pcl::PointCloud<pcl::PointXYZ> &cloud_in,
                const std::vector<size_t> &indices,
                pcl::PointCloud<pcl::PointXYZ> &cloud_out);
template V4R_EXPORTS void
pcl::copyPointCloud<pcl::PointXYZRGB> (const pcl::PointCloud<pcl::PointXYZRGB> &cloud_in,
                const std::vector<size_t> &indices,
                pcl::PointCloud<pcl::PointXYZRGB> &cloud_out);
template V4R_EXPORTS void
pcl::copyPointCloud<pcl::PointXYZRGBNormal> (const pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud_in,
                const std::vector<size_t> &indices,
                pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud_out);
template V4R_EXPORTS void
pcl::copyPointCloud<pcl::PointXYZRGBA> (const pcl::PointCloud<pcl::PointXYZRGBA> &cloud_in,
                const std::vector<size_t> &indices,
                pcl::PointCloud<pcl::PointXYZRGBA> &cloud_out);
template V4R_EXPORTS void
pcl::copyPointCloud<pcl::Normal> (const pcl::PointCloud<pcl::Normal> &cloud_in,
                const std::vector<size_t> &indices,
                pcl::PointCloud<pcl::Normal> &cloud_out);


template V4R_EXPORTS void
pcl::copyPointCloud<pcl::PointXYZ> (const pcl::PointCloud<pcl::PointXYZ> &cloud_in,
                const std::vector<size_t, Eigen::aligned_allocator<size_t> > &indices,
                pcl::PointCloud<pcl::PointXYZ> &cloud_out);
template V4R_EXPORTS void
pcl::copyPointCloud<pcl::PointXYZRGB> (const pcl::PointCloud<pcl::PointXYZRGB> &cloud_in,
                const std::vector<size_t, Eigen::aligned_allocator<size_t> > &indices,
                pcl::PointCloud<pcl::PointXYZRGB> &cloud_out);
template V4R_EXPORTS void
pcl::copyPointCloud<pcl::PointXYZRGBNormal> (const pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud_in,
                const std::vector<size_t, Eigen::aligned_allocator<size_t> > &indices,
                pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud_out);
template V4R_EXPORTS void
pcl::copyPointCloud<pcl::PointXYZRGBA> (const pcl::PointCloud<pcl::PointXYZRGBA> &cloud_in,
                const std::vector<size_t, Eigen::aligned_allocator<size_t> > &indices,
                pcl::PointCloud<pcl::PointXYZRGBA> &cloud_out);
template V4R_EXPORTS void
pcl::copyPointCloud<pcl::Normal> (const pcl::PointCloud<pcl::Normal> &cloud_in,
                const std::vector<size_t, Eigen::aligned_allocator<size_t> > &indices,
                pcl::PointCloud<pcl::Normal> &cloud_out);


template V4R_EXPORTS void
pcl::copyPointCloud<pcl::PointXYZ> (const pcl::PointCloud<pcl::PointXYZ> &cloud_in,
                const std::vector<bool> &indices,
                pcl::PointCloud<pcl::PointXYZ> &cloud_out);
template V4R_EXPORTS void
pcl::copyPointCloud<pcl::PointXYZRGB> (const pcl::PointCloud<pcl::PointXYZRGB> &cloud_in,
                const std::vector<bool> &indices,
                pcl::PointCloud<pcl::PointXYZRGB> &cloud_out);
template V4R_EXPORTS void
pcl::copyPointCloud<pcl::PointXYZRGBNormal> (const pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud_in,
                const std::vector<bool> &indices,
                pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud_out);
template V4R_EXPORTS void
pcl::copyPointCloud<pcl::PointXYZRGBA> (const pcl::PointCloud<pcl::PointXYZRGBA> &cloud_in,
                const std::vector<bool> &indices,
                pcl::PointCloud<pcl::PointXYZRGBA> &cloud_out);
template V4R_EXPORTS void
pcl::copyPointCloud<pcl::Normal> (const pcl::PointCloud<pcl::Normal> &cloud_in,
                const std::vector<bool> &indices,
                pcl::PointCloud<pcl::Normal> &cloud_out);

