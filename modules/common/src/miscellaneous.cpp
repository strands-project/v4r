#include "v4r/common/miscellaneous.h"

//#include <v4r/keypoints/impl/convertCloud.hpp>
//#include <v4r/keypoints/impl/convertNormals.hpp>
//#include <v4r/keypoints/ZAdaptiveNormals.h>
#include <pcl/visualization/cloud_viewer.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/common/impl/miscellaneous.hpp>

namespace v4r
{
namespace common
{

void computeNormals(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud,
                    pcl::PointCloud<pcl::Normal>::Ptr &normals,
                    int method)
{
    normals.reset(new pcl::PointCloud<pcl::Normal>());
    if (method > 3)
        std::cerr << "Normal method specified (" << method << ") does not exist. Using normal method 3." << std::endl;

    if(method == 0)
    {
        pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> n3d;
        n3d.setRadiusSearch (0.01f);
        n3d.setInputCloud (cloud);
        n3d.compute (*normals);
    }
    else if(method == 1)
    {
        pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
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
        pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
        ne.setRadiusSearch ( 0.02f );
        ne.setInputCloud ( cloud );
        ne.compute ( *normals );
    }
    else //if(normal_method_ == 3)
    {
        //kp::ZAdaptiveNormals::Parameter n_param;
        //n_param.adaptive = true;
        //kp::ZAdaptiveNormals nest(n_param);

        //kp::DataMatrix2D<Eigen::Vector3f>::Ptr kp_cloud( new kp::DataMatrix2D<Eigen::Vector3f>() );
        //kp::DataMatrix2D<Eigen::Vector3f>::Ptr kp_normals_tmp( new kp::DataMatrix2D<Eigen::Vector3f>() );
        //kp::convertCloud(*cloud, *kp_cloud);
        //nest.compute(*kp_cloud, *kp_normals_tmp);
        //kp::convertNormals(*kp_normals_tmp, *normals);
        throw "not implemented";
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
//    {
//        pcl::visualization::PCLVisualizer vis("normal computation ouside DOL");
//        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_handler(cloud);
//        vis.addPointCloud(cloud, rgb_handler, "original_cloud");
//        vis.addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (cloud, normals, 10, 0.05, "normals");
//        vis.spin();
//    }
}

template V4R_EXPORTS void convertToFLANN<pcl::Histogram<128>, flann::L1<float> > (const pcl::PointCloud<pcl::Histogram<128> >::ConstPtr & cloud, boost::shared_ptr< flann::Index<flann::L1<float> > > &flann_index); // explicit instantiation.
template V4R_EXPORTS void nearestKSearch<flann::L1<float> > ( boost::shared_ptr< flann::Index< flann::L1<float> > > &index, float * descr, int descr_size, int k, flann::Matrix<int> &indices,
flann::Matrix<float> &distances );
template void V4R_EXPORTS nearestKSearch<flann::L2<float> > ( boost::shared_ptr< flann::Index< flann::L2<float> > > &index, float * descr, int descr_size, int k, flann::Matrix<int> &indices,
flann::Matrix<float> &distances );

//#define PCL_INSTANTIATE_setCloudPose(T) template void v4r::common::setCloudPose<T>(const Eigen::Matrix4f&, pcl::PointCloud<T>&);
//PCL_INSTANTIATE(setCloudPose, PCL_XYZ_POINT_TYPES)
template V4R_EXPORTS void setCloudPose<pcl::PointXYZ>(const Eigen::Matrix4f &tf, pcl::PointCloud<pcl::PointXYZ> &cloud);
template V4R_EXPORTS void setCloudPose<pcl::PointXYZRGB>(const Eigen::Matrix4f &tf, pcl::PointCloud<pcl::PointXYZRGB> &cloud);
template V4R_EXPORTS void setCloudPose<pcl::PointXYZRGBNormal>(const Eigen::Matrix4f &tf, pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud);
template V4R_EXPORTS void setCloudPose<pcl::PointXYZRGBA>(const Eigen::Matrix4f &tf, pcl::PointCloud<pcl::PointXYZRGBA> &cloud);

}
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
