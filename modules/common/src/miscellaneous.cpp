#include <v4r/common/miscellaneous.h>
#include <v4r/common/impl/miscellaneous.hpp>

namespace v4r
{

template V4R_EXPORTS void convertToFLANN<flann::L1<float> > (const std::vector<std::vector<float> > &, boost::shared_ptr< flann::Index<flann::L1<float> > > &flann_index); // explicit instantiation.
template V4R_EXPORTS void convertToFLANN<flann::L2<float> > (const std::vector<std::vector<float> > &, boost::shared_ptr< flann::Index<flann::L2<float> > > &flann_index); // explicit instantiation.
template V4R_EXPORTS void nearestKSearch<flann::L1<float> > ( boost::shared_ptr< flann::Index< flann::L1<float> > > &index, std::vector<float> descr, int k, flann::Matrix<int> &indices,
flann::Matrix<float> &distances );
template void V4R_EXPORTS nearestKSearch<flann::L2<float> > ( boost::shared_ptr< flann::Index< flann::L2<float> > > &index, std::vector<float> descr, int k, flann::Matrix<int> &indices,
flann::Matrix<float> &distances );

template V4R_EXPORTS void setCloudPose<pcl::PointXYZ>(const Eigen::Matrix4f &tf, pcl::PointCloud<pcl::PointXYZ> &cloud);
template V4R_EXPORTS void setCloudPose<pcl::PointXYZRGB>(const Eigen::Matrix4f &tf, pcl::PointCloud<pcl::PointXYZRGB> &cloud);
template V4R_EXPORTS void setCloudPose<pcl::PointXYZRGBNormal>(const Eigen::Matrix4f &tf, pcl::PointCloud<pcl::PointXYZRGBNormal> &cloud);
template V4R_EXPORTS void setCloudPose<pcl::PointXYZRGBA>(const Eigen::Matrix4f &tf, pcl::PointCloud<pcl::PointXYZRGBA> &cloud);


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

