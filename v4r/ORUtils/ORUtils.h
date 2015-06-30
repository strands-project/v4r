#ifndef ORUTILS_H__
#define ORUTILS_H__

#include <pcl/common/common.h>

namespace v4r
{
template <class PointT>
class ORUtils
{
public:
    template <typename
    static void computeNormals(const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
                        pcl::PointCloud<pcl::Normal> &normals,
                        int method);
    static void voxelGridWithOctree(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
                                    pcl::PointCloud<pcl::PointXYZRGB> & voxel_grided,
                                    float resolution);

    static void transformNormal(Eigen::Vector3f & nt,
                                Eigen::Vector3f & normal_out,
                                Eigen::Matrix4f & transform);

    static void transformNormals(pcl::PointCloud<pcl::Normal>::Ptr & normals_cloud,
                                 pcl::PointCloud<pcl::Normal>::Ptr & normals_aligned,
                                 std::vector<int> & indices,
                                 Eigen::Matrix4f & transform);

    static void transformNormals(pcl::PointCloud<pcl::Normal>::ConstPtr & normals_cloud,
                          pcl::PointCloud<pcl::Normal>::Ptr & normals_aligned,
                          Eigen::Matrix4f & transform);
};
}

#endif //OURUTILS_H__


