#ifndef FAAT_PCL_UTILS_MISCELLANEOUS_H_
#define FAAT_PCL_UTILS_MISCELLANEOUS_H_

#include <pcl/common/common.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/octree/octree_pointcloud_pointvector.h>
#include <pcl/octree/impl/octree_iterator.hpp>

namespace v4r
{
namespace common
{
namespace miscellaneous
{

void computeNormals(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud,
                    pcl::PointCloud<pcl::Normal>::Ptr &normals,
                    int method);

inline void transformNormals(pcl::PointCloud<pcl::Normal>::Ptr & normals_cloud,
                             pcl::PointCloud<pcl::Normal>::Ptr & normals_aligned,
                             Eigen::Matrix4f & transform)
{
    normals_aligned.reset (new pcl::PointCloud<pcl::Normal>);
    normals_aligned->points.resize (normals_cloud->points.size ());
    normals_aligned->width = normals_cloud->width;
    normals_aligned->height = normals_cloud->height;
    for (size_t k = 0; k < normals_cloud->points.size (); k++)
    {
        Eigen::Vector3f nt (normals_cloud->points[k].normal_x, normals_cloud->points[k].normal_y, normals_cloud->points[k].normal_z);
        normals_aligned->points[k].normal_x = static_cast<float> (transform (0, 0) * nt[0] + transform (0, 1) * nt[1]
                + transform (0, 2) * nt[2]);
        normals_aligned->points[k].normal_y = static_cast<float> (transform (1, 0) * nt[0] + transform (1, 1) * nt[1]
                + transform (1, 2) * nt[2]);
        normals_aligned->points[k].normal_z = static_cast<float> (transform (2, 0) * nt[0] + transform (2, 1) * nt[1]
                + transform (2, 2) * nt[2]);

        normals_aligned->points[k].curvature = normals_cloud->points[k].curvature;

    }
}

inline void transformNormals(const pcl::PointCloud<pcl::Normal>::ConstPtr & normals_cloud,
                             pcl::PointCloud<pcl::Normal>::Ptr & normals_aligned,
                             const Eigen::Matrix4f & transform)
{
    normals_aligned.reset (new pcl::PointCloud<pcl::Normal>);
    normals_aligned->points.resize (normals_cloud->points.size ());
    normals_aligned->width = normals_cloud->width;
    normals_aligned->height = normals_cloud->height;
    for (size_t k = 0; k < normals_cloud->points.size (); k++)
    {
        Eigen::Vector3f nt (normals_cloud->points[k].normal_x, normals_cloud->points[k].normal_y, normals_cloud->points[k].normal_z);
        normals_aligned->points[k].normal_x = static_cast<float> (transform (0, 0) * nt[0] + transform (0, 1) * nt[1]
                + transform (0, 2) * nt[2]);
        normals_aligned->points[k].normal_y = static_cast<float> (transform (1, 0) * nt[0] + transform (1, 1) * nt[1]
                + transform (1, 2) * nt[2]);
        normals_aligned->points[k].normal_z = static_cast<float> (transform (2, 0) * nt[0] + transform (2, 1) * nt[1]
                + transform (2, 2) * nt[2]);

        normals_aligned->points[k].curvature = normals_cloud->points[k].curvature;
    }
}

inline void transformNormals(const pcl::PointCloud<pcl::Normal>::ConstPtr & normals_cloud,
                             pcl::PointCloud<pcl::Normal>::Ptr & normals_aligned,
                             const std::vector<int> & indices,
                             const Eigen::Matrix4f & transform)
{
    normals_aligned.reset (new pcl::PointCloud<pcl::Normal>);
    normals_aligned->points.resize (indices.size ());
    normals_aligned->width = indices.size();
    normals_aligned->height = 1;
    for (size_t k = 0; k < indices.size(); k++)
    {
        Eigen::Vector3f nt (normals_cloud->points[indices[k]].normal_x,
                normals_cloud->points[indices[k]].normal_y,
                normals_cloud->points[indices[k]].normal_z);

        normals_aligned->points[k].normal_x = static_cast<float> (transform (0, 0) * nt[0] + transform (0, 1) * nt[1]
                + transform (0, 2) * nt[2]);
        normals_aligned->points[k].normal_y = static_cast<float> (transform (1, 0) * nt[0] + transform (1, 1) * nt[1]
                + transform (1, 2) * nt[2]);
        normals_aligned->points[k].normal_z = static_cast<float> (transform (2, 0) * nt[0] + transform (2, 1) * nt[1]
                + transform (2, 2) * nt[2]);

        normals_aligned->points[k].curvature = normals_cloud->points[indices[k]].curvature;

    }
}

inline void transformNormal(Eigen::Vector3f & nt,
                            Eigen::Vector3f & normal_out,
                            Eigen::Matrix4f & transform)
{
    normal_out[0] = static_cast<float> (transform (0, 0) * nt[0] + transform (0, 1) * nt[1] + transform (0, 2) * nt[2]);
    normal_out[1] = static_cast<float> (transform (1, 0) * nt[0] + transform (1, 1) * nt[1] + transform (1, 2) * nt[2]);
    normal_out[2] = static_cast<float> (transform (2, 0) * nt[0] + transform (2, 1) * nt[1] + transform (2, 2) * nt[2]);
}

inline void voxelGridWithOctree(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
                                pcl::PointCloud<pcl::PointXYZRGB> & voxel_grided,
                                float resolution)
{
    pcl::octree::OctreePointCloudPointVector<pcl::PointXYZRGB> octree(resolution);
    octree.setInputCloud(cloud);
    octree.addPointsFromInputCloud();

    pcl::octree::OctreePointCloudPointVector<pcl::PointXYZRGB>::LeafNodeIterator it2;
    const pcl::octree::OctreePointCloudPointVector<pcl::PointXYZRGB>::LeafNodeIterator it2_end = octree.leaf_end();

    int leaves = 0;
    for (it2 = octree.leaf_begin(); it2 != it2_end; ++it2, leaves++)
    {

    }

    voxel_grided.points.resize(leaves);
    voxel_grided.width = leaves;
    voxel_grided.height = 1;
    voxel_grided.is_dense = true;

    int kk=0;
    for (it2 = octree.leaf_begin(); it2 != it2_end; ++it2, kk++)
    {
        pcl::octree::OctreeContainerPointIndices& container = it2.getLeafContainer();
        std::vector<int> indexVector;
        container.getPointIndices (indexVector);

        int r,g,b;
        r = g = b = 0;
        pcl::PointXYZRGB p;
        p.getVector3fMap() = Eigen::Vector3f::Zero();

        for(size_t k=0; k < indexVector.size(); k++)
        {
            p.getVector3fMap() = p.getVector3fMap() +  cloud->points[indexVector[k]].getVector3fMap();
            r += cloud->points[indexVector[k]].r;
            g += cloud->points[indexVector[k]].g;
            b += cloud->points[indexVector[k]].b;
        }

        p.getVector3fMap() = p.getVector3fMap() / static_cast<int>(indexVector.size());
        p.r = r / static_cast<int>(indexVector.size());
        p.g = g / static_cast<int>(indexVector.size());
        p.b = b / static_cast<int>(indexVector.size());
        voxel_grided.points[kk] = p;
    }
}

template<typename PointInT>
inline void
getIndicesFromCloud(const typename pcl::PointCloud<PointInT>::ConstPtr & full_input_cloud,
                    const typename pcl::PointCloud<PointInT>::ConstPtr & search_points,
                    std::vector<int> & indices,
                    float resolution = 0.005f)
{
    pcl::octree::OctreePointCloudSearch<PointInT> octree (resolution);
    octree.setInputCloud (full_input_cloud);
    octree.addPointsFromInputCloud ();

    std::vector<int> pointIdxNKNSearch;
    std::vector<float> pointNKNSquaredDistance;

    indices.resize( search_points->points.size() );
    size_t kept=0;

    for(size_t j=0; j < search_points->points.size(); j++)
    {
        if (octree.nearestKSearch (search_points->points[j], 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
            indices[kept] = pointIdxNKNSearch[0];
            kept++;
        }
    }
    indices.resize(kept);
}

template<typename PointType, typename DistType> void convertToFLANN ( const typename pcl::PointCloud<PointType>::ConstPtr & cloud, typename boost::shared_ptr< flann::Index<DistType> > &flann_index);

template<typename DistType> void nearestKSearch ( typename boost::shared_ptr< flann::Index<DistType> > &index, float * descr, int descr_size, int k, flann::Matrix<int> &indices,
                                                  flann::Matrix<float> &distances );

}
}
}

#endif
