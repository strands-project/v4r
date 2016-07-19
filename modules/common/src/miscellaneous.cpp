#include <v4r/common/miscellaneous.h>
#include <v4r/common/impl/miscellaneous.hpp>

namespace v4r
{

V4R_EXPORTS
void transformNormals(const pcl::PointCloud<pcl::Normal> & normals_cloud,
                             pcl::PointCloud<pcl::Normal> & normals_aligned,
                             const Eigen::Matrix4f & transform)
{
    normals_aligned.points.resize (normals_cloud.points.size ());
    normals_aligned.width = normals_cloud.width;
    normals_aligned.height = normals_cloud.height;

    #pragma omp parallel for schedule(dynamic)
    for (size_t k = 0; k < normals_cloud.points.size (); k++)
    {
        Eigen::Vector3f nt (normals_cloud.points[k].normal_x, normals_cloud.points[k].normal_y, normals_cloud.points[k].normal_z);
        normals_aligned.points[k].normal_x = static_cast<float> (transform (0, 0) * nt[0] + transform (0, 1) * nt[1]
                + transform (0, 2) * nt[2]);
        normals_aligned.points[k].normal_y = static_cast<float> (transform (1, 0) * nt[0] + transform (1, 1) * nt[1]
                + transform (1, 2) * nt[2]);
        normals_aligned.points[k].normal_z = static_cast<float> (transform (2, 0) * nt[0] + transform (2, 1) * nt[1]
                + transform (2, 2) * nt[2]);

        normals_aligned.points[k].curvature = normals_cloud.points[k].curvature;
    }
}


V4R_EXPORTS void transformNormals(const pcl::PointCloud<pcl::Normal> & normals_cloud,
                             pcl::PointCloud<pcl::Normal> & normals_aligned,
                             const std::vector<int> & indices,
                             const Eigen::Matrix4f & transform)
{
    normals_aligned.points.resize (indices.size ());
    normals_aligned.width = indices.size();
    normals_aligned.height = 1;
    for (size_t k = 0; k < indices.size(); k++)
    {
        Eigen::Vector3f nt (normals_cloud.points[indices[k]].normal_x,
                normals_cloud.points[indices[k]].normal_y,
                normals_cloud.points[indices[k]].normal_z);

        normals_aligned.points[k].normal_x = static_cast<float> (transform (0, 0) * nt[0] + transform (0, 1) * nt[1]
                + transform (0, 2) * nt[2]);
        normals_aligned.points[k].normal_y = static_cast<float> (transform (1, 0) * nt[0] + transform (1, 1) * nt[1]
                + transform (1, 2) * nt[2]);
        normals_aligned.points[k].normal_z = static_cast<float> (transform (2, 0) * nt[0] + transform (2, 1) * nt[1]
                + transform (2, 2) * nt[2]);

        normals_aligned.points[k].curvature = normals_cloud.points[indices[k]].curvature;

    }
}


V4R_EXPORTS
void voxelGridWithOctree(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
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
V4R_EXPORTS void
getIndicesFromCloud(const typename pcl::PointCloud<PointInT>::ConstPtr & full_input_cloud,
                    const typename pcl::PointCloud<PointInT>::ConstPtr & search_points,
                    std::vector<int> & indices,
                    float resolution)
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

template<typename PointT, typename Type>
V4R_EXPORTS void
getIndicesFromCloud(const typename pcl::PointCloud<PointT>::ConstPtr & full_input_cloud,
                    const typename pcl::PointCloud<PointT> & search_pts,
                    typename std::vector<Type> & indices,
                    float resolution)
{
    pcl::octree::OctreePointCloudSearch<PointT> octree (resolution);
    octree.setInputCloud (full_input_cloud);
    octree.addPointsFromInputCloud ();

    std::vector<int> pointIdxNKNSearch;
    std::vector<float> pointNKNSquaredDistance;

    indices.resize( search_pts.points.size() );
    size_t kept=0;

    for(size_t j=0; j < search_pts.points.size(); j++)
    {
        if (octree.nearestKSearch (search_pts.points[j], 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
            indices[kept] = pointIdxNKNSearch[0];
            kept++;
        }
    }
    indices.resize(kept);
}

V4R_EXPORTS bool
incrementVector(const std::vector<bool> &v, std::vector<bool> &inc_v)
{
    inc_v = v;

    bool overflow=true;
    for(size_t bit=0; bit<v.size(); bit++)
    {
        if(!v[bit])
        {
            overflow = false;
            break;
        }
    }

    bool carry = v.back();
    inc_v.back() = !v.back();
    for(int bit=v.size()-2; bit>=0; bit--)
    {
        inc_v[bit] = v[ bit ] != carry;
        carry = v[ bit ] && carry;
    }
    return overflow;
}

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

