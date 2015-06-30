#include "ORUtils.h"

#include <v4r/KeypointConversions/convertCloud.hpp>
#include <v4r/KeypointConversions/convertNormals.hpp>
#include <v4r/KeypointTools/ZAdaptiveNormals.hh>

#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/octree/octree_pointcloud_pointvector.h>
#include <pcl/octree/impl/octree_iterator.hpp>



namespace v4r
{

template <typename PointT>
void ORUtils<PointT>::computeNormals(const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
                    pcl::PointCloud<pcl::Normal> &normals,
                    int method)
{
    std::cout << "Computing normals..." << std::endl;
    if(method== 0)
    {
        pcl::NormalEstimationOMP<PointT, pcl::Normal> n3d;
        n3d.setRadiusSearch (0.02f);
        n3d.setInputCloud (cloud);
        n3d.compute (normals);
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
        ne.compute(normals);
    }
    else //if(normal_method_ == 2)
    {

        kp::ZAdaptiveNormals::Parameter n_param;
        n_param.adaptive = true;
        kp::ZAdaptiveNormals nest(n_param);

        kp::DataMatrix2D<Eigen::Vector3f>::Ptr kp_cloud( new kp::DataMatrix2D<Eigen::Vector3f>() );
        kp::DataMatrix2D<Eigen::Vector3f>::Ptr kp_normals_tmp( new kp::DataMatrix2D<Eigen::Vector3f>() );
        kp::convertCloud(*cloud, *kp_cloud);
        nest.compute(*kp_cloud, *kp_normals_tmp);
        kp::convertNormals(*kp_normals_tmp, normals);
    }

    // Normalize normals to unit length
    for ( size_t normal_pt_id = 0; normal_pt_id < normals.points.size(); normal_pt_id++)
    {
        Eigen::Vector3f n1 = normals.points[normal_pt_id].getNormalVector3fMap();
        n1.normalize();
        normals.points[normal_pt_id].normal_x = n1(0);
        normals.points[normal_pt_id].normal_y = n1(1);
        normals.points[normal_pt_id].normal_z = n1(2);
    }
}


void transformNormals(pcl::PointCloud<pcl::Normal>::Ptr & normals_cloud,
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
static void transformNormals(pcl::PointCloud<pcl::Normal>::ConstPtr & normals_cloud,
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

static void transformNormals(pcl::PointCloud<pcl::Normal>::Ptr & normals_cloud,
                             pcl::PointCloud<pcl::Normal>::Ptr & normals_aligned,
                             std::vector<int> & indices,
                             Eigen::Matrix4f & transform)
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

static void transformNormal(Eigen::Vector3f & nt,
                            Eigen::Vector3f & normal_out,
                            Eigen::Matrix4f & transform)
{
    normal_out[0] = static_cast<float> (transform (0, 0) * nt[0] + transform (0, 1) * nt[1] + transform (0, 2) * nt[2]);
    normal_out[1] = static_cast<float> (transform (1, 0) * nt[0] + transform (1, 1) * nt[1] + transform (1, 2) * nt[2]);
    normal_out[2] = static_cast<float> (transform (2, 0) * nt[0] + transform (2, 1) * nt[1] + transform (2, 2) * nt[2]);
}

static void voxelGridWithOctree(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
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

}

template void v4r::ORUtils<pcl::PointXYZRGB>::computeNormals(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud,
                                               pcl::PointCloud<pcl::Normal> &normals,
                                               int method);
