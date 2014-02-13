#include <pcl/common/common.h>
#include <pcl/PolygonMesh.h>
#include <pcl/octree/octree.h>

namespace faat_pcl
{
namespace modelling
{

/*
 * From a sequence of partial scans that are already aligned, create a clean model of the data by removing outliers,
 * double walls as well as optionally applying moving least squares.
 */

template<typename PointT>
void getAveragedCloudFromOctree(typename pcl::PointCloud<PointT>::Ptr & octree_big_cloud,
                                typename pcl::PointCloud<PointT>::Ptr & filtered_big_cloud,
                                float octree_resolution,
                                bool median = false)
{
    typename pcl::octree::OctreePointCloudPointVector<PointT> octree(octree_resolution);
    octree.setInputCloud(octree_big_cloud);
    octree.addPointsFromInputCloud();

    unsigned int leaf_node_counter = 0;
    typename pcl::octree::OctreePointCloudPointVector<PointT>::LeafNodeIterator it2;
    const typename pcl::octree::OctreePointCloudPointVector<PointT>::LeafNodeIterator it2_end = octree.leaf_end();

    filtered_big_cloud->points.resize(octree_big_cloud->points.size());

    int kept = 0;
    for (it2 = octree.leaf_begin(); it2 != it2_end; ++it2)
    {
        ++leaf_node_counter;
        pcl::octree::OctreeContainerPointIndices& container = it2.getLeafContainer();
        // add points from leaf node to indexVector
        std::vector<int> indexVector;
        container.getPointIndices (indexVector);
        //std::cout << "Number of points in this leaf:" << indexVector.size() << std::endl;

        /*if(indexVector.size() <= 1)
      continue;*/

        PointT p;
        p.getVector3fMap() = Eigen::Vector3f::Zero();
        p.getNormalVector3fMap() = Eigen::Vector3f::Zero();
        std::vector<int> rs, gs, bs;
        int r,g,b;
        r = g = b = 0;
        int used = 0;
        for(size_t k=0; k < indexVector.size(); k++)
        {
            p.getVector3fMap() = p.getVector3fMap() +  octree_big_cloud->points[indexVector[k]].getVector3fMap();
            p.getNormalVector3fMap() = p.getNormalVector3fMap() +  octree_big_cloud->points[indexVector[k]].getNormalVector3fMap();
            r += octree_big_cloud->points[indexVector[k]].r;
            g += octree_big_cloud->points[indexVector[k]].g;
            b += octree_big_cloud->points[indexVector[k]].b;
            rs.push_back(octree_big_cloud->points[indexVector[k]].r);
            gs.push_back(octree_big_cloud->points[indexVector[k]].g);
            bs.push_back(octree_big_cloud->points[indexVector[k]].b);
        }

        p.getVector3fMap() = p.getVector3fMap() / static_cast<int>(indexVector.size());
        p.getNormalVector3fMap() = p.getNormalVector3fMap() / static_cast<int>(indexVector.size());
        p.r = r / static_cast<int>(indexVector.size());
        p.g = g / static_cast<int>(indexVector.size());
        p.b = b / static_cast<int>(indexVector.size());

        if(median)
        {
            std::sort(rs.begin(), rs.end());
            std::sort(bs.begin(), bs.end());
            std::sort(gs.begin(), gs.end());
            int size = rs.size() / 2;
            p.r = rs[size];
            p.g = gs[size];
            p.b = bs[size];
        }

        filtered_big_cloud->points[kept] = p;
        kept++;
    }

    filtered_big_cloud->points.resize(kept);
    filtered_big_cloud->width = kept;
    filtered_big_cloud->height = 1;
}

template<typename PointT>
class Sequence
{
    public:
        std::vector < Eigen::Matrix4f > transforms_to_global_;
        std::vector<typename pcl::PointCloud<PointT>::Ptr> original_clouds_;
        std::vector< std::vector<int> > original_indices_;
};

template<typename ScanPointT, typename ModelPointT>
class NiceModelFromSequence
{
    typedef typename pcl::PointCloud<ModelPointT>::Ptr ModelPointTPtr;
    typedef typename pcl::PointCloud<ScanPointT>::Ptr ScanPointTPtr;

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
        }
    }

    //attributes
    std::string input_dir_;
    bool organized_normals_;
    float lateral_sigma_;
    float octree_resolution_;
    float mls_radius_;
    bool mls_;
    bool visualize_;
    float w_t;
    bool median_;

    std::vector < Eigen::Matrix4f > transforms_to_global_;
    std::vector<typename pcl::PointCloud<ScanPointT>::ConstPtr> aligned_clouds_;
    std::vector<typename pcl::PointCloud<ScanPointT>::Ptr> original_clouds_;
    std::vector<std::vector<float> > weights_;
    ScanPointTPtr big_cloud_from_transforms_;
    pcl::PointCloud<pcl::Normal>::Ptr big_cloud_normals_from_transforms_;
    std::vector< std::vector<int> > original_indices_;
    Eigen::Vector4f table_plane_; //gets filled when reading the sequence...
    ModelPointTPtr model_cloud_;

public:

    NiceModelFromSequence()
    {
        mls_radius_ = 0.003f;
        octree_resolution_ = 0.0015f;
        lateral_sigma_ = 0.002f;
        organized_normals_ = true;
        mls_ = true;
        visualize_ = true;
        median_ = true;
        w_t = 0.75f;
    }

    void setDoMls(bool b)
    {
        mls_ = b;
    }

    void setWT(float f)
    {
        w_t = f;
    }

    void setInputClouds(std::vector<typename pcl::PointCloud<ScanPointT>::Ptr> & scans)
    {
        original_clouds_ = scans;
    }

    void setTransformations(std::vector < Eigen::Matrix4f > & transforms_to_global)
    {
        transforms_to_global_ = transforms_to_global;
    }

    void setInputDir(std::string dir)
    {
        input_dir_ = dir;
    }

    void setLateralSigma(float f)
    {
        lateral_sigma_ = f;
    }

    ModelPointTPtr getModelCloud()
    {
        return model_cloud_;
    }

    void setVisualize(bool b)
    {
        visualize_ = b;
    }

    void readSequence();
    void compute();

    void computeFromInputClouds();

    void getSequence(Sequence<ScanPointT> & seq)
    {
        seq.original_clouds_ = original_clouds_;
        seq.transforms_to_global_ = transforms_to_global_;
        seq.original_indices_ = original_indices_;
    }

};
}
}
