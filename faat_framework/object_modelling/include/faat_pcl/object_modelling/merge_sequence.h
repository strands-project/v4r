#include <pcl/common/common.h>
#include <pcl/PolygonMesh.h>
#include <pcl/octree/octree.h>

namespace faat_pcl
{
namespace modelling
{
class stablePlane
{
public:
    Eigen::Vector3f normal_;
    float area_;
    std::vector<int> polygon_indices_;
};

template<typename ModelPointT>
class MergeSequences
{
    typedef typename pcl::PointCloud<ModelPointT>::Ptr PointTPtr;
    bool checkForGround(Eigen::Vector3f & p, Eigen::Vector3f & n)
    {
        if(p[2] < 0.002 && std::abs(n.dot(Eigen::Vector3f::UnitZ())) > 0.95 )
            return true;

        return false;
    }

    bool checkShareVertex(std::vector<uint32_t> & vert1, std::vector<uint32_t> & vert2)
    {
        //check if they share at least a point
        bool share = false;
        for(size_t k=0; k < vert1.size(); k++)
        {
            for(size_t j=0; j < vert2.size(); j++)
            {
                if(vert1[k] == vert2[j])
                {
                    share = true;
                    break;
                }
            }
        }

        return share;
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
        }
    }

    template<typename PointInT>
    inline void
    getIndicesFromCloud(typename pcl::PointCloud<PointInT>::Ptr & processed,
                        typename pcl::PointCloud<PointInT>::Ptr & keypoints_pointcloud,
                        std::vector<int> & indices)
    {
        pcl::octree::OctreePointCloudSearch<PointInT> octree (0.005);
        octree.setInputCloud (processed);
        octree.addPointsFromInputCloud ();

        std::vector<int> pointIdxNKNSearch;
        std::vector<float> pointNKNSquaredDistance;

        for(size_t j=0; j < keypoints_pointcloud->points.size(); j++)
        {
            if (octree.nearestKSearch (keypoints_pointcloud->points[j], 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
            {
                indices.push_back(pointIdxNKNSearch[0]);
            }
        }
    }

    void mergeTriangles(pcl::PolygonMesh::Ptr & mesh_out,
                        PointTPtr & model_cloud,
                        std::vector<stablePlane> & stable_planes);

    //attributes
    float inlier_threshold_;
    float overlap_;
    PointTPtr model_cloud_;
    PointTPtr target_model_;
    int MAX_PLANES_;
    float angular_step_;
    int max_iterations_;
    bool use_color_;
public:
    MergeSequences()
    {
        MAX_PLANES_ = 6;
        angular_step_ = 30.f;
        inlier_threshold_ = 0.01f;
        overlap_ = 0.65f;
        max_iterations_ = 50;
        use_color_ = false;
    }

    void setMaxIterations(int i)
    {
        max_iterations_ = i;
    }

    void setInlierThreshold(float t)
    {
        inlier_threshold_ = t;
    }

    void setOverlap(float o)
    {
        overlap_ = o;
    }

    void setInputSource(PointTPtr & c)
    {
        model_cloud_ = c;
    }

    void setInputTarget(PointTPtr & c)
    {
        target_model_ = c;
    }

    void setUseColor(bool b)
    {
        use_color_ = b;
    }

    void compute(std::vector<std::pair<float, Eigen::Matrix4f> > & res);
};
}
}
