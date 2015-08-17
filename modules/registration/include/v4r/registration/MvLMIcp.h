#ifndef V4R_MVLMICP_H
#define V4R_MVLMICP_H

#include <pcl/common/common.h>
#include <pcl/octree/octree_search.h>

#include <v4r/core/macros.h>

#include <EDT/propagation_distance_field.h>

namespace v4r
{
    namespace Registration
    {
        template<class PointT>
        class V4R_EXPORTS MvLMIcp
        {
            private:
                //private util functions...
                void computeAdjacencyMatrix();
                void fillViewParList();


                int max_iterations_;
                int diff_type;

                //todo: maybe solve this by using a friend class instead of making everything public
            public:
                typedef typename pcl::PointCloud<PointT>::Ptr PointCloudTPtr;

                std::vector<PointCloudTPtr> clouds_;
                std::vector<PointCloudTPtr> clouds_transformed_with_ip_;

                std::vector<pcl::PointCloud<pcl::Normal>::Ptr > normals_;
                std::vector<pcl::PointCloud<pcl::Normal>::Ptr > normals_transformed_with_ip_;

                //initial poses bringing clouds_ into alignment
                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > poses_;

                //final poses after refinement
                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > final_poses_;

                //adjacency matrix representing which views are to be registered
                std::vector< std::vector<bool> > adjacency_matrix_;

                //list of view pairs that will influence the MV process,
                //computed from adjacency_matrix_
                std::vector< std::pair<int, int> > S_;

                float normal_dot_;
                //octrees for the clouds...
                std::vector<boost::shared_ptr<typename pcl::octree::OctreePointCloudSearch<PointT> > > octrees_;

                //distance transforms...
                std::vector<boost::shared_ptr<typename distance_field::PropagationDistanceField<PointT> > > distance_transforms_;

                std::vector<std::vector<float> > weights_;

                double max_correspondence_distance_;

                void setMaxCorrespondenceDistance(double d)
                {
                    max_correspondence_distance_ = d;
                }

                void setMaxIterations(int i)
                {
                    max_iterations_ = i;
                }

                void setDiffType(int i)
                {
                    diff_type = i;
                }

                void setWeights(std::vector<std::vector<float> > & w)
                {
                    weights_ = w;
                }

                void setNormalDot(float d)
                {
                    normal_dot_ = d;
                }


                MvLMIcp();
                void setInputClouds(std::vector<PointCloudTPtr> & clouds)
                {
                    clouds_ = clouds;
                }

                void setPoses(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & poses)
                {
                    poses_ = poses;
                }

                void setNormals(std::vector<pcl::PointCloud<pcl::Normal>::Ptr > & normals)
                {
                    normals_ = normals;
                }

                void compute();

                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > getFinalPoses()
                {
                    return final_poses_;
                }
        };
    }
}
#endif
