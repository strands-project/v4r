#ifndef V4R_REGISTRATION_SPR
#define V4R_REGISTRATION_SPR

#include <pcl/common/common.h>
#include "PartialModelRegistrationBase.h"
#include <pcl/PolygonMesh.h>
#include <pcl/search/octree.h>
#include <v4r/core/macros.h>

namespace v4r
{
    namespace Registration
    {
        class V4R_EXPORTS stablePlane
        {
        public:
            Eigen::Vector3f normal_;
            float area_;
            std::vector<int> polygon_indices_;
        };

        template<class PointT>
        class StablePlanesRegistration : public PartialModelRegistrationBase<PointT>
        {
            private:
                using PartialModelRegistrationBase<PointT>::name_;
                using PartialModelRegistrationBase<PointT>::partial_1;
                using PartialModelRegistrationBase<PointT>::partial_2;
                using PartialModelRegistrationBase<PointT>::poses_;

//                using PartialModelRegistrationBase<PointT>::getTotalNumberOfClouds;
//                using PartialModelRegistrationBase<PointT>::getCloud;
//                using PartialModelRegistrationBase<PointT>::getIndices;
//                using PartialModelRegistrationBase<PointT>::getPose;
//                using PartialModelRegistrationBase<PointT>::PointCloudTPtr;
//                using PartialModelRegistrationBase<PointT>::getNormal;

                void mergeTriangles(pcl::PolygonMesh::Ptr & mesh_out,
                                    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr & model_cloud,
                                    std::vector<stablePlane> & stable_planes);

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

                std::vector<std::vector<stablePlane> > stable_planes_;
                std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> partial_models_with_normals_;

                std::vector<Eigen::Vector3f> computed_planes_;

            public:

                StablePlanesRegistration();
                void compute(int s1, int s2);
                void initialize(std::vector<std::pair<int, int> > & session_ranges);

                //up-vector representing the plane on which the object was modeled...
                //if the object RF is already defined on the plane, then pass UnitZ
                void setSessionPlanes(std::vector<Eigen::Vector3f> & planes)
                {
                    computed_planes_ = planes;
                }
        };
    }
}

#endif
