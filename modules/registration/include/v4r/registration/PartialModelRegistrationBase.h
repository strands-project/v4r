#ifndef V4R_REGISTRATION_PMRB
#define V4R_REGISTRATION_PMRB

#include <pcl/common/common.h>
#include <v4r/core/macros.h>

namespace v4r
{
    namespace Registration
    {

        template<class PointT> class V4R_EXPORTS MultiSessionModelling;

        template<class PointT>
        class V4R_EXPORTS PartialModelRegistrationBase
        {
            protected:
                typedef typename pcl::PointCloud<PointT>::Ptr PointCloudTPtr;

                std::pair<int, int> partial_1, partial_2;
                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > poses_;
                std::string name_;
                MultiSessionModelling<PointT> * msm_;

                size_t getTotalNumberOfClouds();

            public:
                PartialModelRegistrationBase()
                {

                }

                void setSessions(std::pair<int, int> & p1, std::pair<int, int> & p2)
                {
                    partial_1 = p1;
                    partial_2 = p2;
                }

                void setMSM(MultiSessionModelling<PointT> * msm)
                {
                    msm_ = msm;
                }

                virtual void compute(int s1, int s2) = 0;

                virtual void initialize(std::vector<std::pair<int, int> > & session_ranges) = 0;

                void getPoses(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & poses)
                {
                    poses = poses_;
                }

                PointCloudTPtr getCloud(size_t i);
                std::vector<int> & getIndices(size_t i);
                Eigen::Matrix4f getPose(size_t i);
                pcl::PointCloud<pcl::Normal>::Ptr getNormal(size_t i);

        };
    }
}

#endif
