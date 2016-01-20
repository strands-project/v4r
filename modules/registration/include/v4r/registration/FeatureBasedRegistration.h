#ifndef V4R_REGISTRATION_FBR
#define V4R_REGISTRATION_FBR

#define PCL_NO_PRECOMPILE
//#include <pcl/point_types.h>
//#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/search/kdtree.h>
#include "PartialModelRegistrationBase.h"

#include <flann/flann.h>
#include <v4r/core/macros.h>

//PCL_EXPORTS std::ostream& operator << (std::ostream& os, const SIFTHistogram& p);

namespace v4r
{
    namespace Registration
    {
        template<class PointT>
        class V4R_EXPORTS FeatureBasedRegistration : public PartialModelRegistrationBase<PointT>
        {
            private:
                typedef pcl::Histogram<128> SIFTHistogram;

                using PartialModelRegistrationBase<PointT>::name_;
                using PartialModelRegistrationBase<PointT>::partial_1;
                using PartialModelRegistrationBase<PointT>::partial_2;
                using PartialModelRegistrationBase<PointT>::poses_;

//                using PartialModelRegistrationBase<PointT>::getTotalNumberOfClouds;
//                using PartialModelRegistrationBase<PointT>::getCloud;
//                using PartialModelRegistrationBase<PointT>::getNormal;
//                using PartialModelRegistrationBase<PointT>::getIndices;
//                using PartialModelRegistrationBase<PointT>::getPose;
//                using PartialModelRegistrationBase<PointT>::PointCloudTPtr;


                typedef flann::L1<float> DistT;

                class flann_model
                {
                public:
                  int keypoint_id; //points to sift_keypoints_
                  std::vector<float> descr;
                };

                std::vector< flann::Matrix<float> > flann_data_;
                std::vector< flann::Index< DistT > * > flann_index_; //for each session, a flann index
                std::vector< std::vector<flann_model> > flann_models_; //for each session, a flann model

                std::vector<typename pcl::PointCloud<PointT>::Ptr > sift_keypoints_;
                std::vector< pcl::PointCloud<pcl::Normal>::Ptr > sift_normals_;
                std::vector<pcl::PointCloud< SIFTHistogram >::Ptr > sift_features_;
                std::vector<pcl::PointCloud< SIFTHistogram >::Ptr > model_features_;

                bool do_cg_;
                float inlier_threshold_;
                int gc_threshold_;
                int kdtree_splits_;

            public:

                FeatureBasedRegistration();
                void compute(int s1, int s2);
                void initialize(std::vector<std::pair<int, int> > & session_ranges);
                void setDoCG(bool b)
                {
                    do_cg_ = b;
                }

                void setInlierThreshold(float f)
                {
                    inlier_threshold_ = f;
                }

                void setGCThreshold(int t)
                {
                    gc_threshold_ = t;
                }


        };
    }
}
#undef PCL_NO_PRECOMPILE
#endif
