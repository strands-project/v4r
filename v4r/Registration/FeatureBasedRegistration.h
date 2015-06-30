#ifndef V4R_REGISTRATION_FBR
#define V4R_REGISTRATION_FBR

//#define PCL_NO_PRECOMPILE
//#include <pcl/point_types.h>
//#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/search/kdtree.h>
#include "PartialModelRegistrationBase.h"

#include <flann/flann.h>

//PCL_EXPORTS std::ostream& operator << (std::ostream& os, const SIFTHistogram& p);

struct SIFTHistogram
{
    float histogram[128];

    friend std::ostream& operator << (std::ostream& os, const SIFTHistogram& p);
};

POINT_CLOUD_REGISTER_POINT_STRUCT (SIFTHistogram,
    (float[128], histogram, histogramSIFT)
)

namespace v4r
{
    namespace Registration
    {
        template<class PointT>
        class FeatureBasedRegistration : public PartialModelRegistrationBase<PointT>
        {
            private:

                using PartialModelRegistrationBase<PointT>::name_;
                using PartialModelRegistrationBase<PointT>::partial_1;
                using PartialModelRegistrationBase<PointT>::partial_2;
                using PartialModelRegistrationBase<PointT>::poses_;

                using PartialModelRegistrationBase<PointT>::getTotalNumberOfClouds;
                using PartialModelRegistrationBase<PointT>::getCloud;
                using PartialModelRegistrationBase<PointT>::getNormal;
                using PartialModelRegistrationBase<PointT>::getIndices;
                using PartialModelRegistrationBase<PointT>::getPose;
                using PartialModelRegistrationBase<PointT>::PointCloudTPtr;


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


namespace v4r
{
    namespace Registration
    {
        template<typename PointType, typename DistType> void convertToFLANN ( const typename pcl::PointCloud<PointType>::ConstPtr & cloud, typename boost::shared_ptr< flann::Index<DistType> > &flann_index);

        void nearestKSearch ( boost::shared_ptr< flann::Index<flann::L1<float> > > &index, float * descr, int descr_size, int k, flann::Matrix<int> &indices,
                                flann::Matrix<float> &distances );
    }
}
#endif
