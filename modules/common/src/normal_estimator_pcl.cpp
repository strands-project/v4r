#include <v4r/common/normal_estimator_pcl.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/search/kdtree.h>

#include <pcl/impl/instantiate.hpp>
#include <glog/logging.h>

namespace v4r
{

template<typename PointT>
pcl::PointCloud<pcl::Normal>::Ptr
NormalEstimatorPCL<PointT>::compute()
{
    normal_.reset(new pcl::PointCloud<pcl::Normal>);
    normal_->points.resize(input_->height * input_->width);
    normal_->height = input_->height;
    normal_->width = input_->width;

    typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);

    boost::shared_ptr< std::vector<int> > IndicesPtr (new std::vector<int>);
    *IndicesPtr = indices_;

    if( param_.use_omp_ )
    {
        pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
        ne.setRadiusSearch ( param_.radius_ );
        ne.setInputCloud ( input_ );
        ne.setSearchMethod(tree);
        ne.setIndices(IndicesPtr);
        ne.compute ( *normal_ );
    }
    else
    {
        pcl::NormalEstimation<PointT, pcl::Normal> ne;
        ne.setRadiusSearch ( param_.radius_ );
        ne.setInputCloud (input_);
        ne.setSearchMethod(tree);
        ne.setIndices(IndicesPtr);
        ne.compute (*normal_);
    }

    return normal_;
}


#define PCL_INSTANTIATE_NormalEstimatorPCL(T) template class V4R_EXPORTS NormalEstimatorPCL<T>;
PCL_INSTANTIATE(NormalEstimatorPCL, PCL_XYZ_POINT_TYPES )

}
