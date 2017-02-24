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

    typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> );
    tree->setInputCloud(input_);

    boost::shared_ptr< std::vector<int> > IndicesPtr (new std::vector<int>);
    *IndicesPtr = indices_;

    if( param_.use_omp_ )
    {
        pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
        ne.setRadiusSearch ( param_.radius_ );
        ne.setInputCloud ( input_ );
        ne.setSearchMethod(tree);
        if(!indices_.empty())
            ne.setIndices(IndicesPtr);
        ne.compute ( *normal_ );
    }
    else
    {
        pcl::NormalEstimation<PointT, pcl::Normal> ne;
        ne.setRadiusSearch ( param_.radius_ );
        ne.setInputCloud (input_);
        ne.setSearchMethod(tree);
        if(!indices_.empty())
            ne.setIndices(IndicesPtr);
        ne.compute (*normal_);
    }

    indices_.clear();
    return normal_;
}


#define PCL_INSTANTIATE_NormalEstimatorPCL(T) template class V4R_EXPORTS NormalEstimatorPCL<T>;
PCL_INSTANTIATE(NormalEstimatorPCL, PCL_XYZ_POINT_TYPES )

}
