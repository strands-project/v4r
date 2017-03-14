#include <pcl/features/shot.h>
#include <pcl/features/shot_omp.h>
#include <pcl/surface/mls.h>
#include <v4r/features/shot_local_estimator.h>
#include <glog/logging.h>

namespace v4r
{

template<typename PointT>
void
SHOTLocalEstimation<PointT>::compute (std::vector<std::vector<float> > & signatures)
{
    CHECK( cloud_->points.size() == normals_->points.size() );
    signatures.clear();

    typename pcl::PointCloud<PointT>::Ptr cloud_wo_nan (new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr normals_wo_nan (new pcl::PointCloud<pcl::Normal>);
    cloud_wo_nan->points.resize( cloud_->points.size() );
    normals_wo_nan->points.resize( cloud_->points.size() );
    std::vector<int> originalIndices2new ( cloud_->points.size(), -1 );

    size_t kept = 0;
    for(size_t i=0; i<cloud_->points.size(); i++)
    {
        const PointT &p = cloud_->points[i];
        const pcl::Normal &n = normals_->points[i];
        if( pcl::isFinite(p) && pcl::isFinite(n) )
        {
            cloud_wo_nan->points[kept] = p;
            normals_wo_nan->points[kept] = n;
            originalIndices2new[i] = kept;
            kept++;
        }
    }
    cloud_wo_nan->points.resize(kept);
    cloud_wo_nan->width = kept;
    cloud_wo_nan->height = 1;
    normals_wo_nan->points.resize(kept);
    normals_wo_nan->width = kept;
    normals_wo_nan->height = 1;
    originalIndices2new.resize(kept);


    typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud_wo_nan);
    pcl::PointCloud<pcl::SHOT352> shots;
    pcl::SHOTEstimationOMP<PointT, pcl::Normal, pcl::SHOT352> shot_estimate;
    shot_estimate.setNumberOfThreads (0);
    shot_estimate.setSearchMethod (tree);
    shot_estimate.setInputCloud (cloud_wo_nan);
    shot_estimate.setInputNormals (normals_wo_nan);
    boost::shared_ptr<std::vector<int> > indices_p (new std::vector<int> (indices_.size() ));
    for(size_t i=0; i<indices_.size(); i++)
        indices_p->at(i) = originalIndices2new[ indices_[i] ];
    shot_estimate.setIndices(indices_p);
    shot_estimate.setRadiusSearch ( param_.support_radius_ );
    shot_estimate.compute (shots);

    CHECK( shots.points.size() == indices_.size() );

    keypoint_indices_ = indices_;

    signatures = std::vector<std::vector<float> > (shots.points.size(), std::vector<float>(352));

    for (size_t k = 0; k < shots.points.size (); k++)
        for (size_t i = 0; i < 352; i++)
            signatures[k][i] = shots.points[k].descriptor[i];
}

template class V4R_EXPORTS SHOTLocalEstimation<pcl::PointXYZ>;
template class V4R_EXPORTS SHOTLocalEstimation<pcl::PointXYZRGB>;
}

