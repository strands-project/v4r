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
//    if (param_.adaptative_MLS_)
//    {
//        throw std::runtime_error("Adaptive MLS is not implemented yet!");
//        std::cerr << "Using parameter adaptive MLS will break the keypoint indices!" << std::endl; //TODO: Fix this!
//        pcl::MovingLeastSquares<PointT, PointT> mls;
//        typename pcl::search::KdTree<PointT>::Ptr tree;
//        Eigen::Vector4f centroid_cluster;
//        pcl::compute3DCentroid (*in_, centroid_cluster);
//        float dist_to_sensor = centroid_cluster.norm ();
//        float sigma = dist_to_sensor * 0.01f;
//        mls.setSearchMethod (tree);
//        mls.setSearchRadius (sigma);
//        mls.setUpsamplingMethod (mls.SAMPLE_LOCAL_PLANE);
//        mls.setUpsamplingRadius (0.002);
//        mls.setUpsamplingStepSize (0.001);
//        mls.setInputCloud (in_);

//        pcl::PointCloud<PointT> filtered;
//        mls.process (filtered);
//        filtered.is_dense = false;
//        *processed_ = filtered;

//        computeNormals<PointT>(processed_, processed_normals, param_.normal_computation_method_);
//    }

    CHECK( cloud_->points.size() == normals_->points.size() );

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
    boost::shared_ptr<std::vector<int> > IndicesPtr (new std::vector<int> (indices_.size() ));
    for(size_t i=0; i<indices_.size(); i++)
        IndicesPtr->at(i) = originalIndices2new[ indices_[i] ];

    shot_estimate.setIndices(IndicesPtr);
    shot_estimate.setRadiusSearch (param_.support_radius_);
    shot_estimate.compute (shots);

    CHECK( shots.points.size() == indices_.size() );

    int size_feat = 352;
        signatures.resize (shots.points.size (), std::vector<float>(size_feat));

    for (size_t k = 0; k < shots.points.size (); k++)
        for (int i = 0; i < size_feat; i++)
            signatures[k][i] = shots.points[k].descriptor[i];
}

template class V4R_EXPORTS SHOTLocalEstimation<pcl::PointXYZ>;
template class V4R_EXPORTS SHOTLocalEstimation<pcl::PointXYZRGB>;
}

