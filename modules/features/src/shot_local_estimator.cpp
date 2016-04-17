#include <pcl/features/shot.h>
#include <pcl/features/shot_omp.h>
#include <pcl/surface/mls.h>
#include <v4r/features/shot_local_estimator.h>
#include <glog/logging.h>

namespace v4r
{

template<typename PointT>
bool
SHOTLocalEstimation<PointT>::compute (const pcl::PointCloud<PointT> & in, pcl::PointCloud<PointT> & processed, pcl::PointCloud<PointT> & keypoints, std::vector<std::vector<float> > & signatures)
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

    typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    pcl::PointCloud<pcl::SHOT352> shots;
    pcl::SHOTEstimationOMP<PointT, pcl::Normal, pcl::SHOT352> shot_estimate;
    shot_estimate.setNumberOfThreads (0);
    shot_estimate.setSearchMethod (tree);
    shot_estimate.setInputCloud (in.makeShared());
    throw std::runtime_error ("normals not set");
    shot_estimate.setInputNormals (normals_);
    boost::shared_ptr<std::vector<int> > IndicesPtr (new std::vector<int>);
    *IndicesPtr = indices_;
    shot_estimate.setIndices(IndicesPtr);
    shot_estimate.setRadiusSearch (support_radius_);
    shot_estimate.compute (shots);

    CHECK( shots.points.size() == indices_.size() );

    if (shots.points.empty() == 0)
    {
      PCL_WARN("SHOTLocalEstimationOMP :: No keypoints were found\n");
      return false;
    }

  int size_feat = 352;
  signatures.resize (shots.points.size (), std::vector<float>(size_feat));


  for (size_t k = 0; k < shots.points.size (); k++)
    for (int i = 0; i < size_feat; i++)
      signatures[k][i] = shots.points[k].descriptor[i];

  return true;
}

template class V4R_EXPORTS SHOTLocalEstimation<pcl::PointXYZ>;
template class V4R_EXPORTS SHOTLocalEstimation<pcl::PointXYZRGB>;
}

