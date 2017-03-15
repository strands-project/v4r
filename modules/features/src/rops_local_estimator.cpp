#include <v4r/features/rops_local_estimator.h>

#if PCL_VERSION >= 100702


#include <pcl/point_types_conversion.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/features/rops_estimation.h>
#include <glog/logging.h>

namespace v4r
{

template<typename PointT>
void
ROPSLocalEstimation<PointT>::compute (std::vector<std::vector<float> > & signatures)
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

    // Estimate the normals.
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud_, *cloud_xyz);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
    normalEstimation.setInputCloud(cloud_xyz);
    normalEstimation.setRadiusSearch(0.03);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    normalEstimation.setSearchMethod(kdtree);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    normalEstimation.compute(*normals);

    // Perform triangulation.
    pcl::PointCloud<pcl::PointNormal>::Ptr cloudNormals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*cloud_xyz, *normals, *cloudNormals);
    pcl::search::KdTree<pcl::PointNormal>::Ptr kdtree2(new pcl::search::KdTree<pcl::PointNormal>);
    kdtree2->setInputCloud(cloudNormals);
    pcl::GreedyProjectionTriangulation<pcl::PointNormal> triangulation;
    pcl::PolygonMesh triangles;
    triangulation.setSearchRadius(0.025);
    triangulation.setMu(2.5);
    triangulation.setMaximumNearestNeighbors(100);
    triangulation.setMaximumSurfaceAngle(M_PI / 4); // 45 degrees.
    triangulation.setNormalConsistency(false);
    triangulation.setMinimumAngle(M_PI / 18); // 10 degrees.
    triangulation.setMaximumAngle(2 * M_PI / 3); // 120 degrees.
    triangulation.setInputCloud(cloudNormals);
    triangulation.setSearchMethod(kdtree2);
    triangulation.reconstruct(triangles);

    pcl::PointCloud<pcl::Histogram<135> >descriptors;
    pcl::ROPSEstimation<pcl::PointXYZ, pcl::Histogram<135> > rops;
    rops.setInputCloud(cloud_xyz);
    rops.setSearchMethod(kdtree);
    rops.setRadiusSearch(0.03);
    boost::shared_ptr<std::vector<int> > IndicesPtr (new std::vector<int>);
    *IndicesPtr = indices_;
    rops.setIndices(IndicesPtr);
    rops.setTriangles(triangles.polygons);
    // Number of partition bins that is used for distribution matrix calculation.
    rops.setNumberOfPartitionBins(param_.number_of_partition_bin_);
    // The greater the number of rotations is, the bigger the resulting descriptor.
    // Make sure to change the histogram size accordingly.
    rops.setNumberOfRotations(param_.number_of_rotations_);
    // Support radius that is used to crop the local surface of the point.
    rops.setSupportRadius(param_.support_radius_);

    rops.compute(descriptors);

    CHECK( descriptors.points.size() == indices_.size() );
    keypoint_indices_ = indices_;

    int size_feat = 352;
    signatures = std::vector<std::vector<float> > (descriptors.points.size (), std::vector<float>(size_feat) );

    for (size_t k = 0; k < descriptors.points.size (); k++)
        for (int i = 0; i < size_feat; i++)
            signatures[k][i] = descriptors.points[k].histogram[i];
}

template class V4R_EXPORTS ROPSLocalEstimation<pcl::PointXYZ>;
template class V4R_EXPORTS ROPSLocalEstimation<pcl::PointXYZRGB>;

}

#endif
