#include <v4r/common/miscellaneous.h>

namespace v4r
{
namespace common
{


template<typename PointType, typename DistType>
void
convertToFLANN ( const typename pcl::PointCloud<PointType>::ConstPtr & cloud, typename boost::shared_ptr< flann::Index<DistType> > &flann_index)
{
    size_t rows = cloud->points.size ();
    size_t cols = sizeof ( cloud->points[0].histogram ) / sizeof ( float ); // number of histogram bins

    flann::Matrix<float> flann_data ( new float[rows * cols], rows, cols );

    for ( size_t i = 0; i < rows; ++i )
    {
        for ( size_t j = 0; j < cols; ++j )
        {
            flann_data.ptr () [i * cols + j] = cloud->points[i].histogram[j];
        }
    }
    flann_index.reset (new flann::Index<DistType> ( flann_data, flann::KDTreeIndexParams ( 4 ) ) );
    flann_index->buildIndex ();
}

template <typename DistType>
void nearestKSearch ( typename boost::shared_ptr< flann::Index<DistType> > &index, float * descr, int descr_size, int k, flann::Matrix<int> &indices,
                        flann::Matrix<float> &distances )
{
    flann::Matrix<float> p = flann::Matrix<float> ( new float[descr_size], 1, descr_size );
    memcpy ( &p.ptr () [0], &descr[0], p.cols * p.rows * sizeof ( float ) );

    index->knnSearch ( p, indices, distances, k, flann::SearchParams ( 128 ) );
    delete[] p.ptr ();
}


template<typename PointType>
void setCloudPose(const Eigen::Matrix4f &trans, typename pcl::PointCloud<PointType> &cloud)
{
    cloud.sensor_origin_[0] = trans(0,3);
    cloud.sensor_origin_[1] = trans(1,3);
    cloud.sensor_origin_[2] = trans(2,3);
    Eigen::Matrix3f rotation = trans.block<3,3>(0,0);
    Eigen::Quaternionf q(rotation);
    cloud.sensor_orientation_ = q;
}

}
}
