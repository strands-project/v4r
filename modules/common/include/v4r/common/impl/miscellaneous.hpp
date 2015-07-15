#include <v4r/common/miscellaneous.h>

template<typename PointType, typename DistType>
void
v4r::common::miscellaneous::convertToFLANN ( const typename pcl::PointCloud<PointType>::ConstPtr & cloud, typename boost::shared_ptr< flann::Index<DistType> > &flann_index)
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
void v4r::common::miscellaneous::nearestKSearch ( typename boost::shared_ptr< flann::Index<DistType> > &index, float * descr, int descr_size, int k, flann::Matrix<int> &indices,
                        flann::Matrix<float> &distances )
{
    flann::Matrix<float> p = flann::Matrix<float> ( new float[descr_size], 1, descr_size );
    memcpy ( &p.ptr () [0], &descr[0], p.cols * p.rows * sizeof ( float ) );

    index->knnSearch ( p, indices, distances, k, flann::SearchParams ( 128 ) );
    delete[] p.ptr ();
}
