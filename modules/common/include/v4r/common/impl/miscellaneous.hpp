#include <v4r/common/miscellaneous.h>
#include <omp.h>

namespace v4r
{

template<typename DistType>
void
convertToFLANN ( const std::vector<std::vector<float> > &data, boost::shared_ptr< typename flann::Index<DistType> > &flann_index)
{
    if(data.empty()) {
        std::cerr << "No data provided for building Flann index!" << std::endl;
        return;
    }

    size_t rows = data.size ();
    size_t cols = data[0].size(); // number of histogram bins

    flann::Matrix<float> flann_data ( new float[rows * cols], rows, cols );

    for ( size_t i = 0; i < rows; ++i )
    {
        for ( size_t j = 0; j < cols; ++j )
            flann_data.ptr() [i * cols + j] = data[i][j];
    }
    flann_index.reset (new flann::Index<DistType> ( flann_data, flann::KDTreeIndexParams ( 4 ) ) );
    flann_index->buildIndex ();
}

template <typename DistType>
void nearestKSearch ( typename boost::shared_ptr< flann::Index<DistType> > &index, std::vector<float> descr, int k, flann::Matrix<int> &indices,
                        flann::Matrix<float> &distances )
{
    flann::Matrix<float> p = flann::Matrix<float> ( new float[descr.size()], 1, descr.size() );
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



void
computeHistogram (const std::vector<float> & values, std::vector<size_t> &histogram, size_t hist_size, float min, float max)
{
    histogram.resize(hist_size, 0);

    omp_lock_t bin_lock[hist_size];
    for(size_t i=0; i<hist_size; i++)
        omp_init_lock(&bin_lock[i]);

#pragma omp parallel for firstprivate(min, max, hist_size) schedule(dynamic)
    for (size_t j = 0; j<values.size(); j++)
    {
        int pos = std::floor (static_cast<float>(hist_size) * (values[j] - min) / (max - min));

        if(pos < 0)
            pos = 0;

        if(pos > (int)hist_size)
            pos = hist_size - 1;

        omp_set_lock(&bin_lock[pos]);
        histogram[pos]++;
        omp_unset_lock(&bin_lock[pos]);
    }

    for(size_t i=0; i<hist_size; i++)
        omp_destroy_lock(&bin_lock[i]);
}


}


//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::copyPointCloud (const pcl::PointCloud<PointT> &cloud_in,
                     const std::vector<size_t> &indices,
                     pcl::PointCloud<PointT> &cloud_out)
{
  // Do we want to copy everything?
  if (indices.size () == cloud_in.points.size ())
  {
    cloud_out = cloud_in;
    return;
  }

  // Allocate enough space and copy the basics
  cloud_out.points.resize (indices.size ());
  cloud_out.header   = cloud_in.header;
  cloud_out.width    = static_cast<uint32_t>(indices.size ());
  cloud_out.height   = 1;
  cloud_out.is_dense = cloud_in.is_dense;
  cloud_out.sensor_orientation_ = cloud_in.sensor_orientation_;
  cloud_out.sensor_origin_ = cloud_in.sensor_origin_;

  // Iterate over each point
  for (size_t i = 0; i < indices.size (); ++i)
    cloud_out.points[i] = cloud_in.points[indices[i]];
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::copyPointCloud (const pcl::PointCloud<PointT> &cloud_in,
                     const std::vector<size_t, Eigen::aligned_allocator<size_t> > &indices,
                     pcl::PointCloud<PointT> &cloud_out)
{
  // Do we want to copy everything?
  if (indices.size () == cloud_in.points.size ())
  {
    cloud_out = cloud_in;
    return;
  }

  // Allocate enough space and copy the basics
  cloud_out.points.resize (indices.size ());
  cloud_out.header   = cloud_in.header;
  cloud_out.width    = static_cast<uint32_t> (indices.size ());
  cloud_out.height   = 1;
  cloud_out.is_dense = cloud_in.is_dense;
  cloud_out.sensor_orientation_ = cloud_in.sensor_orientation_;
  cloud_out.sensor_origin_ = cloud_in.sensor_origin_;

  // Iterate over each point
  for (size_t i = 0; i < indices.size (); ++i)
    cloud_out.points[i] = cloud_in.points[indices[i]];
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::copyPointCloud (const pcl::PointCloud<PointT> &cloud_in,
                     const std::vector<bool> &mask,
                     pcl::PointCloud<PointT> &cloud_out)
{
  assert(cloud_in.points.size() == mask.size());

  // Allocate enough space and copy the basics
  cloud_out.points.resize (cloud_in.points.size ());
  cloud_out.header   = cloud_in.header;
  cloud_out.width    = static_cast<uint32_t> (mask.size ());
  cloud_out.height   = 1;
  cloud_out.is_dense = cloud_in.is_dense;
  cloud_out.sensor_orientation_ = cloud_in.sensor_orientation_;
  cloud_out.sensor_origin_ = cloud_in.sensor_origin_;

  // Iterate over each point
  size_t kept=0;
  for (size_t i = 0; i < mask.size (); ++i)
  {
      if( mask[i] )
      {
            cloud_out.points[kept] = cloud_in.points[i];
            kept++;
      }
  }
  cloud_out.points.resize(kept);
  cloud_out.width = kept;
}
