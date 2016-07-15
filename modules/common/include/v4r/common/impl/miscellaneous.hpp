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

float
computeHistogramIntersection (const Eigen::VectorXf &histA, const Eigen::VectorXf &histB)
{
    Eigen::MatrixXf histAB (histA.rows(), 2);
    histAB.col(0) = histA;
    histAB.col(1) = histB;

    Eigen::VectorXf minv = histAB.rowwise().minCoeff();
    return minv.sum();
}


void
computeHistogram (const Eigen::MatrixXf &data, Eigen::MatrixXf &histogram, size_t bins, float min, float max)
{
    float bin_size = (max-min) / bins;
    int num_dim = data.cols();
    histogram = Eigen::MatrixXf::Zero (bins, num_dim);

    for (int dim = 0; dim < num_dim; dim++)
    {
        omp_lock_t bin_lock[bins];
        for(size_t pos=0; pos<bins; pos++)
            omp_init_lock(&bin_lock[pos]);

    #pragma omp parallel for firstprivate(min, max, bins) schedule(dynamic)
        for (int j = 0; j<data.rows(); j++)
        {
            int pos = std::floor( (data(j,dim) - min) / bin_size);

            if(pos < 0)
                pos = 0;

            if(pos > (int)bins)
                pos = bins - 1;

            omp_set_lock(&bin_lock[pos]);
            histogram(pos,dim)++;
            omp_unset_lock(&bin_lock[pos]);
        }

        for(size_t pos=0; pos<bins; pos++)
            omp_destroy_lock(&bin_lock[pos]);
    }
}

void
shiftHistogram (const Eigen::VectorXf &hist, Eigen::VectorXf &hist_shifted, bool direction)
{
    int bins = hist.rows();
    hist_shifted = Eigen::VectorXf::Zero(bins);

    if(direction){ //shift right
        hist_shifted.tail(bins - 1) = hist.head(bins-1);
        hist_shifted(bins-1) +=  hist(bins-1);
    }
    else { // shift left
        hist_shifted.head(bins - 1) = hist.tail(bins-1);
        hist_shifted(0) +=  hist(0);
    }
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
