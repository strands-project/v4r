#include <v4r/features/global_color_estimator.h>
#include <v4r/common/histogram.h>
#include <pcl/common/io.h>

namespace v4r
{
template<typename PointT>
bool
GlobalColorEstimator<PointT>::compute (Eigen::MatrixXf &signature)
{
    CHECK(cloud_ && !cloud_->points.empty());

    if(!color_transf_)
        color_transf_.reset(new RGB2CIELAB);

    Eigen::MatrixXf LAB_color;

    if(!indices_.empty())   /// NOTE: setIndices does not seem to work for ESF
    {
        typename pcl::PointCloud<PointT>::Ptr cloud_roi (new pcl::PointCloud<PointT>);
        pcl::copyPointCloud(*cloud_, indices_, *cloud_roi);
        color_transf_->convert( *cloud_roi, LAB_color);
    }
    else
        color_transf_->convert( *cloud_, LAB_color);

    LAB_color.col(0).array() += 50.f;
    LAB_color.col(0) /= 100.f;
    LAB_color.col(1) /= 128.f;
    LAB_color.col(2) /= 128.f;

    Eigen::MatrixXi histogram;
    size_t bins = 5;
    computeHistogram(LAB_color, histogram, bins, 0.f, 1.f);

//    CHECK( histogram.IsRowMajor );

//    histogram.transpose();
//    Eigen::Map<Eigen::VectorXi> histogram_flattened(histogram.data(), histogram.size() );


//    signature = histogram.cast <float> ();

    signature = Eigen::MatrixXf(1, 3);
    Eigen::MatrixXf mean_colors = LAB_color.colwise().mean();
//    Eigen::MatrixXf e = (LAB_color - mean_colors.replicate(LAB_color.rows(), 1));
//    Eigen::MatrixXf std_dev_colors =  ( e.cwiseProduct(e) )/ (LAB_color.rows() - 1);
    signature << mean_colors;//, std_dev_colors;
    indices_.clear();
    return true;
}

template class V4R_EXPORTS GlobalColorEstimator<pcl::PointXYZRGB>;
}

