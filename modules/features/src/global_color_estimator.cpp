#include <v4r/features/global_color_estimator.h>
#include <v4r/common/histogram.h>
#include <pcl/common/io.h>
#include <fstream>

namespace v4r
{

template <typename PointT>
Eigen::MatrixXf
getTransformedRGB(const pcl::PointCloud<PointT> &cloud)
{
    Eigen::MatrixXf rgb( cloud.points.size(), 3);

    for(size_t i=0; i<cloud.points.size(); i++)
    {
        const PointT &p = cloud.points[i];
        rgb.row(i) =  Eigen::Vector3f(p.r, p.g, p.b);
    }


    Eigen::MatrixXf centered = rgb.rowwise() - rgb.colwise().mean();
    Eigen::MatrixXf cov = (centered.adjoint() * centered) / float(rgb.rows() - 1.f);

    Eigen::Vector3f diag = cov.diagonal();
    diag = diag.array().sqrt();


    Eigen::MatrixXf normalized (centered.rows(), centered.cols());

    for(int i=0; i<centered.rows(); i++)
    {
        for(int j=0;j<centered.cols(); j++)
            normalized(i,j) = centered(i,j) / diag(j);
    }

    return normalized;
}

template<>
bool
GlobalColorEstimator<pcl::PointXYZ>::compute (Eigen::MatrixXf &signature)
{
    LOG(FATAL) << "Given point type does not contain color information. Therefore cannot describe color features. Will return nothing.";
    signature = Eigen::MatrixXf();
    return false;
}

template<typename PointT>
bool
GlobalColorEstimator<PointT>::compute (Eigen::MatrixXf &signature)
{
    CHECK(cloud_ && !cloud_->points.empty());

//    if(!color_transf_)
//        color_transf_.reset(new RGB2CIELAB);

    Eigen::MatrixXf rgb;

    if(!indices_.empty())
    {
        typename pcl::PointCloud<PointT>::Ptr cloud_roi (new pcl::PointCloud<PointT>);
        pcl::copyPointCloud(*cloud_, indices_, *cloud_roi);
        rgb = getTransformedRGB(*cloud_roi);
//        color_transf_->convert( *cloud_roi, LAB_color);
    }
    else
    {
        rgb = getTransformedRGB(*cloud_);
//        color_transf_->convert( *cloud_, LAB_color);
    }


    // normalize to (0,1)
//    LAB_color.col(0).array() -= 50.f;
//    LAB_color.col(0) /= 100.f;
//    LAB_color.col(1) /= 100.f;  // just an approximation (LAB is not evenly distributed)
//    LAB_color.col(2) /= 100.f;  // just an approximation (LAB is not evenly distributed)

//    std::cout << "Max: " << LAB_color.colwise().maxCoeff() << std::endl;
//    std::cout << "Min: " << LAB_color.colwise().minCoeff() << std::endl;

    Eigen::MatrixXi histogram;
    computeHistogram(rgb, histogram, param_.num_bins, -param_.std_dev_multiplier_, param_.std_dev_multiplier_);


    Eigen::MatrixXf histogram_f = histogram.cast<float>();
    histogram_f.rowwise().normalize();
    histogram_f.transposeInPlace();
    Eigen::Map<Eigen::VectorXf> histogram_flattened(histogram_f.data(), histogram_f.size() );


//    signature = histogram.cast <float> ();

//    Eigen::MatrixXi signature_int (1, histogram.rows() * histogram.cols());
//    for(int v=0; v<histogram.rows(); v++)
//    {
//        for(int u=0; u<histogram.cols(); u++)
//        {
//            signature_int( 1, v*histogram.cols()+u ) = histogram(v,u);
//        }
//    }

    signature = histogram_flattened.transpose();
    indices_.clear();
    return true;
}

//template class V4R_EXPORTS GlobalColorEstimator<pcl::PointXYZ>;
template class V4R_EXPORTS GlobalColorEstimator<pcl::PointXYZRGB>;
}

