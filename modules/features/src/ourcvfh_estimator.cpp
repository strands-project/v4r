#include <v4r/features/ourcvfh_estimator.h>

#include <v4r/features/pcl_ourcvfh.h>
#include <pcl/search/kdtree.h>
#include <glog/logging.h>

namespace v4r
{

template <typename PointT>
bool
OURCVFHEstimator<PointT>::compute (Eigen::MatrixXf &signature)
{
    CHECK(cloud_ && !cloud_->points.empty() && normals_);

    transforms_.clear();

    typename pcl::search::KdTree<PointT>::Ptr kdtree (new pcl::search::KdTree<PointT>);

    OURCVFHEstimation<PointT, pcl::Normal, pcl::VFHSignature308> ourcvfh;
    ourcvfh.setMinPoints ( param_.min_points_ );
    ourcvfh.setAxisRatio (param_.axis_ratio_);
    ourcvfh.setMinAxisValue (param_.min_axis_value_);
    ourcvfh.setNormalizeBins ( param_.normalize_bins_ );
    ourcvfh.setRefineClusters ( param_.refine_factor_ );
    ourcvfh.setSearchMethod (kdtree);

    if(!indices_.empty())
    {
        typename pcl::PointCloud<PointT>::Ptr cloud_roi (new pcl::PointCloud<PointT>);
        typename pcl::PointCloud<pcl::Normal>::Ptr normals_roi (new pcl::PointCloud<pcl::Normal>);
        pcl::copyPointCloud(*cloud_, indices_, *cloud_roi);
        pcl::copyPointCloud(*normals_, indices_, *normals_roi);
        ourcvfh.setInputCloud(cloud_roi);
        ourcvfh.setInputNormals(normals_roi);
    }
    else
    {
        ourcvfh.setInputCloud (cloud_);
        ourcvfh.setInputNormals(normals_);
    }

    for (float eps_angle_threshold : param_.eps_angle_threshold_vector_)
    {
        for (float curvature_threshold : param_.curvature_threshold_vector_ )
        {
            for (float cluster_tolerance : param_.cluster_tolerance_vector_)
            {
                pcl::PointCloud<pcl::VFHSignature308> cvfh_signatures;
                ourcvfh.setEPSAngleThreshold (eps_angle_threshold);
                ourcvfh.setCurvatureThreshold (curvature_threshold);
                ourcvfh.setClusterTolerance ( cluster_tolerance );
                ourcvfh.compute (cvfh_signatures);

                std::vector<bool> valid_rolls_tmp;
                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_tmp;

                ourcvfh.getTransforms (transforms_tmp);
                ourcvfh.getValidTransformsVec (valid_rolls_tmp);

                size_t kept = 0;
                for (size_t i = 0; i < valid_rolls_tmp.size (); i++)
                {
                    if (valid_rolls_tmp[i])
                    {
                        transforms_tmp[kept] = transforms_tmp[i];
                        cvfh_signatures.points[kept] = cvfh_signatures.points[i];
                        kept++;
                    }
                }

                transforms_tmp.resize (kept);
                cvfh_signatures.points.resize (kept);

                transforms_.insert( transforms_.end(), transforms_tmp.begin(), transforms_tmp.end() );

                signature = Eigen::MatrixXf ( cvfh_signatures.points.size (), 308 );

                for (size_t i = 0; i < cvfh_signatures.points.size (); i++)
                {
                    for (size_t d = 0; d < 308; d++)
                        signature(i,d) = cvfh_signatures.points[i].histogram[d];
                }
            }
        }
    }

    indices_.clear();
    return true;
}

template class V4R_EXPORTS OURCVFHEstimator<pcl::PointXYZ>;
template class V4R_EXPORTS OURCVFHEstimator<pcl::PointXYZRGB>;
}
