#include <v4r/common/normal_estimator_z_adpative.h>
#include <pcl/impl/instantiate.hpp>
#include <pcl/common/eigen.h>
#include <pcl/features/normal_3d.h>
#include <omp.h>

namespace v4r
{

template<typename PointT>
void
ZAdaptiveNormalsPCL<PointT>::computeCovarianceMatrix (const std::vector<int> &indices, const Eigen::Vector3f &mean, Eigen::Matrix3f &cov)
{
    Eigen::Vector3f pt;
    cov.setZero ();

    for (unsigned i = 0; i < indices.size (); ++i)
    {
        pt = input_->points[indices[i]].getVector3fMap() - mean;

        cov(1,1) += pt[1] * pt[1];
        cov(1,2) += pt[1] * pt[2];
        cov(2,2) += pt[2] * pt[2];

        pt *= pt[0];
        cov(0,0) += pt[0];
        cov(0,1) += pt[1];
        cov(0,2) += pt[2];
    }

    cov(1,0) = cov(0,1);
    cov(2,0) = cov(0,2);
    cov(2,1) = cov(1,2);
}

template<typename PointT>
void
ZAdaptiveNormalsPCL<PointT>::getIndices( size_t u, size_t v, int kernel, std::vector<int> &indices)
{
    const PointT &pt = input_->at(u,v);

    for (int vkernel=-kernel; vkernel<=kernel; vkernel++) {
        for (int ukernel=-kernel; ukernel<=kernel; ukernel++) {
            int y = (int)v + vkernel;
            int x = (int)u + ukernel;

            float center_dist = sqrt( vkernel*vkernel + ukernel*ukernel );
            if (x>0 && y>0 && x<(int)input_->width && y<(int)input_->height)
            {
                const PointT &pt1 = input_->at(x,y);
                if( pcl::isFinite(pt1) )
                {
                    float new_sqr_radius = sqr_radius;
                    if(param_.adaptive_)
                    {
                        float val = param_.kappa_ * center_dist * pt1.z + param_.d_;
                        new_sqr_radius = val*val;
                    }

                    if ( ( pt.getVector3fMap() - pt1.getVector3fMap() ).squaredNorm() < new_sqr_radius)
                        indices.push_back( getIdx(x,y) );
                }
            }
        }
    }
}

template<typename PointT>
float
ZAdaptiveNormalsPCL<PointT>::computeNormal( std::vector<int> &indices, Eigen::Matrix3d &eigen_vectors)
{
    if (indices.size()<4)
        return std::numeric_limits<float>::quiet_NaN();

    Eigen::Vector4d mean;
//    mean.setZero();
//    for (size_t j=0; j<indices.size(); j++)
//        mean += input_->points [ indices[j] ].getVector3fMap();
//    mean /= (float)indices.size();

    Eigen::Matrix3d cov;
//    computeCovarianceMatrix (indices, mean, cov);


    pcl::computeMeanAndCovarianceMatrix(*input_, indices, cov, mean);

    Eigen::Vector3d eigen_values;
    pcl::eigen33 (cov, eigen_vectors, eigen_values);

    double eigsum = eigen_values.sum();
    if (eigsum != 0)
        return fabs (eigen_values[0] / eigsum );

    return std::numeric_limits<float>::quiet_NaN();
}


template<typename PointT>
pcl::PointCloud<pcl::Normal>::Ptr
ZAdaptiveNormalsPCL<PointT>::compute()
{
    normal_.reset(new pcl::PointCloud<pcl::Normal>);
    normal_->points.resize(input_->height * input_->width);
    normal_->height = input_->height;
    normal_->width = input_->width;

    EIGEN_ALIGN16 Eigen::Matrix3d eigen_vectors;
    std::vector< int > indices;

#pragma omp parallel for private(eigen_vectors,indices)
    for (size_t v=0; v<input_->height; v++)
    {
        for (size_t u=0; u<input_->width; u++)
        {
            indices.clear();
            const PointT &pt = input_->at(u,v);
            pcl::Normal &n = normal_->at(u,v);
            if( pcl::isFinite(pt) )
            {
                if(param_.adaptive_)
                {
                    int radius_id = std::min<int>( (int)(param_.kernel_radius_.size())-1,  (int)( pt.z * 2 ) ); // *2 => every 0.5 meter another kernel radius
                    getIndices(u,v, param_.kernel_radius_[radius_id], indices);
                }
                else
                    getIndices(u,v, param_.kernel_, indices);
            }

            if (indices.size()<4)
            {
                n.normal_x = n.normal_y = n.normal_z = n.curvature = std::numeric_limits<float>::quiet_NaN();
                continue;
            }

            n.curvature = computeNormal( indices, eigen_vectors);
            n.normal_x = eigen_vectors (0,0);
            n.normal_y = eigen_vectors (1,0);
            n.normal_z = eigen_vectors (2,0);


            if ( n.getNormalVector3fMap().dot(pt.getVector3fMap() ) > 0)
            {
                n.getNormalVector3fMap() *= -1;
                //n.getNormalVector4fMap()[3] = 0;
                //n.getNormalVector4fMap()[3] = -1 * n.getNormalVector4fMap().dot(pt.getVector4fMap());
            }
        }
    }

    return normal_;
}

#define PCL_INSTANTIATE_ZAdaptiveNormalsPCL(T) template class V4R_EXPORTS ZAdaptiveNormalsPCL<T>;
PCL_INSTANTIATE(ZAdaptiveNormalsPCL, PCL_XYZ_POINT_TYPES )

}

