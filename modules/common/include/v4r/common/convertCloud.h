/**
 * $Id$
 *
 * Copyright (c) 2014, Johann Prankl
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#ifndef KP_CONVERT_CLOUD_HPP
#define KP_CONVERT_CLOUD_HPP

#include <float.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <v4r/common/impl/DataMatrix2D.hpp>
#include <v4r/common/PointTypes.h>
#include <omp.h>
#include <glog/logging.h>

namespace v4r 
{


inline void convertCloud(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, DataMatrix2D<PointXYZRGB> &kp_cloud)
{
    kp_cloud.resize(cloud.height, cloud.width);

    for (size_t i=0; i<cloud.points.size(); i++)
    {
        const pcl::PointXYZRGB &pt = cloud.points[i];
        PointXYZRGB &kp = kp_cloud.data[i];

        kp.pt = pt.getVector4fMap();
        kp.rgb = pt.rgb;
    }
}

inline void convertCloud(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, DataMatrix2D<Eigen::Vector3f> &kp_cloud, cv::Mat_<cv::Vec3b> &image)
{
    kp_cloud.resize(cloud.height, cloud.width);
    image = cv::Mat_<cv::Vec3b>(cloud.height, cloud.width);

    for (size_t i=0; i<cloud.points.size(); i++)
    {
        const pcl::PointXYZRGB &pt = cloud.points[i];

        kp_cloud.data[i] = pt.getVector3fMap();
        image(i) = cv::Vec3b(pt.b,pt.g,pt.r);
    }
}

inline void convertCloud(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, DataMatrix2D<Eigen::Vector3f> &kp_cloud)
{
    kp_cloud.resize(cloud.height, cloud.width);

    for (size_t i=0; i<cloud.points.size(); i++)
    {
        const pcl::PointXYZRGB &pt = cloud.points[i];

        kp_cloud.data[i] = pt.getVector3fMap();
    }
}

inline void convertCloud(const pcl::PointCloud<pcl::PointXYZ> &cloud, DataMatrix2D<Eigen::Vector3f> &kp_cloud)
{
    kp_cloud.resize(cloud.height, cloud.width);

    for (size_t i=0; i<cloud.points.size(); i++)
    {
        const pcl::PointXYZ &pt = cloud.points[i];

        kp_cloud.data[i] = pt.getVector3fMap();
    }
}

inline void convertCloud(const DataMatrix2D<PointXYZRGB> &kp_cloud, pcl::PointCloud<pcl::PointXYZRGB> &pcl_cloud)
{
    pcl_cloud.points.resize(kp_cloud.data.size());
    pcl_cloud.width = kp_cloud.cols;
    pcl_cloud.height = kp_cloud.rows;
    pcl_cloud.is_dense = false;

    for (size_t i=0; i<pcl_cloud.points.size(); i++)
    {
        const PointXYZRGB &kp = kp_cloud.data[i];
        pcl::PointXYZRGB &pt = pcl_cloud.points[i];

        pt.getVector4fMap() = kp.pt;
        pt.rgb = kp.rgb;
    }
}

inline void convertCloud(const DataMatrix2D<PointXYZ> &kp_cloud, pcl::PointCloud<pcl::PointXYZ> &pcl_cloud)
{
    pcl_cloud.points.resize(kp_cloud.data.size());
    pcl_cloud.width = kp_cloud.cols;
    pcl_cloud.height = kp_cloud.rows;
    pcl_cloud.is_dense = false;

    for (size_t i=0; i<pcl_cloud.points.size(); i++)
    {
        const PointXYZ &kp = kp_cloud.data[i];
        pcl::PointXYZ &pt = pcl_cloud.points[i];

        pt.getVector3fMap() = kp.pt;
    }
}



// -------------- SAME WITH EIGEN MATRIX-------------------
inline void convertCloud(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, Eigen::Matrix4Xf &matrix)
{
    matrix = Eigen::Matrix4Xf (4, cloud.points.size() );

#pragma omp parallel for schedule (dynamic)
    for (size_t i=0; i<cloud.points.size(); i++)
    {
        const pcl::PointXYZRGB &pt = cloud.points[i];
        matrix.col(i) = pt.getVector4fMap();
    }
}

inline void convertCloud(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, Eigen::Matrix3Xf &matrix)
{
    matrix = Eigen::Matrix3Xf (3, cloud.points.size() );

#pragma omp parallel for schedule (dynamic)
    for (size_t i=0; i<cloud.points.size(); i++)
    {
        const pcl::PointXYZRGB &pt = cloud.points[i];
        matrix.col(i) = pt.getVector3fMap();
    }
}

inline void convertCloud(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const std::vector<int> &indices, Eigen::Matrix3Xf &matrix)
{
    CHECK(!indices.empty());

    matrix = Eigen::Matrix3Xf (3, indices.size() );

#pragma omp parallel for schedule (dynamic)
    for (size_t i=0; i<indices.size(); i++)
    {
        const pcl::PointXYZRGB &pt = cloud.points[ indices[i] ];
        matrix.col(i) = pt.getVector3fMap();
    }
}

inline void convertCloud(const pcl::PointCloud<pcl::PointXYZ> &cloud, Eigen::Matrix3Xf &matrix)
{
    matrix = Eigen::Matrix3Xf (3, cloud.points.size() );

#pragma omp parallel for schedule (dynamic)
    for (size_t i=0; i<cloud.points.size(); i++)
    {
        const pcl::PointXYZ &pt = cloud.points[i];
        matrix.col(i) = pt.getVector3fMap();
    }
}

inline void convertCloud(const pcl::PointCloud<pcl::PointXYZ> &cloud, const std::vector<int> &indices, Eigen::Matrix3Xf &matrix)
{
    CHECK(!indices.empty());

    matrix = Eigen::Matrix3Xf (3, indices.size() );

#pragma omp parallel for schedule (dynamic)
    for (size_t i=0; i<indices.size(); i++)
    {
        const pcl::PointXYZ &pt = cloud.points[ indices[i] ];
        matrix.col(i) = pt.getVector3fMap();
    }
}

inline void convertCloud(const pcl::PointCloud<pcl::Normal> &normal, Eigen::Matrix3Xf &matrix)
{
    matrix = Eigen::Matrix3Xf (3, normal.points.size() );

#pragma omp parallel for schedule (dynamic)
    for (size_t i=0; i<normal.points.size(); i++)
    {
        const pcl::Normal &pt = normal.points[i];
        matrix.col(i) = pt.getNormalVector3fMap();
    }
}


inline void convertCloud(const pcl::PointCloud<pcl::Normal> &normal, const std::vector<int> &indices, Eigen::Matrix3Xf &matrix)
{
    CHECK (!indices.empty());

    matrix = Eigen::Matrix3Xf (3, indices.size() );

#pragma omp parallel for schedule (dynamic)
    for (size_t i=0; i<indices.size(); i++)
    {
        const pcl::Normal &pt = normal.points[ indices[i] ];
        matrix.col(i) = pt.getNormalVector3fMap();
    }
}

inline void convertCloud(const Eigen::Matrix4Xf &matrix, pcl::PointCloud<pcl::PointXYZRGB> &cloud)
{
    cloud.points.resize( matrix.cols() );
    cloud.width = matrix.cols();
    cloud.height = 1;
    cloud.is_dense = false;

#pragma omp parallel for schedule (dynamic)
    for (size_t i=0; i<cloud.points.size(); i++)
    {
        pcl::PointXYZRGB &pt = cloud.points[i];
        pt.getVector4fMap() = matrix.col(i);
    }
}

inline void convertCloud(const Eigen::Matrix3Xf &matrix, pcl::PointCloud<pcl::PointXYZRGB> &cloud)
{
    cloud.points.resize( matrix.cols() );
    cloud.width = matrix.cols();
    cloud.height = 1;
    cloud.is_dense = false;

#pragma omp parallel for schedule (dynamic)
    for (size_t i=0; i<cloud.points.size(); i++)
    {
        pcl::PointXYZRGB &pt = cloud.points[i];
        pt.getVector3fMap() = matrix.col(i);
    }
}

inline void convertCloud(const Eigen::Matrix3Xf &matrix, pcl::PointCloud<pcl::PointXYZ> &cloud)
{
    cloud.points.resize( matrix.cols() );
    cloud.width = matrix.cols();
    cloud.height = 1;
    cloud.is_dense = false;

#pragma omp parallel for schedule (dynamic)
    for (size_t i=0; i<cloud.points.size(); i++)
    {
        pcl::PointXYZ &pt = cloud.points[i];
        pt.getVector3fMap() = matrix.col(i);
    }
}

inline void convertCloud(const Eigen::Matrix3Xf &matrix, pcl::PointCloud<pcl::Normal> &normals)
{
    normals.points.resize( matrix.cols() );
    normals.width = matrix.cols();
    normals.height = 1;
    normals.is_dense = false;

#pragma omp parallel for schedule (dynamic)
    for (size_t i=0; i<normals.points.size(); i++)
    {
        pcl::Normal &pt = normals.points[i];
        pt.getNormalVector3fMap() = matrix.col(i);
    }
}


} //--END--

#endif

