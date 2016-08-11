
#pragma once

#include <pcl/PointIndices.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PCLHeader.h>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <std_msgs/Header.h>

namespace boost
{

namespace serialization
{

template<class Archive>
void serialize(Archive & ar,std_msgs::Header & g, const unsigned int version)
{
    ar & g.seq;
    //ar & g.stamp;
    ar & g.frame_id;
}

template<class Archive>
void serialize(Archive & ar, pcl::PCLHeader & g, const unsigned int version)
{
    ar & g.seq;
    ar & g.stamp;
    ar & g.frame_id;
}

template<class Archive>
void serialize(Archive & ar, pcl::PointXYZ & g, const unsigned int version)
{
    ar & g.x;
    ar & g.y;
    ar & g.z;
}

template<class Archive>
void serialize(Archive & ar, pcl::PointXYZRGB & g, const unsigned int version)
{
    ar & g.x;
    ar & g.y;
    ar & g.z;
    ar & g.rgba;
}

template<class Archive>
void serialize(Archive & ar, pcl::PointNormal & g, const unsigned int version)
{
    ar & g.x;
    ar & g.y;
    ar & g.z;
    ar & g.normal[0];
    ar & g.normal[1];
    ar & g.normal[2];
}

template<class Archive>
void serialize(Archive & ar, pcl::Normal & g, const unsigned int version)
{
    ar & g.normal[0];
    ar & g.normal[1];
    ar & g.normal[2];
}


template<class Archive>
void serialize(Archive & ar,Eigen::Quaternion<float> & g, const unsigned int version)
{
    ar & g.w();
    ar & g.x();
    ar & g.y();
    ar & g.z();
}
template<class Archive>
void serialize(Archive & ar, Eigen::Translation<float, 3> & g, const unsigned int version)
{
    ar & g.x();
    ar & g.y();
    ar & g.z();
}

template<class Archive>
void serialize(Archive & ar, Eigen::UniformScaling<float> & g, const unsigned int version)
{
    ar & g.factor();
}

template<class Archive>
void serialize(Archive & ar, pcl::PointIndices & g, const unsigned int version)
{
    ar & g.indices;
    ar & g.header;
}

template<class Archive>
void serialize(Archive & ar, pcl::PointCloud<pcl::PointNormal> & g, const unsigned int version)
{
    ar & g.header;
    ar & g.points;
    ar & g.height;
    ar & g.width;
    ar & g.is_dense;
}

template<class Archive>
void serialize(Archive & ar, pcl::PointCloud<pcl::PointXYZ> & g, const unsigned int version)
{
    ar & g.header;
    ar & g.points;
    ar & g.height;
    ar & g.width;
    ar & g.is_dense;
}

template<class Archive>
void serialize(Archive & ar, pcl::PointCloud<pcl::PointXYZRGB> & g, const unsigned int version)
{
    ar & g.header;
    ar & g.points;
    ar & g.height;
    ar & g.width;
    ar & g.is_dense;
}

template<class Archive>
void serialize(Archive & ar, pcl::PointCloud<pcl::Normal> & g, const unsigned int version)
{
    ar & g.header;
    ar & g.points;
    ar & g.height;
    ar & g.width;
    ar & g.is_dense;
}


template<class Archive>
void serialize(Archive & ar, Eigen::Transform<float, 3, Eigen::Affine, Eigen::DontAlign> & g, const unsigned int version)
{

}

template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
inline void serialize( Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & t, const unsigned int file_version )
{
    for(size_t i=0; i<t.size(); i++)
        ar & t.data()[i];
}


}
}
