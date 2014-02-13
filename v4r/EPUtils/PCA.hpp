#ifndef EPUTILS_PCA_HPP
#define EPUTILS_PCA_HPP

#include "headers.hpp"

namespace EPUtils
{
  
Eigen::Vector4f getMean(pcl::PointCloud<pcl::Normal>::ConstPtr cloud);
bool computeCovarianceMatrix(pcl::PointCloud<pcl::Normal>::ConstPtr cloud, const Eigen::Vector4f &mean, Eigen::Matrix3f &cov);
void principleAxis(pcl::PointCloud<pcl::Normal>::ConstPtr cloud, std::vector<pcl::Normal> &axis);
  
} //namespace EPUtils

#endif //EPUTILS_PCA_HPP