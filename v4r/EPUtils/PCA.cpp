#include "PCA.hpp"
#include <Eigen/Dense>

namespace EPUtils
{
 
Eigen::Vector4f getMean(pcl::PointCloud<pcl::Normal>::ConstPtr cloud)
{
  Eigen::Vector4f mean;
  mean.setZero();

  for (unsigned int pi=0; pi<cloud->size(); pi++)
  {
    mean[0] += cloud->points.at(pi).normal[0];
    mean[1] += cloud->points.at(pi).normal[1];
    mean[2] += cloud->points.at(pi).normal[2];
  }

  mean /= (float)cloud->size();
  
  return(mean);
  
}

bool computeCovarianceMatrix(pcl::PointCloud<pcl::Normal>::ConstPtr cloud, const Eigen::Vector4f &mean, Eigen::Matrix3f &cov)
{
  bool done = false;
  cov.setZero ();

  for (unsigned pi = 0; pi < cloud->size (); ++pi)
  {
    float x = cloud->points.at(pi).normal[0] - mean[0];
    float y = cloud->points.at(pi).normal[1] - mean[1];
    float z = cloud->points.at(pi).normal[2] - mean[2];

    cov(0,0) += x*x;
    cov(0,1) += x*y;
    cov(0,2) += x*z;
    
    cov(1,0) += y*x;
    cov(1,1) += y*y;
    cov(1,2) += y*z;
    
    cov(2,0) += z*x;
    cov(2,1) += z*y;
    cov(2,2) += z*z;
    
    done = true;
  }
  
  return(done);
}

void principleAxis(pcl::PointCloud<pcl::Normal>::ConstPtr cloud, std::vector<pcl::Normal> &axis)
{
  Eigen::Vector4f mean;
  EIGEN_ALIGN16 Eigen::Matrix3f cov;
  EIGEN_ALIGN16 Eigen::Vector3f eigen_values;
  EIGEN_ALIGN16 Eigen::Matrix3f eigen_vectors;
  
  axis.clear();
  axis.resize(3);
  
  mean = getMean(cloud);
  
  if(computeCovarianceMatrix(cloud,mean,cov))
  {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(cov);
    if (eigensolver.info() != Eigen::Success)
      return;
     
    eigen_values = eigensolver.eigenvalues();
    eigen_vectors = eigensolver.eigenvectors();
    
    //pcl::eigen33 (cov, eigen_vectors, eigen_values);
  
    //std::cerr << "inside principleAxis" << std::endl;
    
    axis.at(0).normal[0] = eigen_vectors (0,0);
    axis.at(0).normal[1] = eigen_vectors (1,0);
    axis.at(0).normal[2] = eigen_vectors (2,0);
  
    axis.at(1).normal[0] = eigen_vectors (0,1);
    axis.at(1).normal[1] = eigen_vectors (1,1);
    axis.at(1).normal[2] = eigen_vectors (2,1);
  
    axis.at(2).normal[0] = eigen_vectors (0,2);
    axis.at(2).normal[1] = eigen_vectors (1,2);
    axis.at(2).normal[2] = eigen_vectors (2,2);
  }
  
}
  
} //namespace EPUtils