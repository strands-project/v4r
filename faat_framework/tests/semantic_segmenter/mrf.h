

#ifndef MRF_H_
#define MRF_H_

#include <stdlib.h>
#include <stdio.h>

//#include <algorithm>
#include <queue>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

namespace hombreViejo
{


  template <typename PointInT>
  //pcl::PointCloud<pcl::PointXYZRGB>::Ptr solveMrfViaBP(const typename pcl::PointCloud<PointInT>::ConstPtr& keypoints,
  pcl::PointCloud<pcl::PointXYZL>::Ptr solveMrfViaBP(const typename pcl::PointCloud<PointInT>::ConstPtr& keypoints,
                                                        std::vector<std::vector<float> > &probs,
                                                        float lambda, int iterations, bool verbose = false);

  template <typename PointInT>
  pcl::PointCloud<pcl::PointXYZL>::Ptr solveMrfViaBP_kNN(const typename pcl::PointCloud<PointInT>::ConstPtr& keypoints,
                                                          std::vector<std::vector<float> > &probs,
                                                          std::vector<std::vector<int> > &ccomps,
                                                          float lambda, int iterations, unsigned k_kNNGraph,
                                                          bool verbose = false);
  template <typename PointInT>
  pcl::PointCloud<pcl::PointXYZL>::Ptr solveMrfViaBP_eBall(const typename pcl::PointCloud<PointInT>::ConstPtr& keypoints,
                                                            std::vector<std::vector<float> > &probs,
                                                            std::vector<std::vector<int> > &ccomps,
                                                            float lambda, int iterations, double r_eBallGraph,
                                                            bool verbose = false);

}

#include "impl/mrf.hpp"

#endif
