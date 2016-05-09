/******************************************************************************
 * Copyright (c) 2013 Aitor Aldoma, Thomas Faeulhammer
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/


#ifndef RECOGNITION_MODEL_H
#define RECOGNITION_MODEL_H

#include <v4r/core/macros.h>
#include <pcl/common/centroid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace v4r
{

/**
 * @brief Class representing a recognition model
 */
template<typename PointT>
class V4R_EXPORTS Model
{
private:
  mutable pcl::visualization::PCLVisualizer::Ptr vis_;
  mutable int vp1_;

public:
  typedef typename pcl::PointCloud<PointT>::Ptr PointTPtr;
  typedef typename pcl::PointCloud<PointT>::ConstPtr PointTPtrConst;
  std::vector<PointTPtr> views_;
  std::vector<std::vector<int> > indices_;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > poses_;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > eigen_pose_alignment_;
  Eigen::MatrixX3f elongations_; /// @brief elongations in meter for each dimension (column) and each view (row)
  std::vector<float> self_occlusions_;
  std::string class_, id_;
  PointTPtr assembled_;
  pcl::PointCloud<pcl::Normal>::Ptr normals_assembled_;
  std::vector <std::string> view_filenames_;
  PointTPtr keypoints_; //model keypoints
  pcl::PointCloud<pcl::Normal>::Ptr kp_normals_; //keypoint normals
  std::map<std::string, Eigen::MatrixXf> signatures_; /// @brief signatures of all local keypoint descriptors. Each element in the map represents a set of keypoint description (e.g. SIFT). The columns of the matrix represent the signature of one keypoint.
  mutable typename std::map<int, PointTPtrConst> voxelized_assembled_;
  mutable typename std::map<int, pcl::PointCloud<pcl::Normal>::ConstPtr> normals_voxelized_assembled_;
  Eigen::Vector4f centroid_;    /// @brief centre of gravity for the whole 3d model
  Eigen::MatrixX3f view_centroid_;  /// @brief centre of gravity for each 2.5D view of the model (each row corresponds to one view)
  bool centroid_computed_;

  pcl::PointCloud<pcl::PointXYZL>::Ptr faces_cloud_labels_;
  typename std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr> voxelized_assembled_labels_;
  bool flip_normals_based_on_vp_;

  Model()
  {
    centroid_computed_ = false;
    flip_normals_based_on_vp_ = false;
  }

  bool getFlipNormalsBasedOnVP() const
  {
      return flip_normals_based_on_vp_;
  }

  void setFlipNormalsBasedOnVP(bool b)
  {
      flip_normals_based_on_vp_ = b;
  }

  Eigen::Vector4f getCentroid()
  {
    if(centroid_computed_)
      return centroid_;

    //compute
    pcl::compute3DCentroid(*assembled_, centroid_);
    centroid_[3] = 0.f;
    centroid_computed_ = true;
    return centroid_;
  }

  bool
  operator== (const Model &other) const
  {
    return (id_ == other.id_) && (class_ == other.class_);
  }

  void computeNormalsAssembledCloud(float radius_normals);

  pcl::PointCloud<pcl::PointXYZL>::Ptr getAssembledSmoothFaces (int resolution_mm);

  typename pcl::PointCloud<PointT>::ConstPtr getAssembled(int resolution_mm) const;

  pcl::PointCloud<pcl::Normal>::ConstPtr getNormalsAssembled (int resolution_mm) const;


  typedef boost::shared_ptr< Model<PointT> > Ptr;
  typedef boost::shared_ptr< Model<PointT> const> ConstPtr;

};

}

#endif
