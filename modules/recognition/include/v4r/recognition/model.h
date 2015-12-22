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

#include <EDT/propagation_distance_field.h>
#include <v4r/core/macros.h>

#include <pcl/common/centroid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>


namespace v4r
{

/**
 * @brief Class representing a recognition model
 */
template<typename PointT>
class V4R_EXPORTS Model
{
  typedef typename pcl::PointCloud<PointT>::Ptr PointTPtr;
  typedef typename pcl::PointCloud<PointT>::ConstPtr PointTPtrConst;
  Eigen::Vector4f centroid_;
  bool centroid_computed_;

public:
  std::vector<PointTPtr> views_;
  std::vector<pcl::PointIndices> indices_;
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > poses_;
  std::vector<float>  self_occlusions_;
  std::string class_, id_;
  PointTPtr assembled_;
  pcl::PointCloud<pcl::Normal>::Ptr normals_assembled_;
  std::vector <std::string> view_filenames_;
  PointTPtr keypoints_; //model keypoints
  pcl::PointCloud<pcl::Normal>::Ptr kp_normals_; //keypoint normals
  mutable typename std::map<int, PointTPtrConst> voxelized_assembled_;
  mutable typename std::map<int, pcl::PointCloud<pcl::Normal>::ConstPtr> normals_voxelized_assembled_;
  //typename boost::shared_ptr<VoxelGridDistanceTransform<PointT> > dist_trans_;
  typename boost::shared_ptr<distance_field::PropagationDistanceField<PointT> > dist_trans_;

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

  void computeNormalsAssembledCloud(float radius_normals) {
    typename pcl::search::KdTree<PointT>::Ptr normals_tree (new pcl::search::KdTree<PointT>);
    typedef typename pcl::NormalEstimationOMP<PointT, pcl::Normal> NormalEstimator_;
    NormalEstimator_ n3d;
    normals_assembled_.reset (new pcl::PointCloud<pcl::Normal> ());
    normals_tree->setInputCloud (assembled_);
    n3d.setRadiusSearch (radius_normals);
    n3d.setSearchMethod (normals_tree);
    n3d.setInputCloud (assembled_);
    n3d.compute (*normals_assembled_);
  }

  pcl::PointCloud<pcl::PointXYZL>::Ptr
  getAssembledSmoothFaces (int resolution_mm)
  {
    if(resolution_mm <= 0)
      return faces_cloud_labels_;

    typename std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr>::iterator it = voxelized_assembled_labels_.find (resolution_mm);
    if (it == voxelized_assembled_labels_.end ())
    {
      double resolution = resolution_mm / (double)1000.f;
      pcl::PointCloud<pcl::PointXYZL>::Ptr voxelized (new pcl::PointCloud<pcl::PointXYZL>);
      pcl::VoxelGrid<pcl::PointXYZL> grid;
      grid.setInputCloud (faces_cloud_labels_);
      grid.setLeafSize (resolution, resolution, resolution);
      grid.setDownsampleAllData(true);
      grid.filter (*voxelized);

      voxelized_assembled_labels_[resolution] = voxelized;
      return voxelized;
    }

    return it->second;
  }

  PointTPtrConst
  getAssembled (int resolution_mm) const
  {
    if(resolution_mm <= 0)
      return assembled_;

    typename std::map<int, PointTPtrConst>::iterator it = voxelized_assembled_.find (resolution_mm);
    if (it == voxelized_assembled_.end ())
    {
      double resolution = (double)resolution_mm / 1000.f;
      PointTPtr voxelized (new pcl::PointCloud<PointT>);
      pcl::VoxelGrid<PointT> grid;
      grid.setInputCloud (assembled_);
      grid.setLeafSize (resolution, resolution, resolution);
      grid.setDownsampleAllData(true);
      grid.filter (*voxelized);

      PointTPtrConst voxelized_const (new pcl::PointCloud<PointT> (*voxelized));
      voxelized_assembled_[resolution] = voxelized_const;
      return voxelized_const;
    }

    return it->second;
  }

  pcl::PointCloud<pcl::Normal>::ConstPtr
  getNormalsAssembled (int resolution_mm) const
  {
    if(resolution_mm <= 0)
      return normals_assembled_;

    typename std::map<int, pcl::PointCloud<pcl::Normal>::ConstPtr >::iterator it = normals_voxelized_assembled_.find (resolution_mm);
    if (it == normals_voxelized_assembled_.end ())
    {
      double resolution = resolution_mm / 1000.f;
      pcl::PointCloud<pcl::PointNormal>::Ptr voxelized (new pcl::PointCloud<pcl::PointNormal>);
      pcl::PointCloud<pcl::PointNormal>::Ptr assembled_with_normals (new pcl::PointCloud<pcl::PointNormal>);
      assembled_with_normals->points.resize(assembled_->points.size());
      assembled_with_normals->width = assembled_->width;
      assembled_with_normals->height = assembled_->height;

      for(size_t i=0; i < assembled_->points.size(); i++) {
        assembled_with_normals->points[i].getVector4fMap() = assembled_->points[i].getVector4fMap();
        assembled_with_normals->points[i].getNormalVector4fMap() = normals_assembled_->points[i].getNormalVector4fMap();
      }

      pcl::VoxelGrid<pcl::PointNormal> grid;
      grid.setInputCloud (assembled_with_normals);
      grid.setLeafSize (resolution, resolution, resolution);
      grid.setDownsampleAllData(true);
      grid.filter (*voxelized);

      pcl::PointCloud<pcl::Normal>::Ptr voxelized_const (new pcl::PointCloud<pcl::Normal> ());
      voxelized_const->points.resize(voxelized->points.size());
      voxelized_const->width = voxelized->width;
      voxelized_const->height = voxelized->height;

      for(size_t i=0; i < voxelized_const->points.size(); i++)
        voxelized_const->points[i].getNormalVector4fMap() = voxelized->points[i].getNormalVector4fMap();


      normals_voxelized_assembled_[resolution] = voxelized_const;
      return voxelized_const;
    }

    return it->second;
  }

  void
  createVoxelGridAndDistanceTransform(int resolution_mm) {
    double resolution = (double)resolution_mm / 1000.f;
    PointTPtrConst assembled (new pcl::PointCloud<PointT> ());
    assembled = getAssembled(resolution_mm);
    dist_trans_.reset(new distance_field::PropagationDistanceField<PointT>(resolution));
    dist_trans_->setInputCloud(assembled);
    dist_trans_->compute();
  }

  void
  getVGDT(boost::shared_ptr<distance_field::PropagationDistanceField<PointT> > & dt) {
    dt = dist_trans_;
  }
};

}

#endif
