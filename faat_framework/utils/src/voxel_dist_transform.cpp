/*
 * voxel_dist_transform.h
 *
 *  Created on: Oct 20, 2012
 *      Author: aitor
 */

#include <faat_pcl/utils/voxel_dist_transform.h>
#include <pcl/common/common.h>

template<typename PointT>
void
faat_pcl::rec_3d_framework::VoxelGridDistanceTransform<PointT>::compute ()
{

  std::cout << "Going to compute voxel transform" << std::endl;

  if (cloud_ == 0)
  {
    PCL_ERROR ("setInputCloud() not called yet...\n");
    return;
  }

  //compute extension of the voxelgrid at the specified resolution_ for cloud_
  pcl::getMinMax3D (*cloud_, min_pt_all, max_pt_all);
  float extend = 0.05f;
  min_pt_all.x -= extend;
  min_pt_all.y -= extend;
  min_pt_all.z -= extend;
  max_pt_all.x += extend;
  max_pt_all.y += extend;
  max_pt_all.z += extend;

  gs_x_ = static_cast<int> (std::ceil (std::abs (max_pt_all.x - min_pt_all.x) / resolution_)) + 1;
  gs_y_ = static_cast<int> (std::ceil (std::abs (max_pt_all.y - min_pt_all.y) / resolution_)) + 1;
  gs_z_ = static_cast<int> (std::ceil (std::abs (max_pt_all.z - min_pt_all.z) / resolution_)) + 1;

  grid_.resize (gs_x_ * gs_y_ * gs_z_);
  for (int i = 0; i < (gs_x_ * gs_y_ * gs_z_); i++)
  {
    grid_[i].label_ = false;
    grid_[i].neigh_ = 0;
    grid_[i].n_ = 0;
    grid_[i].dist_ = 0;
  }

  typename pcl::PointCloud<PointT>::Ptr cloud_internal;

  /*typedef pcl::PointCloud<PointT> CloudM;
  typedef typename pcl::traits::fieldList<typename CloudM::PointType>::type FieldListM;
  float rgb_m;
  bool exists_m;;
  pcl::for_each_type<FieldListM> (pcl::CopyIfFieldExists<typename CloudM::PointType, float> (cloud_->points[0],"rgb", exists_m, rgb_m));

  if(exists_m) {
    size_t num_neigh = 9;
    std::vector<int> nn_indices (num_neigh);
    std::vector<float> nn_distances (num_neigh);

    typedef typename pcl::KdTree<PointT>::Ptr KdTreeInPtr;
    KdTreeInPtr tree_rgb;
    tree_rgb.reset(new pcl::KdTreeFLANN<PointT> (false));
    tree_rgb->setInputCloud (cloud_);

    std::vector<int> indices_to_keep;

    for (size_t j = 0; j < cloud_->points.size (); j++) {
      tree_rgb->nearestKSearch (cloud_->points[j], nn_indices.size(), nn_indices, nn_distances);
      pcl::for_each_type<FieldListM> (pcl::CopyIfFieldExists<typename CloudM::PointType, float> (cloud_->points[j],"rgb", exists_m, rgb_m));
      uint32_t rgb = *reinterpret_cast<int*> (&rgb_m);
      float rm = 0.f;
      float gm = 0.f;
      float bm = 0.f;

      //compute color mean
      std::vector<uint32_t> colors_neigh;
      colors_neigh.resize(nn_indices.size());

      for(size_t i=0; i < nn_indices.size(); i++) {
        pcl::for_each_type<FieldListM> (pcl::CopyIfFieldExists<typename CloudM::PointType, float> (cloud_->points[nn_indices[i]],"rgb", exists_m, rgb_m));
        uint32_t rgb_n = *reinterpret_cast<int*> (&rgb_m);
        rm += static_cast<float>((rgb_n >> 16) & 0x0000ff);
        gm += static_cast<float>((rgb_n >> 8) & 0x0000ff);
        bm += static_cast<float>((rgb_n) & 0x0000ff);
        colors_neigh[i] = rgb_n;
      }

      rm = rm / static_cast<float>(nn_indices.size());
      gm = gm / static_cast<float>(nn_indices.size());
      bm = bm / static_cast<float>(nn_indices.size());

      //compute color variance
      float sum2_r, sum2_g, sum2_b;
      sum2_r = sum2_g = sum2_b = 0;
      for(size_t i=0; i < colors_neigh.size(); i++) {
        float rm_v = static_cast<float>((colors_neigh[i] >> 16) & 0x0000ff);
        float gm_v = static_cast<float>((colors_neigh[i] >> 8) & 0x0000ff);
        float bm_v = static_cast<float>((colors_neigh[i]) & 0x0000ff);
        sum2_r += (rm_v - rm) * (rm_v - rm);
        sum2_g += (gm_v - gm) * (gm_v - gm);
        sum2_b += (bm_v - bm) * (bm_v - bm);
      }

      float variance_r = sum2_r / static_cast<float>(nn_indices.size() - 1);
      float variance_g = sum2_g / static_cast<float>(nn_indices.size() - 1);
      float variance_b = sum2_b / static_cast<float>(nn_indices.size() - 1);

      float sigma = (sqrt(variance_r) + sqrt(variance_g) + sqrt(variance_b)) / 3.f;

      if(sigma > 15.f) {
        indices_to_keep.push_back(j);
      }
    }

    cloud_internal.reset(new pcl::PointCloud<PointT>());
    pcl::copyPointCloud(*cloud_, indices_to_keep, *cloud_internal);
  }
  else
  {
    cloud_internal.reset(new pcl::PointCloud<PointT>(*cloud_));
  }*/

  cloud_internal.reset(new pcl::PointCloud<PointT>(*cloud_));

  //set occupancy
  std::cout << resolution_ << std::endl;
  for (size_t j = 0; j < cloud_internal->points.size (); j++)
  {
    int pos_x, pos_y, pos_z;
    pos_x = static_cast<int> (pcl_round ((cloud_internal->points[j].x - min_pt_all.x) / resolution_));
    pos_y = static_cast<int> (pcl_round ((cloud_internal->points[j].y - min_pt_all.y) / resolution_));
    pos_z = static_cast<int> (pcl_round ((cloud_internal->points[j].z - min_pt_all.z) / resolution_));

    assert(pos_x >= 0);
    assert(pos_y >= 0);
    assert(pos_z >= 0);

    assert(pos_x < gs_x_);
    assert(pos_y < gs_y_);
    assert(pos_z < gs_z_);

    int idx = pos_z * gs_x_ * gs_y_ + pos_y * gs_x_ + pos_x;
    if (grid_[idx].label_)
    {
      grid_[idx].avg_.getVector3fMap() = (grid_[idx].avg_.getVector3fMap() * static_cast<float> (grid_[idx].n_) + cloud_internal->points[j].getVector3fMap ())
          / static_cast<float> (grid_[idx].n_ + 1);

      /*typedef pcl::PointCloud<PointT> CloudM;
      typedef typename pcl::traits::fieldList<typename CloudM::PointType>::type FieldListM;
      float rgb_m;
      bool exists_m;;
      pcl::for_each_type<FieldListM> (pcl::CopyIfFieldExists<typename CloudM::PointType, float> (cloud_internal->points[j],"rgb", exists_m, rgb_m));

      if(exists_m) {
        uint32_t rgb = *reinterpret_cast<int*> (&rgb_m);
        float rm = static_cast<float>((rgb >> 16) & 0x0000ff);
        float gm = static_cast<float>((rgb >> 8) & 0x0000ff);
        float bm = static_cast<float>((rgb) & 0x0000ff);

        float rgb_avg;
        pcl::for_each_type<FieldListM> (pcl::CopyIfFieldExists<typename CloudM::PointType, float> (grid_[idx].avg_,"rgb", exists_m, rgb_avg));

        uint32_t rgb_avguint = *reinterpret_cast<int*> (&rgb_avg);
        float r_avg = static_cast<float>((rgb_avguint >> 16) & 0x0000ff);
        float g_avg = static_cast<float>((rgb_avguint >> 8) & 0x0000ff);
        float b_avg = static_cast<float>((rgb_avguint) & 0x0000ff);

        float new_r = (r_avg * static_cast<float> (grid_[idx].n_) + rm) / static_cast<float> (grid_[idx].n_ + 1);
        float new_g = (g_avg * static_cast<float> (grid_[idx].n_) + gm) / static_cast<float> (grid_[idx].n_ + 1);
        float new_b = (b_avg * static_cast<float> (grid_[idx].n_) + bm) / static_cast<float> (grid_[idx].n_ + 1);

        {
          int rgb = (static_cast<int> (new_r) << 16) | (static_cast<int> (new_g) << 8) | static_cast<int> (new_b);
          pcl::for_each_type<FieldListM> (
              pcl::SetIfFieldExists<typename CloudM::PointType, float>(grid_[idx].avg_, "rgb", rgb));
        }
      }*/

      grid_[idx].n_++;
    }
    else
    {
      grid_[idx].label_ = true;
      grid_[idx].avg_ = cloud_internal->points[j];
      grid_[idx].n_ = 1;
    }
  }

  //compute distance transform
  //we do it a non-efficent way but does not really matter as it is done once for each model
  //during initialization

  pcl::PointCloud<pcl::PointXYZ>::Ptr object (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr background (new pcl::PointCloud<pcl::PointXYZ>);
  for (int x = 0; x < gs_x_; x++)
  {
    for (int y = 0; y < gs_y_; y++)
    {
      for (int z = 0; z < gs_z_; z++)
      {
        int idx = z * gs_x_ * gs_y_ + y * gs_x_ + x;
        pcl::PointXYZ p;
        if (grid_[idx].label_)
        {
          p.getVector3fMap () = grid_[idx].avg_.getVector3fMap();
          object->push_back (p);
        }
        else
        {
          p.getVector3fMap () = Eigen::Vector3f (min_pt_all.x + static_cast<float> (x) * resolution_ + resolution_ / 2.f,
                                                 min_pt_all.y + static_cast<float> (y) * resolution_ + resolution_ / 2.f,
                                                 min_pt_all.z + static_cast<float> (z) * resolution_ + resolution_ / 2.f);

          background->push_back (p);
        }
      }
    }
  }

  typename pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (object);
  std::vector<int> indices;
  std::vector<float> distances;

  for (size_t j = 0; j < background->points.size (); j++)
  {
    int f = tree->nearestKSearch (*background, static_cast<int> (j), 1, indices, distances);
    if (f > 0)
    {
      int idx_background, idx_object;

      {
        int pos_x, pos_y, pos_z;
        pos_x = static_cast<int> (pcl_round ((background->points[j].x - min_pt_all.x - resolution_ / 2.f) / resolution_));
        pos_y = static_cast<int> (pcl_round ((background->points[j].y - min_pt_all.y - resolution_ / 2.f) / resolution_));
        pos_z = static_cast<int> (pcl_round ((background->points[j].z - min_pt_all.z - resolution_ / 2.f) / resolution_));

        assert(pos_x >= 0);
        assert(pos_y >= 0);
        assert(pos_z >= 0);

        assert(pos_x < gs_x_);
        assert(pos_y < gs_y_);
        assert(pos_z < gs_z_);

        idx_background = pos_z * gs_x_ * gs_y_ + pos_y * gs_x_ + pos_x;
      }

      {
        int pos_x, pos_y, pos_z;
        pos_x = static_cast<int> (pcl_round ((object->points[indices[0]].x - min_pt_all.x) / resolution_));
        pos_y = static_cast<int> (pcl_round ((object->points[indices[0]].y - min_pt_all.y) / resolution_));
        pos_z = static_cast<int> (pcl_round ((object->points[indices[0]].z - min_pt_all.z) / resolution_));

        assert(pos_x >= 0);
        assert(pos_y >= 0);
        assert(pos_z >= 0);

        assert(pos_x < gs_x_);
        assert(pos_y < gs_y_);
        assert(pos_z < gs_z_);

        idx_object = pos_z * gs_x_ * gs_y_ + pos_y * gs_x_ + pos_x;
      }

      grid_[idx_background].neigh_ = &(grid_[idx_object]);
      grid_[idx_background].dist_ = distances[0];
    }
  }

  voxelized_cloud_.reset (new pcl::PointCloud<PointT>);
  int pidx = 0;
  for (int x = 0; x < gs_x_; x++)
  {
    for (int y = 0; y < gs_y_; y++)
    {
      for (int z = 0; z < gs_z_; z++)
      {
        int idx = z * gs_x_ * gs_y_ + y * gs_x_ + x;
        if (grid_[idx].label_)
        {
          PointT p;
          p.getVector3fMap () = grid_[idx].avg_.getVector3fMap();
          voxelized_cloud_->push_back (p);
          grid_[idx].idx_ = pidx++;
        }
      }
    }
  }

  initialized_ = true;
  //visualizeGrid();
}

template<typename PointT>
void
faat_pcl::rec_3d_framework::VoxelGridDistanceTransform<PointT>::visualizeGrid() {
  pcl::visualization::PCLVisualizer vis ("visualizeIVData");

  typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  typename  pcl::PointCloud<PointT>::Ptr cloud_free (new pcl::PointCloud<PointT>);
  for (int x = 0; x < gs_x_; x++)
  {
    for (int y = 0; y < gs_y_; y++)
    {
      for (int z = 0; z < gs_z_; z++)
      {
        int idx = z * gs_x_ * gs_y_ + y * gs_x_ + x;
        if (grid_[idx].label_)
        {
          pcl::PointXYZRGB p;
          p.getVector3fMap () = Eigen::Vector3f (static_cast<float> (x), static_cast<float> (y), static_cast<float> (z));
          //p.getVector3fMap () = grid_[idx].avg_.getVector3fMap();
          typedef pcl::PointCloud<PointT> CloudM;
          typedef typename pcl::traits::fieldList<typename CloudM::PointType>::type FieldListM;
          float rgb_m;
          bool exists_m;;
          pcl::for_each_type<FieldListM> (pcl::CopyIfFieldExists<typename CloudM::PointType, float> (grid_[idx].avg_,"rgb", exists_m, rgb_m));
          if(exists_m) {
            p.rgb = rgb_m;
          } else {
            p.r = 255;
            p.g = 0;
            p.b = 255;
          }
          cloud->push_back (p);
        } else {
          PointT p;
          p.getVector3fMap () = Eigen::Vector3f (static_cast<float> (x), static_cast<float> (y), static_cast<float> (z));
          //p.getVector3fMap () = grid_[idx].avg_.getVector3fMap();
          cloud_free->push_back (p);
        }
      }
    }
  }

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> random_handler (cloud);
  vis.addPointCloud<pcl::PointXYZRGB> (cloud, random_handler, "original points");

  {
    pcl::visualization::PointCloudColorHandlerCustom<PointT> random_handler (cloud_free, 255, 255, 0);
    vis.addPointCloud<PointT> (cloud_free, random_handler, "free points");
  }

  vis.addCoordinateSystem (100, 0);
  vis.spin ();
}

template<typename PointT>
void
faat_pcl::rec_3d_framework::VoxelGridDistanceTransform<PointT>::getCorrespondence (const PointT & p, int * idx,
                                                                                              float * dist, float sigma, float * color_distance)
{
  int pos_x, pos_y, pos_z;
  pos_x = static_cast<int> (pcl_round ((p.x - min_pt_all.x) / resolution_));
  pos_y = static_cast<int> (pcl_round ((p.y - min_pt_all.y) / resolution_));
  pos_z = static_cast<int> (pcl_round ((p.z - min_pt_all.z) / resolution_));

  if (pos_x < 0 || pos_x >= gs_x_)
  {
    *dist = std::numeric_limits<float>::max ();
    *idx = -1;
    return;
  }

  if (pos_y < 0 || pos_y >= gs_y_)
  {
    *dist = std::numeric_limits<float>::max ();
    *idx = -1;
    return;
  }

  if (pos_z < 0 || pos_z >= gs_z_)
  {
    *dist = std::numeric_limits<float>::max ();
    *idx = -1;
    return;
  }

  assert(pos_x >= 0);
  assert(pos_y >= 0);
  assert(pos_z >= 0);

  assert(pos_x < gs_x_);
  assert(pos_y < gs_y_);
  assert(pos_z < gs_z_);

  int grid_idx = pos_z * gs_x_ * gs_y_ + pos_y * gs_x_ + pos_x;
  if(grid_[grid_idx].label_) {
    //voxel is occupied
    *idx = grid_[grid_idx].idx_;
    *dist = (p.getVector3fMap () - grid_[grid_idx].avg_.getVector3fMap()).norm ();
  } else {
    //voxel is free, get neigh
    *idx = grid_[grid_idx].neigh_->idx_;
    *dist = (p.getVector3fMap () - grid_[grid_idx].neigh_->avg_.getVector3fMap()).norm ();
  }

  if(sigma == 0.f) {
    *color_distance = 1.f;
  } else {
    float rgb_p, rgb_v;
    bool exists_m;
    bool exists_s;

    typedef pcl::PointCloud<PointT> CloudM;
    typedef typename pcl::traits::fieldList<typename CloudM::PointType>::type FieldListM;

    pcl::for_each_type<FieldListM> (
                                    pcl::CopyIfFieldExists<typename CloudM::PointType, float> (p, "rgb", exists_m, rgb_p));


    if(grid_[grid_idx].label_) {
      pcl::for_each_type<FieldListM> (
                                    pcl::CopyIfFieldExists<typename CloudM::PointType, float> (grid_[grid_idx].avg_,
                                                                                               "rgb", exists_s, rgb_v));
    } else {
      pcl::for_each_type<FieldListM> (
                                    pcl::CopyIfFieldExists<typename CloudM::PointType, float> (grid_[grid_idx].neigh_->avg_,
                                                                                               "rgb", exists_s, rgb_v));
    }

    if (exists_m && exists_s)
    {
      uint32_t rgb = *reinterpret_cast<int*> (&rgb_p);
      uint8_t rm = (rgb >> 16) & 0x0000ff;
      uint8_t gm = (rgb >> 8) & 0x0000ff;
      uint8_t bm = (rgb) & 0x0000ff;

      rgb = *reinterpret_cast<int*> (&rgb_v);
      uint8_t rs = (rgb >> 16) & 0x0000ff;
      uint8_t gs = (rgb >> 8) & 0x0000ff;
      uint8_t bs = (rgb) & 0x0000ff;
      Eigen::Vector3f yuvm, yuvs;

      float ym = 0.257f * rm + 0.504f * gm + 0.098f * bm + 16; //between 16 and 235
      float um = -(0.148f * rm) - (0.291f * gm) + (0.439f * bm) + 128;
      float vm = (0.439f * rm) - (0.368f * gm) - (0.071f * bm) + 128;

      float ys = 0.257f * rs + 0.504f * gs + 0.098f * bs + 16;
      float us = -(0.148f * rs) - (0.291f * gs) + (0.439f * bs) + 128;
      float vs = (0.439f * rs) - (0.368f * gs) - (0.071f * bs) + 128;

      yuvm = Eigen::Vector3f (static_cast<float> (ym), static_cast<float> (um), static_cast<float> (vm));
      yuvs = Eigen::Vector3f (static_cast<float> (ys), static_cast<float> (us), static_cast<float> (vs));

      float sigma2 = sigma * sigma;
      yuvm[0] *= 0.5f;
      yuvs[0] *= 0.5f;
      *color_distance = std::exp ((-0.5f * (yuvm - yuvs).squaredNorm ()) / (sigma2));
    }
  }
}

template class faat_pcl::rec_3d_framework::VoxelGridDistanceTransform<pcl::PointXYZRGB>;
template class faat_pcl::rec_3d_framework::VoxelGridDistanceTransform<pcl::PointXYZ>;
