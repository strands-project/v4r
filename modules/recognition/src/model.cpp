#include <v4r/recognition/model.h>
#include <sstream>


namespace v4r
{
template<typename PointT>
void
Model<PointT>::computeNormalsAssembledCloud(float radius_normals)
{
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


template<typename PointT>
typename pcl::PointCloud<PointT>::ConstPtr
Model<PointT>::getAssembled (int resolution_mm) const
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
      voxelized_assembled_[resolution_mm] = voxelized_const;
      return voxelized_const;
    }

    return it->second;
}

template<typename PointT>
pcl::PointCloud<pcl::Normal>::ConstPtr
Model<PointT>::getNormalsAssembled (int resolution_mm) const
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


      normals_voxelized_assembled_[resolution_mm] = voxelized_const;
      return voxelized_const;
    }

    return it->second;
}

template<typename PointT>
pcl::PointCloud<pcl::PointXYZL>::Ptr
Model<PointT>::getAssembledSmoothFaces (int resolution_mm)
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

      voxelized_assembled_labels_[resolution_mm] = voxelized;
      return voxelized;
    }

    return it->second;
}

template class V4R_EXPORTS Model<pcl::PointXYZ>;
template class V4R_EXPORTS Model<pcl::PointXYZRGB>;
}

