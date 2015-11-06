/******************************************************************************
 * Copyright (c) 2012 Aitor Aldoma
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

/**
*
*      @author Aitor Aldoma
*      @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
*      @date March, 2012
*      @brief object model database
*/

#ifndef V4R_VIEWS_SOURCE_H_
#define V4R_VIEWS_SOURCE_H_

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <EDT/propagation_distance_field.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d_omp.h>
#include <boost/regex.hpp>
#include <v4r/core/macros.h>
#include <v4r/io/filesystem.h>

namespace bf = boost::filesystem;

namespace v4r
{
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

    /**
     * \brief Abstract data source class, manages filesystem, incremental training, etc.
     * \author Aitor Aldoma
     */

    template<typename PointInT>
    class V4R_EXPORTS Source
    {

    protected:
      typedef Model<PointInT> ModelT;
      typedef boost::shared_ptr<ModelT> ModelTPtr;

      std::vector<ModelTPtr> models_;
      std::string path_;
      float model_scale_;
      bool load_views_;
      float radius_normals_;
      float resolution_;
      bool compute_normals_;
      bool load_into_memory_;

      //List of model ids that will be loaded
      std::vector<std::string> model_list_to_load_;

      void
      getIdAndClassFromFilename (const std::string & filename, std::string & id, std::string & classname)
      {
        std::vector < std::string > strs;
        boost::split (strs, filename, boost::is_any_of ("/\\"));
        std::string name = strs[strs.size () - 1];

        classname = strs[0];
        id = name.substr (0, name.length () - 4);
      }

      void
      createClassAndModelDirectories (const std::string & training_dir, const std::string & class_str, const std::string & id_str)
      {
        std::vector < std::string > strs;
        boost::split (strs, class_str, boost::is_any_of ("/\\"));

        std::stringstream ss;
        ss << training_dir << "/";
        for (size_t i = 0; i < strs.size (); i++)
        {
          ss << strs[i] << "/";
          io::createDirIfNotExist(ss.str ());
        }

        ss << id_str;
        io::createDirIfNotExist(ss.str ());
      }

    public:

      Source(float resolution = 0.001f) {
        resolution_ = resolution;
        load_views_ = true;
        compute_normals_ = false;
        load_into_memory_ = true;
      }

      bool isModelIdInList(const std::string & id)
      {
          if(model_list_to_load_.empty())
              return true;

          for(size_t i=0; i < model_list_to_load_.size(); i++)
          {
              if(id.compare(model_list_to_load_[i]) == 0)
              {
                  return true;
              }
          }

          return false;
      }

      void setModelList(const std::vector<std::string> & list)
      {
          model_list_to_load_ = list;
      }

      void
      setLoadIntoMemory(bool b)
      {
        load_into_memory_ = b;
      }

      bool
      getLoadIntoMemory()
      {
        return load_into_memory_;
      }

      virtual void
      loadInMemorySpecificModelAndView(const std::string & dir, ModelT & model, int view_id)
      {
        (void)dir;
        (void)model;
        (void)view_id;
        PCL_ERROR("This function is not implemented in this Source class\n");
      }

      virtual void
      loadInMemorySpecificModel(const std::string & dir, ModelT & model)
      {
        (void)dir;
        (void)model;
        PCL_ERROR("This function is not implemented in this Source class\n");
      }

      float
      getScale ()
      {
        return model_scale_;
      }

      void
      setRadiusNormals(float r) {
        radius_normals_ = r;
        compute_normals_ = true;
      }

      void
      setModelScale (float s)
      {
        model_scale_ = s;
      }

      void
      voxelizeAllModels (int resolution_mm)
      {
        for (size_t i = 0; i < models_.size (); i++)
        {
          models_[i]->getAssembled (resolution_mm);
          if(compute_normals_)
            models_[i]->getNormalsAssembled (resolution_mm);
        }
      }

      /**
       * \brief Generate model representation
       */
      virtual void
      generate (const std::string & training_dir = std::string())=0;

      /**
       * \brief Get the generated model
       */
      std::vector<ModelTPtr>
      getModels ()
      {
        return models_;
      }

      bool
      getModelById (const std::string & model_id, ModelTPtr & m) const
      {
        for(size_t i=0; i<models_.size(); i++)
        {
            if(models_[i]->id_.compare(model_id)==0)
            {
                m = models_[i];
                return true;
            }
        }
        return false;
      }

      bool
      isModelAlreadyTrained (const ModelT m, const std::string & base_dir, const std::string & descr_name) const
      {
        std::stringstream dir;
        dir << base_dir << "/" << m.class_ << "/" << m.id_ << "/" << descr_name;
        bf::path desc_dir = dir.str ();
        if (bf::exists (desc_dir))
        {
          std::cout << dir.str () << " exists..." << std::endl;
          return true;
        }

        std::cout << dir.str () << " does not exist..." << std::endl;
        return false;
      }

      std::string
      getModelDescriptorDir (const ModelT m, const std::string & base_dir, const std::string & descr_name)
      {
        std::stringstream dir;
        dir << base_dir << "/" << m.class_ << "/" << m.id_ << "/" << descr_name;
        return dir.str ();
      }

      std::string
      getModelDirectory (const ModelT m, const std::string & base_dir)
      {
        std::stringstream dir;
        dir << base_dir << "/" << m.class_ << "/" << m.id_;
        return dir.str ();
      }

      std::string
      getModelClassDirectory (const ModelT m, const std::string & base_dir)
      {
        std::stringstream dir;
        dir << base_dir << "/" << m.class_;
        return dir.str ();
      }

      void
      removeDescDirectory (const ModelT &m, const std::string & base_dir, const std::string & descr_name)
      {
        const std::string dir = base_dir + "/" + m.class_ + "/" + m.id_ + "/" + descr_name;

        bf::path desc_dir = dir;
        if (bf::exists (desc_dir))
        bf::remove_all (desc_dir);
      }

      void
      setPath (const std::string & path)
      {
        path_ = path;
      }

      void setLoadViews(bool load)
      {
        load_views_ = load;
      }

      void
      createVoxelGridAndDistanceTransform(float resolution)
      {
        for (size_t i = 0; i < models_.size (); i++)
          models_[i]->createVoxelGridAndDistanceTransform (resolution);
      }
    };
}

#endif /* REC_FRAMEWORK_VIEWS_SOURCE_H_ */
