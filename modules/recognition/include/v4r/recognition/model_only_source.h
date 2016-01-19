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


#ifndef V4R_MODEL_VIEWS_SOURCE_H_
#define V4R_MODEL_VIEWS_SOURCE_H_

#include "source.h"
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "v4r/common/faat_3d_rec_framework_defines.h"
#include "vtk_model_sampling.h"
#include <vtkTransformPolyDataFilter.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <v4r/io/filesystem.h>

namespace v4r
{

    /**
     * \brief Data source class based on partial views from sensor.
     * In this case, the training data is obtained directly from a depth sensor.
     * The filesystem should contain pcd files (representing a view of an object in
     * camera coordinates) and each view needs to be associated with a txt file
     * containing a 4x4 matrix representing the transformation from camera coordinates
     * to a global object coordinates frame.
     * \author Aitor Aldoma
     */

    template<typename Full3DPointT = pcl::PointXYZRGBNormal, typename PointInT = pcl::PointXYZRGB>
      class V4R_EXPORTS ModelOnlySource : public Source<PointInT>
      {
        typedef Source<PointInT> SourceT;
        typedef Model<PointInT> ModelT;
        typedef boost::shared_ptr<ModelT> ModelTPtr;

        using SourceT::path_;
        using SourceT::models_;
        using SourceT::model_scale_;
        using SourceT::load_into_memory_;
        using SourceT::radius_normals_;
        using SourceT::compute_normals_;
        std::string ext_;

        void computeFaces(ModelT & model);

      public:
        ModelOnlySource ()
        {
          load_into_memory_ = false;
          ext_ = "pcd";
        }

        void setExtension(const std::string &e)
        {
            ext_ = e;
        }

        void
        loadOrGenerate (const std::string & model_path, ModelT & model)

        {
          if(ext_.compare("pcd") == 0)
          {
              typename pcl::PointCloud<Full3DPointT>::Ptr modell (new pcl::PointCloud<Full3DPointT>);
              typename pcl::PointCloud<Full3DPointT>::Ptr modell_voxelized (new pcl::PointCloud<Full3DPointT>);
              pcl::io::loadPCDFile(model_path, *modell);

              float voxel_grid_size = 0.001f;
              typename pcl::VoxelGrid<Full3DPointT> grid_;
              grid_.setInputCloud (modell);
              grid_.setLeafSize (voxel_grid_size, voxel_grid_size, voxel_grid_size);
              grid_.setDownsampleAllData (true);
              grid_.filter (*modell_voxelized);

              model.normals_assembled_.reset(new pcl::PointCloud<pcl::Normal>);
              model.assembled_.reset (new pcl::PointCloud<PointInT>);

              pcl::copyPointCloud(*modell_voxelized, *model.assembled_);
              pcl::copyPointCloud(*modell_voxelized, *model.normals_assembled_);

              for(size_t kk=0; kk < model.normals_assembled_->points.size(); kk++)
              {
                  Eigen::Vector3f normal = model.normals_assembled_->points[kk].getNormalVector3fMap();
                  normal.normalize();
                  model.normals_assembled_->points[kk].getNormalVector3fMap() = normal;
              }

              computeFaces(model);

          }
          else if(ext_.compare("ply") == 0)
          {
              typename pcl::PointCloud<PointInT>::Ptr model_cloud(new pcl::PointCloud<PointInT>());
              uniform_sampling<PointInT> (model_path, 100000, *model_cloud, model_scale_);

              float resolution = 0.001f;
              pcl::VoxelGrid<PointInT> grid_;
              grid_.setInputCloud (model_cloud);
              grid_.setLeafSize (resolution, resolution, resolution);
              grid_.setDownsampleAllData(true);

              model.assembled_.reset (new pcl::PointCloud<PointInT>);
              grid_.filter (*(model.assembled_));

              if(compute_normals_)
              {
                std::cout << "Computing normals for ply models... " << radius_normals_ << std::endl;
                model.computeNormalsAssembledCloud(radius_normals_);
                model.setFlipNormalsBasedOnVP(true);
              }
          }
        }

        /**
         * \brief Creates the model representation of the training set, generating views if needed
         */
        void
        generate ()
        {
            models_.clear();
            std::vector < std::string > model_files = io::getFilesInDirectory (path_, ".*3D_model.pcd",  true);
            std::cout << "There are " << model_files.size() << " models." << std::endl;

          for (const std::string &model_file : model_files)
          {
            ModelTPtr m(new ModelT);

            std::vector < std::string > strs;
            boost::split (strs, model_file, boost::is_any_of ("/\\"));

            if (strs.size () == 2)  // class_name/id_name/3D_model.pcd
            {
                m->id_ = strs[0];
            }
            else if (strs.size()==3)
            {
                m->class_ = strs[0];
                m->id_ = strs[1];
            }
            else
            {
                std::cerr << "Given path " << path_ << " does not have required file structure: (optional: object_class_name)/object_id_name/3D_model.pcd !" << std::endl;
                m->id_ = strs[0];
            }

            //check which of them have been trained using training_dir and the model_id_
            //load views, poses and self-occlusions for those that exist
            //generate otherwise
            loadOrGenerate (path_+"/"+model_file, *m);
            models_.push_back (m);
          }
        }
      };
}

#endif
