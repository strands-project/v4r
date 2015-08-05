/*
 * ply_source.h
 *
 *  Created on: Mar 9, 2012
 *      Author: aitor
 */

#ifndef FAAT_PCL_REC_FRAMEWORK_MODEL_VIEWS_SOURCE_H_
#define FAAT_PCL_REC_FRAMEWORK_MODEL_VIEWS_SOURCE_H_

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

        void setExtension(std::string e)
        {
            ext_ = e;
        }

        void
        loadOrGenerate (std::string & dir, std::string & model_path, ModelT & model)
        {
          if(ext_.compare("pcd") == 0)
          {
              std::stringstream full_model;
              full_model << path_ << "/" << "/" << model.class_ << "/" << model.id_;
              typename pcl::PointCloud<Full3DPointT>::Ptr modell (new pcl::PointCloud<Full3DPointT>);
              typename pcl::PointCloud<Full3DPointT>::Ptr modell_voxelized (new pcl::PointCloud<Full3DPointT>);
              pcl::io::loadPCDFile(full_model.str(), *modell);

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
              uniform_sampling (model_path, 100000, *model_cloud, model_scale_);

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
        generate (std::string & training_dir)
        {
          //get models in directory
          std::vector < std::string > files;

          v4r::io::getFilesInDirectory (path_, files, "", ".*.pcd", false);
          std::cout << files.size() << std::endl;

          models_.reset (new std::vector<ModelTPtr>);

          for (size_t i = 0; i < files.size (); i++)
          {
            ModelTPtr m(new ModelT());

            std::vector < std::string > strs;
            boost::split (strs, files[i], boost::is_any_of ("/\\"));
            std::string name = strs[strs.size () - 1];

            if (strs.size () == 1)
            {
              m->id_ = strs[0];
            }
            else
            {
              std::stringstream ss;
              for (int i = 0; i < (static_cast<int> (strs.size ()) - 1); i++)
              {
                ss << strs[i];
                if (i != (static_cast<int> (strs.size ()) - 1))
                  ss << "/";
              }

              m->class_ = ss.str ();
              m->id_ = strs[strs.size () - 1];
            }

            std::cout << m->class_ << " . " << m->id_ << std::endl;
            //check which of them have been trained using training_dir and the model_id_
            //load views, poses and self-occlusions for those that exist
            //generate otherwise

            std::stringstream model_path;
            model_path << path_ << "/" << files[i];
            std::string path_model = model_path.str ();
            loadOrGenerate (training_dir, path_model, *m);

            models_->push_back (m);

            //std::cout << files[i] << std::endl;
          }
        }
      };
}

#endif /* REC_FRAMEWORK_MESH_SOURCE_H_ */
