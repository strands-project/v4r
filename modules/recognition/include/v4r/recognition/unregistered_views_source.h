/*
 * ply_source.h
 *
 *  Created on: Mar 9, 2012
 *      Author: aitor
 */

#ifndef FAAT_PCL_REC_FRAMEWORK_UNREG_VIEWS_SOURCE_H_
#define FAAT_PCL_REC_FRAMEWORK_UNREG_VIEWS_SOURCE_H_

#include "source.h"
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "v4r/ORUtils/faat_3d_rec_framework_defines.h"
#include <v4r/utils/filesystem_utils.h>


namespace v4r
{
  namespace rec_3d_framework
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

    template<typename PointInT = pcl::PointXYZRGB>
      class UnregisteredViewsSource : public Source<PointInT>
      {
        typedef Source<PointInT> SourceT;
        typedef Model<PointInT> ModelT;
        typedef boost::shared_ptr<ModelT> ModelTPtr;

        using SourceT::path_;
        using SourceT::models_;
        using SourceT::createTrainingDir;
        using SourceT::model_scale_;
        using SourceT::load_into_memory_;
        using SourceT::createClassAndModelDirectories;

        std::string model_structure_; //directory with all the views, indices, poses, etc...
        std::string view_prefix_;
        std::string indices_prefix_;

      public:
        UnregisteredViewsSource ()
        {
          view_prefix_ = std::string ("");
          indices_prefix_ = std::string("object_indices_");
          load_into_memory_ = true;
        }

        void
        setModelStructureDir(std::string dir)
        {
          model_structure_ = dir;
        }

        void
        setPrefix (std::string & pre)
        {
          view_prefix_ = pre;
        }

        void
        loadInMemorySpecificModel(std::string & dir, ModelT & model)
        {
          std::stringstream pathmodel;
          pathmodel << dir << "/" << model.class_ << "/" << model.id_;
          //bf::path model_name = dir;

          //if (bf::exists (trained_dir))
          //{
            //for (size_t i = 0; i < model.view_filenames_.size (); i++)
            //{
              //std::stringstream view_file;
              //view_file << model.view_filenames_[0];
              typename pcl::PointCloud<PointInT>::Ptr cloud (new pcl::PointCloud<PointInT> ());
              pcl::io::loadPCDFile (model.view_filenames_[0], *cloud);

              std::string directory, filename;
              char sep = '/';
               #ifdef _WIN32
                  sep = '\\';
               #endif

              size_t position = model.view_filenames_[0].rfind(sep);
                 if (position != std::string::npos)
                 {
                    directory = model.view_filenames_[0].substr(0, position);
                    filename = model.view_filenames_[0].substr(position+1, model.view_filenames_[0].length()-1);
                 }

              std::stringstream indices_file;
              indices_file << directory << sep << indices_prefix_ << filename;

              pcl::PointCloud<IndexPoint> obj_indices_cloud;

              pcl::io::loadPCDFile (indices_file.str(), obj_indices_cloud);
              pcl::PointIndices indices;
              indices.indices.resize(obj_indices_cloud.points.size());
              for(size_t kk=0; kk < obj_indices_cloud.points.size(); kk++)
                indices.indices[kk] = obj_indices_cloud.points[kk].idx;

              model.views_->push_back (cloud);
              model.indices_->push_back(indices);
            //}
          //}
        }

        void
        loadOrGenerate (std::string & model_path, ModelT & model)
        {
          model.views_.reset (new std::vector<typename pcl::PointCloud<PointInT>::Ptr>);
          model.indices_.reset (new std::vector<pcl::PointIndices>);

            //load views and poses
            model.view_filenames_.push_back(model_path);

            if(load_into_memory_)
            {
              loadInMemorySpecificModel(model_path, model);

              /*typename pcl::PointCloud<PointInT>::Ptr model_cloud(new pcl::PointCloud<PointInT>);
              assembleModelFromViewsAndPoses(model, *(model.poses_), *(model.indices_), model_cloud);

              pcl::visualization::PCLVisualizer vis ("assembled model...");
              pcl::visualization::PointCloudColorHandlerRGBField<PointInT> random_handler (model_cloud);
              vis.addPointCloud<PointInT> (model_cloud, random_handler, "points");
              vis.addCoordinateSystem(0.1);
              vis.spin ();*/
            }
        }

//        bool
//        isleafDirectory (bf::path & path)
//        {
//          bf::directory_iterator end_itr;
//          bool no_dirs_inside = true;
//          for (bf::directory_iterator itr (path); itr != end_itr; ++itr)
//          {
//            if (bf::is_directory (*itr))
//            {
//              no_dirs_inside = false;
//            }
//          }

//          return no_dirs_inside;
//        }

        /**
         * \brief Creates the model representation of the training set, generating views if needed
         */
        void
        generate (std::string & dummy)
        {
          //create training dir fs if not existent
          //createTrainingDir (training_dir);

          models_.reset (new std::vector<ModelTPtr>);

          //get models in directory
          std::vector < std::string > folders;
          v4r::utils::getFoldersInDirectory (path_, "", folders);
          std::cout << "There are " << folders.size() << " folders. " << std::endl;

          for (size_t i = 0; i < folders.size (); i++)
          {
              std::stringstream class_path;
              class_path << path_ << "/" << folders[i];
              std::vector < std::string > filesInRelFolder;
              v4r::utils::getFilesInDirectory (class_path.str(), filesInRelFolder, "", ".*.pcd", false);
              std::cout << "There are " <<  filesInRelFolder.size() << " files in folder " << folders[i] << ". " << std::endl;

              for (size_t kk = 0; kk < filesInRelFolder.size (); kk++)
              {
                  if(filesInRelFolder[kk].length() > indices_prefix_.length())
                  {
                      if((filesInRelFolder[kk].compare(0, indices_prefix_.length(), indices_prefix_)==0 ))
                      {
                          std::cout << filesInRelFolder[kk] << " is not a cloud. " << std::endl;
                          continue;
                      }
                  }

                  ModelTPtr m(new ModelT());
                  m->class_ = folders[i];
                  m->id_ = filesInRelFolder[kk];

                  std::stringstream model_path;
                  model_path << class_path.str() << "/" << filesInRelFolder[kk];
                  std::string path_model = model_path.str ();
                  std::cout << "Calling loadOrGenerate path_model: " << path_model << ", m_class: " << m->class_ << ", m_id: " << m->id_ << std::endl;
                  loadOrGenerate (path_model, *m);

                  models_->push_back (m);
              }
          }
        }
      };
  }
}

#endif /* REC_FRAMEWORK_MESH_SOURCE_H_ */
