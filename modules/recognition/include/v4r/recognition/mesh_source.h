/*
 * ply_source.h
 *
 *  Created on: Mar 9, 2012
 *      Author: aitor
 */

#ifndef FAAT_PCL_REC_FRAMEWORK_MESH_SOURCE_H_
#define FAAT_PCL_REC_FRAMEWORK_MESH_SOURCE_H_

#include "source.h"
#include <pcl/apps/render_views_tesselated_sphere.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include "vtk_model_sampling.h"
#include <boost/function.hpp>
#include <vtkTransformPolyDataFilter.h>

#include <v4r/common/faat_3d_rec_framework_defines.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/io/eigen.h>
#include <v4r/io/filesystem.h>
#include <v4r/tomgine/tgTomGineThread.h>
#include <v4r/tomgine/tgShapeCreator.h>
#include <v4r/tomgine/PointCloudRendering.h>

namespace v4r
{
    /**
     * \brief Data source class based on mesh models
     * \author Aitor Aldoma
     */

    template<typename PointInT>
      class V4R_EXPORTS MeshSource : public Source<PointInT>
      {
        typedef Source<PointInT> SourceT;
        typedef Model<PointInT> ModelT;
        typedef boost::shared_ptr<ModelT> ModelTPtr;

        using SourceT::path_;
        using SourceT::models_;
        using SourceT::createTrainingDir;
        using SourceT::model_scale_;
        using SourceT::radius_normals_;
        using SourceT::compute_normals_;
        using SourceT::load_into_memory_;

        int tes_level_;
        int resolution_;
        float radius_sphere_;
        float view_angle_;
        bool gen_organized_;
        boost::function<bool (const Eigen::Vector3f &)> campos_constraints_func_;

      public:

        using SourceT::setFilterDuplicateViews;

        MeshSource () :
        SourceT ()
        {
          gen_organized_ = false;
          load_into_memory_ = true;
        }

        ~MeshSource(){};

        void
        setTesselationLevel (int lev)
        {
          tes_level_ = lev;
        }

        void
        setCamPosConstraints (boost::function<bool
        (const Eigen::Vector3f &)> & bb)
        {
          campos_constraints_func_ = bb;
        }

        void
        setResolution (int res)
        {
          resolution_ = res;
        }

        void
        setRadiusSphere (float r)
        {
          radius_sphere_ = r;
        }

        void
        setViewAngle (float a)
        {
          view_angle_ = a;
        }

        void
        loadInMemorySpecificModel(std::string & dir, ModelT & model)
        {
          const std::string pathmodel = dir + "/" + model.class_ + "/" + model.id_;

          for (size_t i = 0; i < model.view_filenames_.size (); i++)
          {
            const std::string view_file = pathmodel + "/" + model.view_filenames_[i];
            typename pcl::PointCloud<PointInT>::Ptr cloud (new pcl::PointCloud<PointInT> ());
            pcl::io::loadPCDFile (view_file, *cloud);

            model.views_->push_back (cloud);

            std::string file_replaced1 (model.view_filenames_[i]);
            boost::replace_all (file_replaced1, "view", "pose");
            boost::replace_all (file_replaced1, ".pcd", ".txt");

            std::string file_replaced2 (model.view_filenames_[i]);
            boost::replace_all (file_replaced2, "view", "entropy");
            boost::replace_all (file_replaced2, ".pcd", ".txt");

            //read pose as well
            const std::string pose_file = pathmodel + "/" + file_replaced1;

            Eigen::Matrix4f pose;
            v4r::io::readMatrixFromFile(pose_file, pose);

            model.poses_->push_back (pose);

            //read entropy as well
            const std::string entropy_file = pathmodel + "/" + file_replaced2;
            float entropy = 0;
            v4r::io::readFloatFromFile (entropy_file, entropy);
            model.self_occlusions_->push_back (entropy);

          }
        }

        void
        loadOrGenerate (std::string & dir, std::string & model_path, ModelT & model)
        {
          const std::string pathmodel = dir + "/" + model.class_ + "/" + model.id_;

          model.views_.reset (new std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>);
          model.poses_.reset (new std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >);
          model.self_occlusions_.reset (new std::vector<float>);
          model.assembled_.reset (new pcl::PointCloud<pcl::PointXYZ>);
          uniform_sampling (model_path, 100000, *model.assembled_, model_scale_);

          if(compute_normals_) {
            std::cout << "Computing normals..." << std::endl;
            model.computeNormalsAssembledCloud(radius_normals_);
          }

          /*pcl::visualization::PCLVisualizer vis("results");
          pcl::visualization::PointCloudColorHandlerCustom<PointInT> random_handler (model.assembled_, 255, 0, 0);
          vis.addPointCloud<PointInT> (model.assembled_, random_handler, "points");
          vis.addPointCloudNormals<PointInT, pcl::Normal>(model.assembled_, model.normals_assembled_, 50, 0.02, "normals");
          vis.spin();*/

          if (v4r::io::getFilesInDirectory(pathmodel, model.view_filenames_, "", ".*view_.*.pcd", false) != -1)
          {
            if(load_into_memory_)
            {
              loadInMemorySpecificModel(dir, model);
              /*typename pcl::PointCloud<PointInT>::Ptr model_cloud(new pcl::PointCloud<PointInT>);
              assembleModelFromViewsAndPoses(model, *(model.poses_), *(model.indices_), model_cloud);

              pcl::visualization::PCLVisualizer vis ("assembled model...");
              pcl::visualization::PointCloudColorHandlerRGBField<PointInT> random_handler (model_cloud);
              vis.addPointCloud<PointInT> (model_cloud, random_handler, "points");
              vis.addCoordinateSystem(0.1);
              vis.spin ();*/
            }
          }
          else
          {

            //load PLY model and scale it
            /*vtkSmartPointer < vtkPLYReader > reader = vtkSmartPointer<vtkPLYReader>::New ();
            reader->SetFileName (model_path.c_str ());
            reader->Update();

            vtkSmartPointer < vtkTransform > trans = vtkSmartPointer<vtkTransform>::New ();
            trans->Scale (model_scale_, model_scale_, model_scale_);
            trans->Modified ();
            trans->Update ();

            vtkSmartPointer < vtkTransformFilter > filter_scale = vtkSmartPointer<vtkTransformFilter>::New ();
            filter_scale->SetTransform (trans);
            filter_scale->SetInputConnection (reader->GetOutputPort ());
						  std::cout << "Step 4" << std::endl;
            vtkSmartPointer < vtkPolyDataMapper > mapper = vtkSmartPointer<vtkPolyDataMapper>::New ();
            mapper->SetInputConnection (filter_scale->GetOutputPort ());
            mapper->Update ();*/

            int polyhedron = TomGine::tgShapeCreator::ICOSAHEDRON;
            int img_width = resolution_;
            int img_height = resolution_;
            const bool scale = true;
            const bool center = true;
            TomGine::PointCloudRendering pcr( model_path, scale, center );
            TomGine::tgModel sphere;
            TomGine::tgShapeCreator::CreateSphere(sphere, radius_sphere_, tes_level_, polyhedron);
//            sphere.m_line_width = 1.0f;
            TomGine::tgCamera cam;
            cam.SetViewport(img_width, img_height);

            // To preserve Kinect camera parameters (640x480 / f=525)
            const float f = 200.f;
            const float cx = img_width / 2.f;
            const float cy = img_height / 2.f;
            cam.SetIntrinsicCV(f, f, cx , cy, 0.01f, 10.0f);
            int pc_id(-1);

            for(size_t i=0; i<sphere.m_vertices.size(); i++)
            {
                TomGine::vec3& pos = sphere.m_vertices[i].pos;

                // set camera
                if(cross(pos,TomGine::vec3(0,1,0)).length()<TomGine::epsilon)
                    cam.LookAt(pos, TomGine::vec3(0,0,0), TomGine::vec3(0,0,1));
                else
                    cam.LookAt(pos, TomGine::vec3(0,0,0), TomGine::vec3(0,1,0));
                cam.ApplyTransform();

                const bool use_world_coordinate_system = false;
                pcr.Generate(cam, use_world_coordinate_system);
                const cv::Mat4f& pointcloud = pcr.GetPointCloud(i);

                // convert to PCL point cloud
                typename pcl::PointCloud<PointInT>::Ptr cloud (new pcl::PointCloud<PointInT>);
                cloud->points.resize(pointcloud.rows * pointcloud.cols);

                if (gen_organized_)
                {
                    cloud->width = pointcloud.cols;
                    cloud->height = pointcloud.rows;
                    for(int row_id=0; row_id<pointcloud.rows; row_id++) {
                        for(int col_id=0; col_id<pointcloud.cols; col_id++){
                            const cv::Vec4f pt = pointcloud.at<cv::Vec4f>(row_id, col_id);
                            const float x = pt[0];
                            const float y = pt[1];
                            const float z = pt[2];
//                            const float rgb = pt[3];

                            PointInT &p = cloud->at(col_id, row_id);
                            p.x = x;
                            p.y = y;
                            p.z = z;
                        }
                    }
                }
                else
                {
                    cloud->height = 1;
                    size_t kept=0;
                    for(int row_id=0; row_id<pointcloud.rows; row_id++) {
                        for(int col_id=0; col_id<pointcloud.cols; col_id++){
                            const cv::Vec4f pt = pointcloud.at<cv::Vec4f>(row_id, col_id);
                            const float x = pt[0];
                            const float y = pt[1];
                            const float z = pt[2];
//                            const float rgb = pt[3];

                            if ( std::isfinite(z) ) {
                                PointInT &p = cloud->points[kept];
                                p.x = x;
                                p.y = y;
                                p.z = z;
                                kept++;
                            }
                        }
                    }
                    cloud->points.resize(kept);
                    cloud->width = kept;
                }

                TomGine::mat4 tg_pose = cam.GetPose();

                Eigen::Matrix4f tf;
                for(size_t row_id=0; row_id<4; row_id++)
                    for(size_t col_id=0; col_id<4; col_id++)
                        tf(col_id, row_id) = tg_pose.data[row_id*4 + col_id];

                model.views_->push_back (cloud);
                model.poses_->push_back (tf);
                model.self_occlusions_->push_back (0); // NOT IMPLEMENTED
            }
			
            //generate views
//            pcl::apps::RenderViewsTesselatedSphere render_views;
//            render_views.setUseVertices (false);
//            render_views.setRadiusSphere (radius_sphere_);
//            render_views.setComputeEntropies (true);
//            render_views.setTesselationLevel (tes_level_);
//            render_views.setViewAngle (view_angle_);
//            //render_views.addModelFromPolyData (mapper->GetInput ());
//            render_views.addModelFromPolyData (mapper);
//            render_views.setGenOrganized(gen_organized_);
//            render_views.setCamPosConstraints(campos_constraints_func_);
//            render_views.generateViews ();

            std::stringstream direc;
            direc << dir << "/" << model.class_ << "/" << model.id_;
            this->createClassAndModelDirectories (dir, model.class_, model.id_);

            for (size_t i = 0; i < model.views_->size (); i++)
            {
              //save generated model for future use
              std::stringstream path_view;
              path_view << direc.str () << "/view_" << i << ".pcd";
              pcl::io::savePCDFileBinary (path_view.str (), *(model.views_->at (i)));

              std::stringstream path_pose;
              path_pose << direc.str () << "/pose_" << i << ".txt";

              v4r::io::writeMatrixToFile( path_pose.str (), model.poses_->at (i));

              std::stringstream path_entropy;
              path_entropy << direc.str () << "/entropy_" << i << ".txt";
              v4r::io::writeFloatToFile (path_entropy.str (), model.self_occlusions_->at (i));
            }

            loadOrGenerate (dir, model_path, model);

          }
        }

        /**
         * \brief Creates the model representation of the training set, generating views if needed
         */
        void
        generate (std::string & training_dir)
        {

          //create training dir fs if not existent
          v4r::io::createDirIfNotExist(training_dir);

          //get models in directory
          std::vector < std::string > files;
          v4r::io::getFilesInDirectory(path_, files, "", ".*.ply", true);

          models_.reset (new std::vector<ModelTPtr>);

          for (size_t i = 0; i < files.size (); i++)
          {
            ModelTPtr m(new ModelT);
            this->getIdAndClassFromFilename (files[i], m->id_, m->class_);

            //check which of them have been trained using training_dir and the model_id_
            //load views, poses and self-occlusions for those that exist
            //generate otherwise
            std::cout << files[i] << std::endl;
            std::stringstream model_path;
            model_path << path_ << "/" << files[i];
            std::string path_model = model_path.str ();
            loadOrGenerate (training_dir, path_model, *m);

            models_->push_back (m);
          }
          std::cout << "End of generate function" << std::endl;
        }
      };
}

#endif /* REC_FRAMEWORK_MESH_SOURCE_H_ */
