/*
 * affordance_manager.h
 *
 *  Created on: Jul 18, 2011
 *      Author: aitor
 */

#ifndef MAIN_WINDOW_H_
#define MAIN_WINDOW_H_

#include <QVTKWidget.h>
#include <QObject>
#include <QtGui>
#include "pcl/visualization/pcl_visualizer.h"
#include <boost/filesystem.hpp>
#include <QLineEdit>
#include <faat_pcl/3d_rec_framework/tools/or_evaluator.h>
#include <pcl/filters/voxel_grid.h>
#include <faat_pcl/3d_rec_framework/pc_source/model_only_source.h>
#include <faat_pcl/utils/miscellaneous.h>

namespace bf = boost::filesystem;

struct ModelPose {
  int nid_;
  std::string id_;
  Eigen::Matrix4f transform_;
};

class MainWindow : public QObject
{
   Q_OBJECT

   typedef pcl::PointXYZRGB PointT;
   typedef faat_pcl::rec_3d_framework::Model<PointT> ModelT;
   typedef boost::shared_ptr<ModelT> ModelTPtr;

public:
   MainWindow(int argc, char *argv[]);

   float getModelXSize() {
     return model_xsize_;
   }

   float getPoseXSize() {
     return pose_xsize_;
   }

   void moveCurrentObject(int direction, float step) {

   }

   /*void moveCurrentObject(int direction, float step) {
     pviz_->removeShape("curr_mesh");
     pviz_->removeShape("curr_mesh_file");

     counter_(direction) += step;

     //translation on plane transformation matrix
     Eigen::Matrix4f translation_plane;
     translation_plane.setIdentity();
     translation_plane(0,3) = counter_(0);
     translation_plane(1,3) = counter_(1);
     translation_plane(2,3) = counter_(2);

     //X
     Eigen::Matrix4f x,y,z;
     x.setIdentity();
     y.setIdentity();
     z.setIdentity();

     x(1,1) = cos(counter_(3));
     x(2,2) = x(1,1);

     x(2,1) = sin(counter_(3));
     x(1,2) = -x(2,1);

     //Y
     y(0,0) = cos(counter_(4));
     y(2,2) = y(0,0);

     y(0,2) = sin(counter_(4));
     y(2,0) = -y(0,2);

     //Z
     z(0,0) = cos(counter_(5));
     z(1,1) = z(0,0);

     z(1,0) = sin(counter_(5));
     z(0,1) = -z(1,0);

     Eigen::Matrix4f transform = translation_plane * x * y * z;

     pcl::PointCloud<pcl::PointXYZ>::Ptr sampled_ (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::transformPointCloud (*model_clouds_[current_model_], *sampled_, transform);
    pviz_->addPointCloud (sampled_, "curr_mesh");
    //pviz_->addModelFromPLYFile(loaded_models_[selected_model_], trans, "curr_mesh");

    if (!dir_models_vis_.empty ())
    {
      std::stringstream pathPly;
      std::string model_name = loaded_models_[current_model_];
      boost::replace_all(model_name,dir_models_,"");
      pathPly << dir_models_vis_ << "/" << model_name;

      std::cout << loaded_models_[current_model_] << " " << pathPly.str() << std::endl;

      vtkSmartPointer < vtkTransform > poseTransform = vtkSmartPointer<vtkTransform>::New ();
      vtkSmartPointer < vtkTransform > scale_models = vtkSmartPointer<vtkTransform>::New ();
      scale_models->Scale (model_scale_, model_scale_, model_scale_);

      vtkSmartPointer < vtkMatrix4x4 > mat = vtkSmartPointer<vtkMatrix4x4>::New ();
      for (size_t kk = 0; kk < 4; kk++)
      {
        for (size_t k = 0; k < 4; k++)
        {
          mat->SetElement (kk, k, transform (kk, k));
        }
      }

      poseTransform->SetMatrix (mat);
      poseTransform->Modified ();
      poseTransform->Concatenate (scale_models);

      pviz_->addModelFromPLYFile (pathPly.str(), poseTransform, "curr_mesh_file");
    }
    pviz_->spinOnce(0.1,true);

     last_transform_ = transform;
   }*/

   void initialPoseEstimate(Eigen::Vector3f trans)
   {
       if(selected_hypothesis_ < 0)
           return;

       Eigen::Matrix4f mat = hypotheses_poses_[selected_hypothesis_];
       mat.block<3,1>(0,3) = trans;
       hypotheses_poses_[selected_hypothesis_] = mat;

       int i = selected_hypothesis_;
       pcl::PointCloud<PointT>::ConstPtr model_cloud = sequence_hypotheses_[i]->getAssembled(0.003f);
       pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>(*model_cloud));
       pcl::transformPointCloud(*cloud, *cloud, hypotheses_poses_[i]);

       //remove highlighted and hypothesis
       pviz_->removePointCloud("highlighted");
       std::stringstream model_name;
       model_name << "hypotheses_" << i;
       pviz_->removePointCloud(model_name.str());

       //add again with new pose
       pviz_->addPointCloud(cloud, model_name.str(), pviz_v2_);
       pcl::visualization::PointCloudColorHandlerCustom<PointT> scene_handler(cloud, 0, 255, 0);
       pviz_->addPointCloud(cloud, scene_handler, "highlighted");
       pviz_->spinOnce(0.1, true);
   }

   void addSelectedModelCloud(int selected_model)
   {
     ModelTPtr clicked_model = loaded_models_[selected_model];

     //create hypothesis
     QStringList list;
     size_t i;
     for (i = 0; i < sequence_hypotheses_.size(); i++)
     {
       list << QString(sequence_hypotheses_[i]->id_.c_str());
     }

     sequence_hypotheses_.push_back(clicked_model);
     hypotheses_poses_.push_back(Eigen::Matrix4f::Identity());
     std::stringstream model_name;
     model_name << "hypotheses_" << i;

     pcl::PointCloud<PointT>::ConstPtr model_cloud = sequence_hypotheses_[i]->getAssembled(0.003f);
     pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>(*model_cloud));
     pcl::transformPointCloud(*cloud, *cloud, hypotheses_poses_[i]);

     list << QString(sequence_hypotheses_[i]->id_.c_str());

     model_list_->setModel(new QStringListModel(list));

     pviz_->addPointCloud(cloud, model_name.str(), pviz_v2_);

     /*if(selected_model > (int)(loaded_models_.size() - 1))
      return;

     counter_.setZero();
     pviz_->removeShape("curr_mesh");
     pviz_->addPointCloud(model_clouds_[selected_model], "curr_mesh");
     current_model_ = selected_model;
     icp_button_->setEnabled(true);
     gc_icp_button_->setEnabled(true);*/
   }

   pcl::PointCloud<PointT>::Ptr scene_merged_cloud_;

public slots:
  void lock_with_icp();
  void save_model();
  void remove_selected();
  void model_list_clicked(const QModelIndex & idx);
  void x_plus();
  void x_minus();
  void y_plus();
  void y_minus();
  void z_plus();
  void z_minus();
  void xr_plus();
  void xr_minus();
  void yr_plus();
  void yr_minus();
  void zr_plus();
  void zr_minus();
  void refine_pose_changed(int state);
private:

  void poseRefinementButtonPressed();

  void getModelsInDirectoryNonRecursive (bf::path & dir, std::string & rel_path_so_far, std::vector<std::string> & relative_paths, std::string & ext);

  void fillScene()
  {
      scene_merged_cloud_.reset(new pcl::PointCloud<PointT>);
      pcl::PointCloud<PointT>::Ptr merged_cloud (new pcl::PointCloud<PointT>);

      for (size_t i = 0; i < single_scenes_.size (); i++)
      {
          pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
          pcl::transformPointCloud(*single_scenes_[i], *cloud, single_clouds_to_global_[i]);
          *merged_cloud += *cloud;
      }

      faat_pcl::utils::miscellaneous::voxelGridWithOctree(merged_cloud, *scene_merged_cloud_, 0.003f);

      pcl::visualization::PointCloudColorHandlerRGBField<PointT> scene_handler(scene_merged_cloud_);
      pviz_->addPointCloud(scene_merged_cloud_, scene_handler, "merged_cloud",pviz_v1_);
      pviz_->spinOnce(0.1, true);
  }

  void readResultsFile(std::string result_file)
  {
      std::ifstream in;
      in.open (result_file.c_str (), std::ifstream::in);

      std::string line;
      while (std::getline(in, line))
      {
          std::vector < std::string > strs_2;
          boost::split (strs_2, line, boost::is_any_of ("\t"));
          std::cout << strs_2.size() << std::endl;
          std::string id = strs_2[0];

          Eigen::Matrix4f matrix;
          int k = 0;
          for (int i = 1; i < 17; i++, k++)
          {
            matrix (k / 4, k % 4) = static_cast<float> (atof (strs_2[i].c_str ()));
          }

          std::cout << id << std::endl;
          std::cout << matrix << std::endl;

          {
              std::vector < std::string > strs_2;
              boost::split (strs_2, id, boost::is_any_of ("/\\"));
              std::cout << strs_2[strs_2.size () - 1] << std::endl;
              ModelTPtr model;
              bool found = source_->getModelById (strs_2[strs_2.size () - 1], model);
              sequence_hypotheses_.push_back(model);
          }

          hypotheses_poses_.push_back(matrix);
      }

      in.close ();
  }

  void fillHypotheses()
  {
      pviz_->removeAllPointClouds(pviz_v2_);


      QStringList list;

      for (size_t i = 0; i < sequence_hypotheses_.size(); i++)
      {

        std::stringstream model_name;
        model_name << "hypotheses_" << i;

        pcl::PointCloud<PointT>::ConstPtr model_cloud = sequence_hypotheses_[i]->getAssembled(0.003f);
        pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>(*model_cloud));
        pcl::transformPointCloud(*cloud, *cloud, hypotheses_poses_[i]);

        //create scale transform...
        pviz_->addPointCloud(cloud, model_name.str(), pviz_v2_);
        list << QString(sequence_hypotheses_[i]->id_.c_str());

        //loaded_models_.push_back(DIR_PATH_ + "/" + files[i]);
        //loaded_models_.push_back(sequence_hypotheses_[i]->id_);
      }

      model_list_->setModel(new QStringListModel(list));

  }

  void fillModels() {

    boost::shared_ptr<std::vector<ModelTPtr> > models = source_->getModels();
    size_t kk = (models->size ()) + 1;
    model_clouds_.resize(models->size ());

    double x_step = 1.0 / (float)kk;

    int viewport;
    for (size_t i = 0; i < models->size (); i++)
    {

      std::stringstream model_name;
      model_name << "poly_" << i;

      model_clouds_[i].reset(new pcl::PointCloud<PointT>);
      pcl::PointCloud<PointT>::ConstPtr model_cloud = models->at(i)->getAssembled(0.003f);
      pviz_models_->createViewPort (i * x_step, 0, (i + 1) * x_step, 200, viewport);

      //create scale transform...
      pviz_models_->addPointCloud(model_cloud, model_name.str(), viewport);
      loaded_models_.push_back(models->at(i));
    }

    pviz_models_->spinOnce(0.1, true);
  }

  QLineEdit * input_z_;
  QVTKWidget * vtk_widget_;
  QVTKWidget * vtk_widget_models_;
  QVTKWidget * vtk_widget_poses_;

  pcl::visualization::PCLVisualizer * pviz_;
  pcl::visualization::PCLVisualizer * pviz_models_;

  float model_scale_;
  float model_xsize_;
  float pose_xsize_;

  int CG_SIZE_;
  float CG_THRESHOLD_;
  std::string dir_models_;
  std::string pcd_file_;

  std::vector<pcl::PointCloud<PointT>::Ptr> model_clouds_;

  QWidget * mainWindow_;
  QCheckBox * refine_pose_;

  QListView * model_list_;

  QPushButton *save_model_;
  QPushButton *icp_button_;

  boost::shared_ptr < faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB> > source_;

  std::vector<Eigen::Matrix4f> single_clouds_to_global_;
  std::vector< pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator< pcl::PointCloud<PointT>::Ptr> > single_scenes_;
  std::vector<ModelTPtr> sequence_hypotheses_;
  std::vector<Eigen::Matrix4f> hypotheses_poses_;
  pcl::PointCloud<PointT>::Ptr merged_cloud_for_refinement_;
  pcl::PointCloud<PointT>::Ptr merged_cloud_for_refinement_start_;

  int selected_hypothesis_;
  std::vector<ModelTPtr> loaded_models_;
  int pviz_v1_, pviz_v2_;
  float translation_step_;
  float rotation_step_;
  Eigen::Matrix4f pose_refined_;
  std::string export_ground_truth_to_;
  std::vector<std::string> scene_names_;
  bool icp_scene_to_model_;
};

#endif /* AFFORDANCE_MANAGER_H_ */
