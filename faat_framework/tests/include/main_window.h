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
#include <faat_pcl/3d_rec_framework/utils/vtk_model_sampling.h>
#include <pcl/filters/voxel_grid.h>

namespace bf = boost::filesystem;

struct ModelPose {
  int nid_;
  std::string id_;
  Eigen::Matrix4f transform_;
};

class MainWindow : public QObject
{
   Q_OBJECT

public:
   MainWindow(int argc, char *argv[]);

   float getModelXSize() {
     return model_xsize_;
   }

   float getPoseXSize() {
     return pose_xsize_;
   }

   void moveCurrentObject(int direction, float step) {
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
   }

   void addSelectedStablePose(int selected_model) {

   }

   void addSelectedModelCloud(int selected_model) {
     if(selected_model > (int)(loaded_models_.size() - 1))
      return;

     counter_.setZero();
     pviz_->removeShape("curr_mesh");
     pviz_->addPointCloud(model_clouds_[selected_model], "curr_mesh");
     current_model_ = selected_model;
     icp_button_->setEnabled(true);
     gc_icp_button_->setEnabled(true);
   }

public slots:
  void cut_z();
  void lock_with_icp();
  void lock_with_gc_icp();
  void save_model();
  void save_to_disk();
  void scene_list_clicked(const QModelIndex & idx);

private:

  int getNumberOfModels(std::string DIR_PATH_) {
    bf::path ply_files_dir = DIR_PATH_;
    std::vector < std::string > files;
    std::string start = "";
    std::string ext = std::string ("ply");
    faat_pcl::rec_3d_framework::or_evaluator::getModelsInDirectory(ply_files_dir, start, files, ext);
    return (int)files.size();
  }

  void fillModels(std::string DIR_PATH_) {

    bf::path ply_files_dir = DIR_PATH_;
    std::vector < std::string > files;
    std::string start = "";
    std::string ext = std::string ("ply");
    faat_pcl::rec_3d_framework::or_evaluator::getModelsInDirectory(ply_files_dir, start, files, ext);

    size_t kk = (files.size ()) + 1;
    model_clouds_.resize(files.size ());

    double x_step = 1.0 / (float)kk;

    int viewport;
    for (size_t i = 0; i < files.size (); i++)
    {

      std::stringstream model_name;
      model_name << "poly_" << i;

      model_clouds_[i].reset(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr sampled_(new pcl::PointCloud<pcl::PointXYZ>);
      std::string path = DIR_PATH_ + "/" + files[i];
      faat_pcl::rec_3d_framework::uniform_sampling(path,100000, *sampled_, model_scale_);

      pviz_models_->createViewPort (i * x_step, 0, (i + 1) * x_step, 200, viewport);

      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*sampled_, centroid);
      centroids_.push_back(centroid);

      Eigen::Matrix4f transform;
      transform.setIdentity();
      transform(0,3) = -centroid[0];
      transform(1,3) = -centroid[1];
      transform(2,3) = -centroid[2];
      pcl::transformPointCloud(*sampled_, *sampled_, transform);
      //create scale transform...
      pviz_models_->addPointCloud(sampled_, model_name.str(), viewport);
      //vtkSmartPointer < vtkTransform > scale_models = vtkSmartPointer<vtkTransform>::New ();
      //scale_models->Scale(model_scale_, model_scale_, model_scale_);
      //pviz_models_->addModelFromPLYFile(DIR_PATH_ + "/" + files[i], scale_models, model_name.str(), viewport);

      float resolution = 0.001f;
      pcl::VoxelGrid<pcl::PointXYZ> grid_;
      grid_.setInputCloud (sampled_);
      grid_.setLeafSize (resolution, resolution, resolution);
      grid_.setDownsampleAllData(true);
      grid_.filter (*model_clouds_[i]);
      loaded_models_.push_back(DIR_PATH_ + "/" + files[i]);
    }

    pviz_models_->setRepresentationToSurfaceForAllActors();
  }

  QLineEdit * input_z_;
  QVTKWidget * vtk_widget_;
  QVTKWidget * vtk_widget_models_;
  QVTKWidget * vtk_widget_poses_;

  pcl::visualization::PCLVisualizer * pviz_;
  pcl::visualization::PCLVisualizer * pviz_models_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr orig_cloud_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr curr_cloud_;
  int current_model_;

  float model_scale_;
  float model_xsize_;
  float pose_xsize_;

  int CG_SIZE_;
  float CG_THRESHOLD_;
  std::string dir_models_;
  std::string dir_models_vis_;
  std::string dir_output_;
  std::string GT_DIR_;
  std::string training_dir_;
  std::string pcd_file_;

  int selected_model_;

  std::vector<std::string> scenes_list_;
  std::vector<std::string> loaded_models_;
  std::vector<std::string> addedToVis_;

  Eigen::Matrix4f transOCtoCanonical_;
  Eigen::Matrix4f transCanonicalToCC_;
  Eigen::Matrix4f last_transform_;

  Eigen::VectorXf counter_; //direction 0 and 2 for translation, 1 is for rotation about plane
  std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr, Eigen::aligned_allocator< pcl::PointCloud<
      pcl::PointXYZ>::Ptr> > scene_clusters_;

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> model_clouds_;
  std::vector<ModelPose> model_poses_;
  QWidget * mainWindow_;

  QListView * model_list_;
  QListView * scene_list_;
  int id_save_models_;
  QPushButton *save_model_;
  QPushButton *icp_button_;
  QPushButton *gc_icp_button_;
  double chop_at_z_;
  std::string scene_name_;
  std::string current_scene_id_;
  std::vector<Eigen::Vector4f> centroids_;

  faat_pcl::rec_3d_framework::or_evaluator::OREvaluator<pcl::PointXYZ> or_eval_;
};

#endif /* AFFORDANCE_MANAGER_H_ */
