/*
 * affordance_manager.h
 *
 *  Created on: Jan 2015
 *      Author: Aitor Aldoma, Thomas Faeulhammer
 */

#ifndef MAIN_WINDOW_H_
#define MAIN_WINDOW_H_

#ifndef Q_MOC_RUN
#include <QVTKWidget.h>
#include <QObject>
#include <QtGui>
#include "pcl/visualization/pcl_visualizer.h"
#include <boost/filesystem.hpp>
#include <QLineEdit>
#include "or_evaluator.h"
#include <pcl/filters/voxel_grid.h>
#include <v4r/recognition/model_only_source.h>
#include <v4r/common/miscellaneous.h>
#endif

namespace bf = boost::filesystem;

class MainWindow : public QObject
{
   Q_OBJECT

   typedef pcl::PointXYZRGB PointT;
   typedef v4r::Model<PointT> ModelT;
   typedef boost::shared_ptr<ModelT> ModelTPtr;

   Eigen::Vector4f zero_origin;

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

   void initialPoseEstimate(Eigen::Vector3f trans)
   {
       if(selected_hypothesis_ < 0)
       {
           std::cout << "No hypothesis selected." << std::endl;
           return;
       }

       hypotheses_poses_[ selected_hypothesis_ ].block<3,1>(0,3) = trans;

       std::stringstream model_name;
       model_name << "hypotheses_" << selected_hypothesis_;

       pcl::PointCloud<PointT>::ConstPtr model_cloud = sequence_hypotheses_[selected_hypothesis_]->getAssembled( 3 );
       pcl::PointCloud<PointT>::Ptr model_cloud_transformed(new pcl::PointCloud<PointT>(*model_cloud));
       pcl::transformPointCloud(*model_cloud, *model_cloud_transformed, hypotheses_poses_[selected_hypothesis_]);

       pviz_->removePointCloud(model_name.str(), pviz_v2_);
       pviz_->addPointCloud(model_cloud_transformed, model_name.str(), pviz_v2_);

       pviz_->removePointCloud("highlighted");
       pcl::visualization::PointCloudColorHandlerCustom<PointT> scene_handler(model_cloud_transformed, 0, 255, 0);
       pviz_->addPointCloud(model_cloud_transformed, scene_handler, "highlighted");

       pviz_->spinOnce(100, true);
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

     pcl::PointCloud<PointT>::ConstPtr model_cloud = sequence_hypotheses_[i]->getAssembled( 3 );
     pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>(*model_cloud));
     pcl::transformPointCloud(*cloud, *cloud, hypotheses_poses_[i]);

     list << QString(sequence_hypotheses_[i]->id_.c_str());

     model_list_->setModel(new QStringListModel(list));
     QModelIndex index = model_list_->model()->index(model_list_->model()->rowCount()-1,0);
     model_list_->selectionModel()->select( index, QItemSelectionModel::Select );
     selected_hypothesis_ = sequence_hypotheses_.size()-1;
     selected_scene_ = -1;
     enablePoseRefinmentButtons(true);

     pviz_->addPointCloud(cloud, model_name.str(), pviz_v2_);
     updateSelectedHypothesis();

     /*if(selected_model > (int)(loaded_models_.size() - 1))
      return;

     counter_.setZero();
     pviz_->removeShape("curr_mesh");
     pviz_->addPointCloud(model_clouds_[selected_model], "curr_mesh");
     current_model_ = selected_model;
     icp_button_->setEnabled(true);
     gc_icp_button_->setEnabled(true);*/
   }

   void selectScene(int id)
   {
       selected_scene_ = id;
       for(size_t i=0; i< view_viewport.size(); i++)
       {
           pviz_scenes_->setBackgroundColor(0,0,0,view_viewport[i]);
       }
       if(selected_scene_>=0)
       {
         pviz_scenes_->setBackgroundColor(1,1,1,view_viewport[selected_scene_]);
       }
   }

   void selectModel(int id)
   {
       selected_hypothesis_ = id;
       for(size_t i=0; i< view_viewport.size(); i++)
       {
           pviz_scenes_->setBackgroundColor(0,0,0,view_viewport[i]);
       }
       if(selected_scene_>=0)
       {
         pviz_scenes_->setBackgroundColor(1,1,1,view_viewport[selected_scene_]);
       }

   }

   void updateHighlightedScene(bool highlight=false)
   {
       for (int i = 0; i < (int)single_scenes_.size (); i++)
       {
           if( i == selected_scene_)
           {
               std::stringstream cloud_name;
               cloud_name << "view_" << i;
               pviz_->removePointCloud("highlighted");
               pviz_->removePointCloud(cloud_name.str());
               pviz_->addPointCloud(single_scenes_[i], cloud_name.str(), pviz_v1_);

               if(highlight)
               {
                   pcl::visualization::PointCloudColorHandlerCustom<PointT> scene_handler(single_scenes_[i], 0, 0, 255);
                   pviz_->addPointCloud(single_scenes_[i], scene_handler, "highlighted", pviz_v1_);
               }
           }
       }
       pviz_->spinOnce(100, true);
   }

   void clear()
   {
       single_clouds_to_global_.clear();
       single_scenes_.clear();
       sequence_hypotheses_.clear();
       hypotheses_poses_.clear();

       selected_hypothesis_ = -1;
       selected_scene_ = -1;
       scene_names_.clear();
       scene_merged_cloud_.reset (new pcl::PointCloud<PointT>());
       loaded_models_.clear();
       pviz_->removePointCloud("highlighted");
   }

   void fillScene();

   pcl::PointCloud<PointT>::Ptr scene_merged_cloud_;

public Q_SLOTS:
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
  void next();
  void prev();
  void enablePoseRefinmentButtons(bool flag);
  void updateSelectedHypothesis();
//  void refine_pose_changed(int state);
private:

  void readResultsFile(const std::string &result_file);
  void fillHypotheses();
  void fillModels();
  void fillViews();

  QLineEdit * input_z_;
  QVTKWidget * vtk_widget_;
  QVTKWidget * vtk_widget_models_;
  QVTKWidget * vtk_widget_scenes_;
  QVTKWidget * vtk_widget_poses_;

  pcl::visualization::PCLVisualizer * pviz_;
  pcl::visualization::PCLVisualizer * pviz_models_;
  pcl::visualization::PCLVisualizer * pviz_scenes_;

  double model_scale_;
  float model_xsize_;
  float pose_xsize_;

  std::string dir_models_;
  std::string base_path_;
  std::vector<pcl::PointCloud<PointT>::Ptr> model_clouds_;

  QWidget * mainWindow_;
  QListView * model_list_;
  QPushButton *save_model_;
  QPushButton *icp_button_;
  QPushButton * remove_highlighted_hypotheses_;
  QPushButton * x_plus_;
  QPushButton * x_minus_;
  QPushButton * y_plus_;
  QPushButton * y_minus_;
  QPushButton * z_plus_;
  QPushButton * z_minus_;
  QPushButton * xr_plus_;
  QPushButton * xr_minus_;
  QPushButton * yr_plus_;
  QPushButton * yr_minus_;
  QPushButton * zr_plus_;
  QPushButton * zr_minus_;
  QPushButton * next_;
  QPushButton * prev_;
  QTextEdit *trans_step_sz_te_;
  QTextEdit *rot_step_sz_te_;
  QTextEdit *icp_iter_te_;
  QLabel * icp_iter_label;
  QLabel * trans_step_label;
  QLabel * rot_step_label;
  std::vector<int> view_viewport;

  boost::shared_ptr < v4r::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB> > source_;

  std::vector<Eigen::Matrix4f> single_clouds_to_global_;
  std::vector< pcl::PointCloud<PointT>::Ptr, Eigen::aligned_allocator< pcl::PointCloud<PointT>::Ptr> > single_scenes_;
  std::vector<ModelTPtr> sequence_hypotheses_;
  std::vector<Eigen::Matrix4f> hypotheses_poses_;

  int selected_hypothesis_;
  int selected_scene_;
  std::vector<ModelTPtr> loaded_models_;
  int pviz_v1_, pviz_v2_;
  float translation_step_;
  float rotation_step_;
  std::string export_ground_truth_to_;
  std::vector<std::string> scene_names_;
  bool icp_scene_to_model_;
  float inlier_;
  std::vector< std::string> test_sequences_;
  size_t sequence_id_;
  int resolution_mm_;


  int model_viewport_;
};

#endif /* AFFORDANCE_MANAGER_H_ */

