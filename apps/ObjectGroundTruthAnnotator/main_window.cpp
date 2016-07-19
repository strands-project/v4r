/*
 * main_window.cpp
 *
 *  Created on: Jan, 2015
 *      Author: Aitor Aldoma, Thomas Faeulhammer
 */

#ifndef Q_MOC_RUN
#include "main_window.h"
#include <boost/filesystem.hpp>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/icp.h>
#include <vtkRenderWindow.h>
#include <pcl/common/angles.h>
#include <v4r/recognition//voxel_based_correspondence_estimation.h>
#include <v4r/io/filesystem.h>
#include <v4r/io/eigen.h>
#include <iostream>
#include <fstream>
#include <string>
#include <QKeyEvent>

#include <boost/program_options.hpp>
#endif

namespace bf = boost::filesystem;
namespace po = boost::program_options;

void
mouse_callback_scenes (const pcl::visualization::MouseEvent& mouse_event, void* cookie)
{
 MainWindow * main_w = (MainWindow *)cookie;

 if (mouse_event.getType() == pcl::visualization::MouseEvent::MouseDblClick && mouse_event.getButton() == pcl::visualization::MouseEvent::LeftButton)
 {
   int scene_clicked = round(mouse_event.getX () / main_w->getModelXSize());
   cout << "Clicked in scene window :: " << mouse_event.getX () << " , " << mouse_event.getY () << " , " << scene_clicked << endl;
   main_w->selectScene(scene_clicked);
   main_w->selectModel(-1);
   main_w->enablePoseRefinmentButtons(true);
   main_w->updateHighlightedScene(true);
 }
}

void
mouse_callback_models (const pcl::visualization::MouseEvent& mouse_event, void* cookie)
{
 MainWindow * main_w = (MainWindow *)cookie;

 if (mouse_event.getType() == pcl::visualization::MouseEvent::MouseDblClick && mouse_event.getButton() == pcl::visualization::MouseEvent::LeftButton)
 {
   int model_clicked = round(mouse_event.getX () / main_w->getModelXSize());
   cout << "Clicked in model window :: " << mouse_event.getX () << " , " << mouse_event.getY () << " , " << model_clicked << endl;

   main_w->addSelectedModelCloud(model_clicked);
 }
}

void pp_callback (const pcl::visualization::PointPickingEvent& event, void* cookie)
{
  if (event.getPointIndex () == -1)
  {
    return;
  }

  MainWindow * main_w = (MainWindow *)cookie;

  pcl::PointXYZRGB current_point;
  event.getPoint(current_point.x, current_point.y, current_point.z);

  main_w->initialPoseEstimate(current_point.getVector3fMap());
}

void keyboard_callback (const pcl::visualization::KeyboardEvent& event, void* cookie)
{
  MainWindow * main_w = (MainWindow *)cookie;
  float step = 0.01;

  if (event.getKeyCode())
  {
    unsigned char key = event.getKeyCode();

    if(event.keyDown()) {

      if(key == 'z')
      {
        main_w->moveCurrentObject(5,0.1);
      }
      else if(key == 'x')
      {
        main_w->moveCurrentObject(3,0.1);
      }
      else if(key == 'y')
      {
        main_w->moveCurrentObject(4,0.1);
      }
      else if(key == 'v')
      {
        main_w->moveCurrentObject(2,step);
      }
      else if(key == 'b')
      {
        main_w->moveCurrentObject(2,-step);
      }
      else if(key == 'd')
      {
        main_w->remove_selected();
      }
    }
  }
  else {
    if(event.keyDown()) {
      if(event.getKeySym() == "Left") {
        std::cout << "Moving left" << std::endl;
        main_w->moveCurrentObject(0,step);
      } else if (event.getKeySym() == "Right"){
        main_w->moveCurrentObject(0,-step);
      } else if (event.getKeySym() == "Up"){
        main_w->moveCurrentObject(1,step);
      } else if (event.getKeySym() == "Down"){
        main_w->moveCurrentObject(1,-step);
      }
    }
  }
}

void MainWindow::model_list_clicked(const QModelIndex & idx)
{
  std::cout << "model list clicked..." << static_cast<int>(idx.row()) << std::endl;
  enablePoseRefinmentButtons(true);

  pviz_->removePointCloud("highlighted");
  selected_hypothesis_ = static_cast<int>(idx.row());
  selectScene(-1);
  ModelTPtr model = sequence_hypotheses_[ selected_hypothesis_ ];

  pcl::PointCloud<PointT>::ConstPtr model_cloud = model->getAssembled( resolution_mm_ );
  pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>(*model_cloud));
  pcl::transformPointCloud(*model_cloud, *cloud, hypotheses_poses_[ selected_hypothesis_ ]);

  pcl::visualization::PointCloudColorHandlerCustom<PointT> scene_handler(cloud, 0, 255, 0);
  pviz_->addPointCloud(cloud, scene_handler, "highlighted");
  pviz_->spinOnce(100, true);

}

void MainWindow::fillScene()
{
    pviz_->removePointCloud("merged_cloud");
    pviz_->removeAllPointClouds(pviz_v1_);
    scene_merged_cloud_.reset(new pcl::PointCloud<PointT>);

    for (size_t i = 0; i < single_scenes_.size (); i++)
    {
        std::stringstream cloud_name;
        cloud_name << "view_" << i;
//        pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
//        pcl::transformPointCloud(*single_scenes_[i], *cloud, single_clouds_to_global_[i]);
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> scene_handler(single_scenes_[i]);
        pviz_->addPointCloud(single_scenes_[i], scene_handler, cloud_name.str(), pviz_v1_);
    }

    pviz_->spinOnce(100, true);
}

void MainWindow::fillModels()
{
  std::vector<ModelTPtr> models = source_->getModels();
  size_t kk = (models.size ()) + 1;
  double x_step = 1.0 / (float)kk;
  model_clouds_.resize(models.size ());

  pviz_models_->removeAllPointClouds();

  for (size_t i = 0; i < models.size (); i++)
  {
    std::stringstream model_name;
    model_name << "poly_" << i;

    model_clouds_[i].reset(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::ConstPtr model_cloud = models.at(i)->getAssembled( resolution_mm_ );
    pviz_models_->createViewPort (i * x_step, 0, (i + 1) * x_step, 200, model_viewport_);

    //create scale transform...
    pviz_models_->addPointCloud(model_cloud, model_name.str(), model_viewport_);
    loaded_models_.push_back(models.at(i));
  }

  pviz_models_->spinOnce(100, true);
}

void MainWindow::fillHypotheses()
{
    QStringList list;
    pviz_->removeAllPointClouds(pviz_v2_);

    for (size_t i = 0; i < sequence_hypotheses_.size(); i++)
    {

      std::stringstream model_name;
      model_name << "hypotheses_" << i;

      pcl::PointCloud<PointT>::ConstPtr model_cloud = sequence_hypotheses_[i]->getAssembled( resolution_mm_ );
      pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>(*model_cloud));
      pcl::transformPointCloud(*cloud, *cloud, hypotheses_poses_[i]);

      //create scale transform...
      pviz_->addPointCloud(cloud, model_name.str(), pviz_v2_);
      list << QString(sequence_hypotheses_[i]->id_.c_str());
    }

    model_list_->setModel(new QStringListModel(list));
}


void MainWindow::readResultsFile(const std::string &result_file)
{
    std::ifstream in;
    in.open (result_file.c_str (), std::ifstream::in);

    std::string line;
    while (std::getline(in, line))
    {
        std::vector < std::string > strs_2;
        boost::split (strs_2, line, boost::is_any_of ("\t,\b,| "));
        std::string id = strs_2[0];

        Eigen::Matrix4f matrix;
        size_t k = 0;
        for (size_t i = 1; i < 17; i++, k++)
        {
          std::stringstream Str;
          Str << strs_2[i];
          double d;
          Str >> d;
          matrix (k / 4, k % 4) = static_cast<float>(d);
        }

        std::cout << id << std::endl;
        std::cout << matrix << std::endl;

        {
            std::vector < std::string > strs_2;
            boost::split (strs_2, id, boost::is_any_of ("/\\"));
            std::cout << strs_2[strs_2.size () - 1] << std::endl;
            ModelTPtr model;
            bool found = source_->getModelById (strs_2[strs_2.size () - 1], model);
            if(found)
                sequence_hypotheses_.push_back(model);
            else
                std::cerr << "Model " << strs_2[strs_2.size () - 1] << " not found!" << std::endl;
        }

        hypotheses_poses_.push_back(matrix);
    }

    in.close ();
}

void MainWindow::fillViews()
{
  scene_merged_cloud_.reset(new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr merged_cloud (new pcl::PointCloud<PointT>);

  double x_step = 1.0 / (float) (single_scenes_.size ()+1);

  view_viewport.resize (single_scenes_.size());
  std::fill(view_viewport.begin(), view_viewport.end(), 0);

  for (size_t i = 0; i < single_scenes_.size (); i++)
  {
      std::stringstream view_name;
      view_name << "view_" << i;

      pviz_scenes_->createViewPort (i * x_step, 0, (i + 1) * x_step, 200, view_viewport[i]);
      pviz_scenes_->addPointCloud(single_scenes_[i], view_name.str(), view_viewport[i]);
  }

  pviz_->spinOnce(100, true);
}

void MainWindow::lock_with_icp()
{
    pcl::PointCloud<PointT>::Ptr merged_cloud (new pcl::PointCloud<PointT>);
    for (size_t i = 0; i < single_scenes_.size (); i++)
    {
        if(selected_scene_!=i)
        {
            pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
            pcl::transformPointCloud(*single_scenes_[i], *cloud, v4r::RotTrans2Mat4f(single_scenes_[i]->sensor_orientation_, single_scenes_[i]->sensor_origin_));

            cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
            cloud->sensor_origin_ = zero_origin;
            *merged_cloud += *cloud;
        }
    }
    v4r::voxelGridWithOctree(merged_cloud, *scene_merged_cloud_, 0.003f);

    QString icp_iter_str = icp_iter_te_->toPlainText();
    bool *okay = new bool();
    int icp_iter = icp_iter_str.toInt(okay);
    if(!*okay)
    {
        std::cerr << "Could not convert icp iterations text field to a number. Is it a number? Setting it to the default value " << 20 << std::endl;
        icp_iter = 20;
    }

    pcl::PointCloud<pcl::PointXYZRGB> output;
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> reg;
    reg.setMaximumIterations (icp_iter);
    reg.setEuclideanFitnessEpsilon (1e-12);
    reg.setTransformationEpsilon (0.0001f * 0.0001f);
    reg.setMaxCorrespondenceDistance(0.005f);

    std::cout << "Started ICP... " << std::endl;

    if(selected_hypothesis_>=0)
    {
        ModelTPtr model = sequence_hypotheses_[selected_hypothesis_];
        boost::shared_ptr < distance_field::PropagationDistanceField<pcl::PointXYZRGB> > dt;
//        model->getVGDT (dt);

        pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr model_cloud;
        model_cloud = model->getAssembled( resolution_mm_ );
//        dt->getInputCloud (model_cloud);

        pcl::PointCloud<PointT>::Ptr model_cloud_transformed(new pcl::PointCloud<PointT>(*model_cloud));
        pcl::transformPointCloud(*model_cloud, *model_cloud_transformed, hypotheses_poses_[selected_hypothesis_]);

        PointT minPoint, maxPoint;
        pcl::getMinMax3D(*model_cloud_transformed, minPoint, maxPoint);
        float max_corr_distance = 0.1f;
        minPoint.x -= max_corr_distance;
        minPoint.y -= max_corr_distance;
        minPoint.z -= max_corr_distance;
        maxPoint.x += max_corr_distance;
        maxPoint.y += max_corr_distance;
        maxPoint.z += max_corr_distance;

        pcl::CropBox<PointT> cropFilter;
        cropFilter.setInputCloud (scene_merged_cloud_);
        cropFilter.setMin(minPoint.getVector4fMap());
        cropFilter.setMax(maxPoint.getVector4fMap());
        pcl::PointCloud<PointT>::Ptr scene_cropped_to_model (new pcl::PointCloud<PointT> ());
        cropFilter.filter (*scene_cropped_to_model);
        if(icp_scene_to_model_)
        {
            reg.setInputTarget (model_cloud_transformed); //model
            reg.setInputSource (scene_merged_cloud_); //scene
        }
        else
        {
            reg.setInputTarget (scene_merged_cloud_); //scene
            reg.setInputSource (model_cloud_transformed); //scene
        }
        reg.align (output);

        hypotheses_poses_[selected_hypothesis_] = reg.getFinalTransformation() * hypotheses_poses_[selected_hypothesis_];
        updateSelectedHypothesis();
    }
    else if (selected_scene_>0)
    {
        pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
        pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
        Eigen::Matrix4f cloud_to_global = v4r::RotTrans2Mat4f(single_scenes_[ selected_scene_ ]->sensor_orientation_, single_scenes_[ selected_scene_ ]->sensor_origin_);
        pcl::transformPointCloud(*single_scenes_[ selected_scene_ ], *cloud, cloud_to_global);
        cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
        cloud->sensor_origin_ = zero_origin;

        pcl::PassThrough<PointT> pass;
        pass.setFilterLimits (0.f, 3);
        pass.setFilterFieldName ("z");
        pass.setInputCloud (cloud);
        pass.setKeepOrganized (false);
        pass.filter (*cloud_filtered);
        reg.setInputSource (cloud_filtered);

        reg.setInputTarget(scene_merged_cloud_);
        reg.align (output);
        v4r::setCloudPose(reg.getFinalTransformation() * cloud_to_global , *single_scenes_[selected_scene_]);
        updateHighlightedScene();
    }
    std::cout << "ICP finished..." << std::endl;
}


void
MainWindow::enablePoseRefinmentButtons(bool flag)
{
    icp_button_->setEnabled(flag);
    remove_highlighted_hypotheses_->setEnabled(flag);
    x_plus_->setEnabled(flag);
    x_minus_->setEnabled(flag);
    y_plus_->setEnabled(flag);
    y_minus_->setEnabled(flag);
    z_plus_->setEnabled(flag);
    z_minus_->setEnabled(flag);
    xr_plus_->setEnabled(flag);
    xr_minus_->setEnabled(flag);
    yr_plus_->setEnabled(flag);
    yr_minus_->setEnabled(flag);
    zr_plus_->setEnabled(flag);
    zr_minus_->setEnabled(flag);
    trans_step_sz_te_->setEnabled(flag);
    rot_step_sz_te_->setEnabled(flag);
    icp_iter_te_->setEnabled(flag);
    icp_iter_label->setEnabled(flag);
    trans_step_label->setEnabled(flag);
    rot_step_label->setEnabled(flag);
}

void
MainWindow::remove_selected()
{
  std::cout << "Removed selected hypothesis " << selected_hypothesis_ << " from the " << sequence_hypotheses_.size() << " hypotheses present in total. "<< std::endl;

  if(selected_hypothesis_ < 0)
      return;

  sequence_hypotheses_.erase(sequence_hypotheses_.begin() + selected_hypothesis_);
  hypotheses_poses_.erase(hypotheses_poses_.begin() + selected_hypothesis_);
  pviz_->removePointCloud("highlighted");

  selected_hypothesis_ = -1;
  fillHypotheses();
  enablePoseRefinmentButtons(false);

  pviz_->spinOnce(100, true);
}

void MainWindow::updateSelectedHypothesis()
{
    std::stringstream model_name;
    model_name << "hypotheses_" << selected_hypothesis_;
    pviz_->removePointCloud("highlighted");

    if(selected_hypothesis_>=0)
    {
        pviz_->removePointCloud(model_name.str(), pviz_v2_);
        pcl::PointCloud<PointT>::ConstPtr model_cloud = sequence_hypotheses_[selected_hypothesis_]->getAssembled( resolution_mm_ );
        pcl::PointCloud<PointT>::Ptr model_cloud_transformed(new pcl::PointCloud<PointT>(*model_cloud));
        pcl::transformPointCloud(*model_cloud, *model_cloud_transformed, hypotheses_poses_[selected_hypothesis_]);

        pviz_->addPointCloud(model_cloud_transformed, model_name.str(), pviz_v2_);

        pcl::visualization::PointCloudColorHandlerCustom<PointT> scene_handler(model_cloud_transformed, 0, 255, 0);
        pviz_->addPointCloud(model_cloud_transformed, scene_handler, "highlighted");
    }

    pviz_->spinOnce(0.1, true);
}

void MainWindow::x_plus()
{
    QString trans_step_str = trans_step_sz_te_->toPlainText();
    bool *okay = new bool();
    float trans_step = trans_step_str.toFloat(okay) / 100.f;
    if(!*okay)
    {
        std::cerr << "Could not convert translation step size. Is it a number? I set it to the default value " << translation_step_ << std::endl;
        trans_step = translation_step_;
    }
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,1>(0,3) = Eigen::Vector3f(trans_step, 0, 0);

    if(selected_hypothesis_>=0)
    {
        hypotheses_poses_[ selected_hypothesis_ ] = hypotheses_poses_[selected_hypothesis_] * m4f;
        updateSelectedHypothesis();
    }
    else if(selected_scene_>=0)
    {
        single_scenes_[ selected_scene_ ]->sensor_origin_[0] +=trans_step;
        updateHighlightedScene();
    }
}

void MainWindow::x_minus()
{
    QString trans_step_str = trans_step_sz_te_->toPlainText();
    bool *okay = new bool();
    float trans_step = trans_step_str.toFloat(okay) / 100.f;
    if(!*okay)
    {
        std::cerr << "Could not convert translation step size. Is it a number? I set it to the default value " << translation_step_ << std::endl;
        trans_step = translation_step_;
    }
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,1>(0,3) = Eigen::Vector3f(-trans_step, 0, 0);

    if(selected_hypothesis_>=0)
    {
        hypotheses_poses_[ selected_hypothesis_ ] = hypotheses_poses_[selected_hypothesis_] * m4f;
        updateSelectedHypothesis();
    }
    else if(selected_scene_>=0)
    {
        single_scenes_[ selected_scene_ ]->sensor_origin_[0] -=trans_step;
        updateHighlightedScene();
    }
}

void MainWindow::y_plus()
{
    QString trans_step_str = trans_step_sz_te_->toPlainText();
    bool *okay = new bool();
    float trans_step = trans_step_str.toFloat(okay) / 100.f;
    if(!*okay)
    {
        std::cerr << "Could not convert translation step size. Is it a number? I set it to the default value " << translation_step_ << std::endl;
        trans_step = translation_step_;
    }
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,1>(0,3) = Eigen::Vector3f(0, trans_step, 0);

    if(selected_hypothesis_>=0)
    {
        hypotheses_poses_[ selected_hypothesis_ ] = hypotheses_poses_[selected_hypothesis_] * m4f;
        updateSelectedHypothesis();
    }
    else if(selected_scene_>=0)
    {
        single_clouds_to_global_[ selected_scene_ ] = single_clouds_to_global_[ selected_scene_ ] * m4f;
        updateHighlightedScene();
    }
}

void MainWindow::y_minus()
{
    QString trans_step_str = trans_step_sz_te_->toPlainText();
    bool *okay = new bool();
    float trans_step = trans_step_str.toFloat(okay) / 100.f;
    if(!*okay)
    {
        std::cerr << "Could not convert translation step size. Is it a number? I set it to the default value " << translation_step_ << std::endl;
        trans_step = translation_step_;
    }
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,1>(0,3) = Eigen::Vector3f(0, -trans_step, 0);

    if(selected_hypothesis_>=0)
    {
        hypotheses_poses_[ selected_hypothesis_ ] = hypotheses_poses_[selected_hypothesis_] * m4f;
        updateSelectedHypothesis();
    }
    else if(selected_scene_>=0)
    {
        single_clouds_to_global_[ selected_scene_ ] = single_clouds_to_global_[ selected_scene_ ] * m4f;
        updateHighlightedScene();
    }
}


void MainWindow::z_plus()
{
    QString trans_step_str = trans_step_sz_te_->toPlainText();
    bool *okay = new bool();
    float trans_step = trans_step_str.toFloat(okay) / 100.f;
    if(!*okay)
    {
        std::cerr << "Could not convert translation step size. Is it a number? I set it to the default value " << translation_step_ << std::endl;
        trans_step = translation_step_;
    }
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,1>(0,3) = Eigen::Vector3f(0, 0, trans_step);

    if(selected_hypothesis_>=0)
    {
        hypotheses_poses_[ selected_hypothesis_ ] = hypotheses_poses_[selected_hypothesis_] * m4f;
        updateSelectedHypothesis();
    }
    else if(selected_scene_>=0)
    {
        single_clouds_to_global_[ selected_scene_ ] = single_clouds_to_global_[ selected_scene_ ] * m4f;
        updateHighlightedScene();
    }
}

void MainWindow::z_minus()
{
    QString trans_step_str = trans_step_sz_te_->toPlainText();
    bool *okay = new bool();
    float trans_step = trans_step_str.toFloat(okay) / 100.f;
    if(!*okay)
    {
        std::cerr << "Could not convert translation step size. Is it a number? I set it to the default value " << translation_step_ << std::endl;
        trans_step = translation_step_;
    }
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,1>(0,3) = Eigen::Vector3f(0, 0, -trans_step);

    if(selected_hypothesis_>=0)
    {
        hypotheses_poses_[ selected_hypothesis_ ] = hypotheses_poses_[selected_hypothesis_] * m4f;
        updateSelectedHypothesis();
    }
    else if(selected_scene_>=0)
    {
        single_scenes_[ selected_scene_ ]->sensor_origin_[3] -= trans_step;
        updateHighlightedScene();
    }
}

void MainWindow::xr_plus()
{
    QString rot_step_str = rot_step_sz_te_->toPlainText();
    bool *okay = new bool();
    float rot_step = rot_step_str.toFloat(okay);
    if(!*okay)
    {
         std::cerr << "Could not convert rotation step size. Is it a number? I set it to the default value " << translation_step_ << std::endl;
         rot_step = rotation_step_;
    }
    rot_step = pcl::deg2rad( rot_step);
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(rot_step, Eigen::Vector3f::UnitX());
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,3>(0,0) = m;

    if(selected_hypothesis_>=0)
    {
        hypotheses_poses_[ selected_hypothesis_ ] = hypotheses_poses_[selected_hypothesis_] * m4f;
        updateSelectedHypothesis();
    }
    else if(selected_scene_>=0)
    {
        single_clouds_to_global_[ selected_scene_ ] = single_clouds_to_global_[ selected_scene_ ] * m4f;
        updateHighlightedScene();
    }
}

void MainWindow::xr_minus()
{
    QString rot_step_str = rot_step_sz_te_->toPlainText();
    bool *okay = new bool();
    float rot_step = rot_step_str.toFloat(okay);
    if(!*okay)
    {
         std::cerr << "Could not convert rotation step size. Is it a number? I set it to the default value " << translation_step_ << std::endl;
         rot_step = rotation_step_;
    }
    rot_step = pcl::deg2rad( rot_step);
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(-rot_step, Eigen::Vector3f::UnitX());
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,3>(0,0) = m;

    if(selected_hypothesis_>=0)
    {
        hypotheses_poses_[ selected_hypothesis_ ] = hypotheses_poses_[selected_hypothesis_] * m4f;
        updateSelectedHypothesis();
    }
    else if(selected_scene_>=0)
    {
        single_clouds_to_global_[ selected_scene_ ] = single_clouds_to_global_[ selected_scene_ ] * m4f;
        updateHighlightedScene();
    }
}

void MainWindow::yr_plus()
{
    QString rot_step_str = rot_step_sz_te_->toPlainText();
    bool *okay = new bool();
    float rot_step = rot_step_str.toFloat(okay);
    if(!*okay)
    {
         std::cerr << "Could not convert rotation step size. Is it a number? I set it to the default value " << translation_step_ << std::endl;
         rot_step = rotation_step_;
    }
    rot_step = pcl::deg2rad( rot_step);
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(rot_step, Eigen::Vector3f::UnitY());
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,3>(0,0) = m;

    if(selected_hypothesis_>=0)
    {
        hypotheses_poses_[ selected_hypothesis_ ] = hypotheses_poses_[selected_hypothesis_] * m4f;
        updateSelectedHypothesis();
    }
    else if(selected_scene_>=0)
    {
        single_clouds_to_global_[ selected_scene_ ] = single_clouds_to_global_[ selected_scene_ ] * m4f;
        updateHighlightedScene();
    }
}

void MainWindow::yr_minus()
{
    QString rot_step_str = rot_step_sz_te_->toPlainText();
    bool *okay = new bool();
    float rot_step = rot_step_str.toFloat(okay);
    if(!*okay)
    {
         std::cerr << "Could not convert rotation step size. Is it a number? I set it to the default value " << translation_step_ << std::endl;
         rot_step = rotation_step_;
    }
    rot_step = pcl::deg2rad( rot_step);
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(-rot_step, Eigen::Vector3f::UnitY());
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,3>(0,0) = m;

    if(selected_hypothesis_>=0)
    {
        hypotheses_poses_[ selected_hypothesis_ ] = hypotheses_poses_[selected_hypothesis_] * m4f;
        updateSelectedHypothesis();
    }
    else if(selected_scene_>=0)
    {
        single_clouds_to_global_[ selected_scene_ ] = single_clouds_to_global_[ selected_scene_ ] * m4f;
        updateHighlightedScene();
    }
}


void MainWindow::zr_plus()
{
    QString rot_step_str = rot_step_sz_te_->toPlainText();
    bool *okay = new bool();
    float rot_step = rot_step_str.toFloat(okay);
    if(!*okay)
    {
         std::cerr << "Could not convert rotation step size. Is it a number? I set it to the default value " << translation_step_ << std::endl;
         rot_step = rotation_step_;
    }
    rot_step = pcl::deg2rad( rot_step);
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(rot_step, Eigen::Vector3f::UnitZ());
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,3>(0,0) = m;

    if(selected_hypothesis_>=0)
    {
        hypotheses_poses_[ selected_hypothesis_ ] = hypotheses_poses_[selected_hypothesis_] * m4f;
        updateSelectedHypothesis();
    }
    else if(selected_scene_>=0)
    {
        single_clouds_to_global_[ selected_scene_ ] = single_clouds_to_global_[ selected_scene_ ] * m4f;
        updateHighlightedScene();
    }
}

void MainWindow::zr_minus()
{
    QString rot_step_str = rot_step_sz_te_->toPlainText();
    bool *okay = new bool();
    float rot_step = rot_step_str.toFloat(okay);
    if(!*okay)
    {
         std::cerr << "Could not convert rotation step size. Is it a number? I set it to the default value " << translation_step_ << std::endl;
         rot_step = rotation_step_;
    }
    rot_step = pcl::deg2rad( rot_step);
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(-rot_step, Eigen::Vector3f::UnitZ());
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,3>(0,0) = m;

    if(selected_hypothesis_>=0)
    {
        hypotheses_poses_[ selected_hypothesis_ ] = hypotheses_poses_[selected_hypothesis_] * m4f;
        updateSelectedHypothesis();
    }
    else if(selected_scene_>=0)
    {
        single_clouds_to_global_[ selected_scene_ ] = single_clouds_to_global_[ selected_scene_ ] * m4f;
        updateHighlightedScene();
    }
}

void MainWindow::next()
{
    clear();
    const std::string sequence_path = base_path_ + "/" + test_sequences_[++sequence_id_];
    std::vector<std::string> files;
    v4r::io::getFilesInDirectory( sequence_path, files, "", ".*.pcd", true);   // get scenes
    std::cout << "Number of scenes in directory is:" << files.size () << std::endl;
    for (size_t i = 0; i < files.size (); i++)
    {
      scene_names_.push_back(files[i]);
      const std::string file = sequence_path + "/" + files[i];
      std::cout << files[i] << std::endl;

      //read cloud
      pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
      pcl::io::loadPCDFile(file, *cloud);
      single_scenes_.push_back(cloud);
    }

    std::stringstream filestr;
    filestr << sequence_path << "/results_3d.txt";
    readResultsFile(filestr.str());

    fillScene ();
    fillHypotheses ();
    fillViews ();
}
void MainWindow::prev()
{
    clear();
}

void MainWindow::save_model()
{
    std::string gt_or_ouput_dir = export_ground_truth_to_ + "/" + test_sequences_[sequence_id_];
    v4r::io::createDirIfNotExist(gt_or_ouput_dir);

        for (size_t i=0; i < single_scenes_.size(); i++)
        {
            std::string scene = scene_names_[i];
            boost::replace_all (scene, ".pcd", "");
            const Eigen::Matrix4f tf = v4r::RotTrans2Mat4f(single_scenes_[i]->sensor_orientation_, single_scenes_[i]->sensor_origin_);

//            std::stringstream camera_pose_out_fn_ss;
//            camera_pose_out_fn_ss << gt_or_ouput_dir << "/transformation_ " << scene << ".txt";
//            v4r::io::writeMatrixToFile(camera_pose_out_fn_ss.str(), single_clouds_to_global_[i]);

            // for occlusion computation------
            pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloudXYZ(new pcl::PointCloud<pcl::PointXYZ>);
            scene_cloudXYZ->points.resize(single_scenes_[i]->points.size());
            for(size_t pt_id = 0; pt_id < single_scenes_[i]->points.size(); pt_id++)
            {
                scene_cloudXYZ->points[pt_id].x = single_scenes_[i]->points[pt_id].x;
                scene_cloudXYZ->points[pt_id].y = single_scenes_[i]->points[pt_id].y;
                scene_cloudXYZ->points[pt_id].z = single_scenes_[i]->points[pt_id].z;
            }
            pcl::transformPointCloud(*scene_cloudXYZ,*scene_cloudXYZ, tf);

            pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree (0.001);
            octree.setInputCloud (scene_cloudXYZ);
            octree.addPointsFromInputCloud ();

            std::vector<int> pointIdxNKNSearch;
            std::vector<float> pointNKNSquaredDistance;
            // -------------------------------

            std::map<std::string, int> id_count;
            std::map<std::string, int>::iterator id_c_it;

            for (size_t k=0; k < sequence_hypotheses_.size(); k++)
            {
                ModelTPtr model = sequence_hypotheses_[k];
                std::string model_id_replaced = model->id_;
                boost::replace_all (model_id_replaced, ".pcd", "");

                id_c_it = id_count.find(model_id_replaced);
                if(id_c_it == id_count.end())
                {
                    id_count[model_id_replaced] = 0;
                    id_c_it = id_count.find(model_id_replaced);
                }
                else
                {
                    id_c_it->second++;
                }

                const Eigen::Matrix4f cloud_to_global = v4r::RotTrans2Mat4f(single_scenes_[i]->sensor_orientation_, single_scenes_[i]->sensor_origin_);
                Eigen::Matrix4f transform = cloud_to_global.inverse() * hypotheses_poses_[k];

                std::stringstream pose_file_ss;
                pose_file_ss << gt_or_ouput_dir << "/" << scene << "_" << model_id_replaced << "_" << id_c_it->second << ".txt";
                std::cout << pose_file_ss.str() << std::endl;
                v4r::io::writeMatrixToFile(pose_file_ss.str(), transform);

                //compute occlusion value
                size_t overlap = 0;
                std::vector<int> indices;
                pcl::PointCloud<PointT>::ConstPtr model_cloud = model->getAssembled( resolution_mm_ );
                pcl::PointCloud<PointT>::Ptr model_aligned(new pcl::PointCloud<PointT>());
                pcl::transformPointCloud(*model_cloud, *model_aligned, hypotheses_poses_[k]);

                for(size_t kk=0; kk < model_aligned->points.size(); kk++)
                {
                    pcl::PointXYZ p;
                    p.getVector3fMap() = model_aligned->points[kk].getVector3fMap();
                    if (octree.nearestKSearch (p, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
                    {
                        float d = sqrt (pointNKNSquaredDistance[0]);
                        if (d < inlier_)
                        {
                            overlap++;
                            indices.push_back(kk);
                        }
                    }
                }

                float occlusion_value = 1.f - overlap / static_cast<float>(model_aligned->points.size());
                std::stringstream occlusion_file;
                occlusion_file << gt_or_ouput_dir << "/" << scene << "_occlusion_" << model_id_replaced << "_" << id_c_it->second << ".txt";
                std::cout << occlusion_file.str() << std::endl;
                v4r::io::writeFloatToFile(occlusion_file.str(), occlusion_value);
            }
        }

        std::stringstream results_3d_file_ss;
        results_3d_file_ss << base_path_ << "/results_3d_edited.txt";
        std::ofstream out;
        out.open(results_3d_file_ss.str().c_str());

        if(!out)
        {
            std::cout << "Can't write edited 3d results into file " << results_3d_file_ss.str() << ". " << std::endl;
        }

        for (size_t i=0; i < sequence_hypotheses_.size(); i++)
        {
            ModelTPtr model = sequence_hypotheses_[i];
            std::string model_id_replaced = model->id_;
            out << model_id_replaced << " ";
            for (size_t row=0; row <4; row++)
            {
                for(size_t col=0; col<4; col++)
                {
                    out << hypotheses_poses_[i](row, col) << " ";
                }
            }
            out << std::endl;
        }
        out.close();
}

MainWindow::MainWindow(int argc, char *argv[])
{
  translation_step_ = 0.01f;
  rotation_step_ = pcl::deg2rad(5.f);

  model_xsize_ = 200;
  pose_xsize_ = 300;

  inlier_ = 0.003f;
  resolution_mm_ = 3;

  zero_origin[0] = zero_origin[1] = zero_origin[2] = zero_origin[3] = 0.f;
  sequence_id_ = 0;

  export_ground_truth_to_ = "/tmp/exported_ground_truth/";
  icp_scene_to_model_ = false;//true;

  po::options_description desc("Ground-Truth Annotation Tool\n======================================\n**Allowed options");
  desc.add_options()
          ("help,h", "produce help message")
          ("models_dir,m", po::value<std::string>(&dir_models_)->required(), "directory containing the model .pcd files")
          ("pcd_file,p", po::value<std::string>(&base_path_)->required(), "Directory with the to be annotated scenes stored as point clouds (.pcd). The camera pose is taken directly from the pcd header fields \"sensor_orientation_\" and \"sensor_origin_\" and it also looks for an initial annotation file \"results_3d.txt\". If this annotation file exists, the program reads each row as \"\\path\\to\\object_model.pcd t11 t12 t13 t14 t21 ...  t34 .. 0 0 0 1\" where t are elements of the 4x4 homogenous transformation matrix bringing the model into the world coordinate system. If the test directory contains subdirectories, each subdirectory is considered as seperate sequence for multiview recognition.")
          ("export_ground_truth_to,o", po::value<std::string>(&export_ground_truth_to_)->default_value("/tmp/exported_ground_truth/"), "Output directory")
          ("icp_scene_to_model", po::value<bool>(&icp_scene_to_model_)->default_value(false), "if true, does pose refinement.")
          ("model_scale", po::value<double>(&model_scale_)->default_value(model_scale_), "model scale")
          ("resolution_mm", po::value<int>(&resolution_mm_)->default_value(resolution_mm_), "Resolution in mm used for ICP and visualization")
;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help"))
  {
      std::cout << desc << std::endl;
      return;
  }

  try
  {
      po::notify(vm);
  }
  catch(std::exception& e)
  {
      std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
      return;
  }

  if(!v4r::io::getFoldersInDirectory( base_path_, "", test_sequences_) )
  {
      std::cerr << "No subfolders in directory " << base_path_ << ". " << std::endl << std::endl << desc << std::endl;
      test_sequences_.push_back("");
  }

  source_.reset (new v4r::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>);
  source_->setPath (dir_models_);
  source_->setLoadViews (false);
  source_->setLoadIntoMemory(false);
  source_->generate ();
//  source_->createVoxelGridAndDistanceTransform (0.005f);

  QApplication app(argc,argv);
  mainWindow_ = new QWidget;
  mainWindow_->resize(1680,1000);

  vtk_widget_ = new QVTKWidget;
  vtk_widget_->resize(800, 256);

  pviz_ = new pcl::visualization::PCLVisualizer("test_viz",true);
  pviz_->createViewPort(0, 0, 1, 0.5, pviz_v1_);
  pviz_->createViewPort(0, 0.5, 1, 1, pviz_v2_);
  vtk_widget_->SetRenderWindow(pviz_->getRenderWindow());
  pviz_->addCoordinateSystem(0.3, pviz_v1_);

  pviz_->registerKeyboardCallback(&keyboard_callback, (void*)(this));
  pviz_->registerPointPickingCallback (pp_callback, (void*)(this));

  vtk_widget_->show();

  QVBoxLayout *main_vl = new QVBoxLayout;
  QVBoxLayout *control_vl = new QVBoxLayout;

  icp_button_ = new QPushButton("Lock with ICP");
  save_model_ = new QPushButton("Export ground truth");
  remove_highlighted_hypotheses_ = new QPushButton("Remove selected hypotheses");
  x_plus_ = new QPushButton("X+");
  x_minus_ = new QPushButton("X-");
  y_plus_ = new QPushButton("Y+");
  y_minus_ = new QPushButton("Y-");
  z_plus_ = new QPushButton("Z+");
  z_minus_ = new QPushButton("Z-");
  xr_plus_ = new QPushButton("XR+");
  xr_minus_ = new QPushButton("XR-");
  yr_plus_ = new QPushButton("YR+");
  yr_minus_ = new QPushButton("YR-");
  zr_plus_ = new QPushButton("ZR+");
  zr_minus_ = new QPushButton("ZR-");
  next_ = new QPushButton("->");
  prev_ = new QPushButton("<-");
  model_list_ = new QListView;

  QObject::connect(icp_button_, SIGNAL(clicked(bool)), this, SLOT(lock_with_icp()));
  QObject::connect(save_model_, SIGNAL(clicked(bool)), this, SLOT(save_model()));
  QObject::connect(remove_highlighted_hypotheses_, SIGNAL(clicked(bool)), this, SLOT(remove_selected()));
  QObject::connect(x_plus_, SIGNAL(clicked(bool)), this, SLOT(x_plus()));
  QObject::connect(x_minus_, SIGNAL(clicked(bool)), this, SLOT(x_minus()));
  QObject::connect(y_plus_, SIGNAL(clicked(bool)), this, SLOT(y_plus()));
  QObject::connect(y_minus_, SIGNAL(clicked(bool)), this, SLOT(y_minus()));
  QObject::connect(z_plus_, SIGNAL(clicked(bool)), this, SLOT(z_plus()));
  QObject::connect(z_minus_, SIGNAL(clicked(bool)), this, SLOT(z_minus()));
  QObject::connect(xr_plus_, SIGNAL(clicked(bool)), this, SLOT(xr_plus()));
  QObject::connect(xr_minus_, SIGNAL(clicked(bool)), this, SLOT(xr_minus()));
  QObject::connect(yr_plus_, SIGNAL(clicked(bool)), this, SLOT(yr_plus()));
  QObject::connect(yr_minus_, SIGNAL(clicked(bool)), this, SLOT(yr_minus()));
  QObject::connect(zr_plus_, SIGNAL(clicked(bool)), this, SLOT(zr_plus()));
  QObject::connect(zr_minus_, SIGNAL(clicked(bool)), this, SLOT(zr_minus()));
  QObject::connect(next_, SIGNAL(clicked(bool)), this, SLOT(next()));
  QObject::connect(prev_, SIGNAL(clicked(bool)), this, SLOT(prev()));
  QObject::connect(model_list_, SIGNAL(clicked(const QModelIndex&)), this, SLOT(model_list_clicked(const QModelIndex&)));

//  Qt::Key key = Qt::Key_Up;
//  qDebug() << QKeySequence(key).toString(); // prints "Up"

  QHBoxLayout *trans_x = new QHBoxLayout;
  trans_x->addWidget(x_minus_);
  trans_x->addWidget(x_plus_);

  QHBoxLayout *trans_y = new QHBoxLayout;
  trans_y->addWidget(y_minus_);
  trans_y->addWidget(y_plus_);

  QHBoxLayout *trans_z = new QHBoxLayout;
  trans_z->addWidget(z_minus_);
  trans_z->addWidget(z_plus_);

  QHBoxLayout *trans_step_box = new QHBoxLayout;
  trans_step_label = new QLabel("step(cm)");
  trans_step_label->setFixedSize(90,30);
  trans_step_sz_te_  = new QTextEdit;
  trans_step_sz_te_->setInputMethodHints(Qt::ImhDigitsOnly);
  trans_step_sz_te_->setText("5");
  trans_step_sz_te_->setFixedSize(50,30);
  trans_step_box->addWidget(trans_step_label);
  trans_step_box->addWidget(trans_step_sz_te_);

  QVBoxLayout * refine_pose_translation = new QVBoxLayout;
  refine_pose_translation->addLayout(trans_x);
  refine_pose_translation->addLayout(trans_y);
  refine_pose_translation->addLayout(trans_z);
  refine_pose_translation->addLayout(trans_step_box);

  QHBoxLayout *rot_x = new QHBoxLayout;
  rot_x->addWidget(xr_minus_);
  rot_x->addWidget(xr_plus_);

  QHBoxLayout *rot_y = new QHBoxLayout;
  rot_y->addWidget(yr_minus_);
  rot_y->addWidget(yr_plus_);

  QHBoxLayout *rot_z = new QHBoxLayout;
  rot_z->addWidget(zr_minus_);
  rot_z->addWidget(zr_plus_);


  QHBoxLayout *rot_step_box = new QHBoxLayout;
  rot_step_label = new QLabel("step (deg)");
  rot_step_label->setFixedSize(90,30);
  rot_step_sz_te_  = new QTextEdit;
  rot_step_sz_te_->setInputMethodHints(Qt::ImhDigitsOnly);
  rot_step_sz_te_->setText("5");
  rot_step_sz_te_->setFixedSize(50,30);
  rot_step_box->addWidget(rot_step_label);
  rot_step_box->addWidget(rot_step_sz_te_);

  QVBoxLayout * refine_pose_rotation = new QVBoxLayout;
  refine_pose_rotation->addLayout(rot_x);
  refine_pose_rotation->addLayout(rot_y);
  refine_pose_rotation->addLayout(rot_z);
  refine_pose_rotation->addLayout(rot_step_box);

  QHBoxLayout *layout_refine_pose_manually = new QHBoxLayout;
  layout_refine_pose_manually->addLayout(refine_pose_translation);
  layout_refine_pose_manually->addLayout(refine_pose_rotation);

  QHBoxLayout *icp_hbox = new QHBoxLayout;
  icp_iter_label = new QLabel("iterations");
  icp_iter_label->setFixedSize(70,30);
  icp_iter_te_  = new QTextEdit;
  icp_iter_te_->setInputMethodHints(Qt::ImhDigitsOnly);
  icp_iter_te_->setText("50");
  icp_iter_te_->setFixedSize(40,30);
  icp_button_->setFixedWidth(150);
  icp_hbox->addWidget(icp_button_);
  icp_hbox->addWidget(icp_iter_label);
  icp_hbox->addWidget(icp_iter_te_);

  QVBoxLayout *layout_refine_pose_total = new QVBoxLayout;
  layout_refine_pose_total->addLayout(layout_refine_pose_manually);
  layout_refine_pose_total->addLayout(icp_hbox);
  layout_refine_pose_total->addWidget(save_model_);
  layout_refine_pose_total->addWidget(next_);
  layout_refine_pose_total->addWidget(prev_);

  QHBoxLayout *layout_vis = new QHBoxLayout;

  pviz_models_ = new pcl::visualization::PCLVisualizer("viz_models",true);
  pviz_models_->registerMouseCallback (mouse_callback_models, (void*)(this));

  pviz_scenes_ = new pcl::visualization::PCLVisualizer("viz_scenes",true);
  pviz_scenes_->registerMouseCallback (mouse_callback_scenes, (void*)(this));

  std::vector<ModelTPtr> models = source_->getModels();
  size_t models_size = models.size();
  size_t scenes_size = 20;//single_scenes_.size ();

  vtk_widget_models_ = new QVTKWidget;
  vtk_widget_models_->resize(models_size*model_xsize_, model_xsize_);
  vtk_widget_models_->SetRenderWindow(pviz_models_->getRenderWindow());

  vtk_widget_scenes_ = new QVTKWidget;
  vtk_widget_scenes_->resize(scenes_size*model_xsize_, model_xsize_);
  vtk_widget_scenes_->SetRenderWindow(pviz_scenes_->getRenderWindow());

  layout_vis->addLayout(control_vl);
  layout_vis->addWidget(vtk_widget_);

  QVBoxLayout * model_list_layout = new QVBoxLayout;
  model_list_->setMaximumWidth(400);

  model_list_layout->setSizeConstraint(QLayout::SetMaximumSize);

  QLabel * label1 = new QLabel("Hypotheses:");

  model_list_layout->addWidget(label1);
  model_list_layout->addWidget(model_list_);
  model_list_layout->addWidget(remove_highlighted_hypotheses_);

  control_vl->addLayout(model_list_layout);
  control_vl->addLayout(layout_refine_pose_total);
  control_vl->setSizeConstraint(QLayout::SetMaximumSize);

  main_vl->addLayout(layout_vis);

  QScrollArea * scroll_models_ = new QScrollArea();
  scroll_models_->setWidget(vtk_widget_models_);
  scroll_models_->setFixedHeight(model_xsize_ + 20);
  scroll_models_->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Fixed);
  main_vl->addWidget(scroll_models_);

  QScrollArea * scroll_scenes_ = new QScrollArea();
  scroll_scenes_->setWidget(vtk_widget_scenes_);
  scroll_scenes_->setFixedHeight(model_xsize_ + 20);
  scroll_scenes_->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Fixed);
  main_vl->addWidget(scroll_scenes_);

  enablePoseRefinmentButtons(false);
  selected_hypothesis_ = -1;
  selected_scene_ = -1;

  mainWindow_->setLayout(main_vl);
  mainWindow_->show();

  const std::string sequence_path = base_path_ + "/" + test_sequences_[sequence_id_];
  std::vector<std::string> files;
  v4r::io::getFilesInDirectory( sequence_path, files, "", ".*.pcd", true);   // get scenes
  std::cout << "Number of scenes in directory is:" << files.size () << std::endl;
  for (size_t i = 0; i < files.size (); i++)
  {
    scene_names_.push_back(files[i]);
    const std::string file = sequence_path + "/" + files[i];
    std::cout << files[i] << std::endl;

    //read cloud
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::io::loadPCDFile(file, *cloud);
    single_scenes_.push_back(cloud);
  }

  std::stringstream filestr;
  filestr << sequence_path << "/results_3d.txt";
  readResultsFile(filestr.str());

  fillModels ();
  fillScene ();
  fillHypotheses ();
  fillViews ();

  app.exec();
}

