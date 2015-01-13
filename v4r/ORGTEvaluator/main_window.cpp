/*
 * main_window.cpp
 *
 *  Created on: Jul 20, 2011
 *      Author: aitor
 */

#include "main_window.h"
#include <boost/filesystem.hpp>
#include "pcl/io/pcd_io.h"
#include <pcl/filters/passthrough.h>
#include "pcl/registration/icp.h"
#include <pcl/console/parse.h>
#include <vtkRenderWindow.h>
#include "pcl/filters/crop_box.h"
#include <pcl/common/angles.h>
#include <v4r/ORFramework/voxel_based_correspondence_estimation.h>
#include <v4r/ORUtils/filesystem_utils.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <iostream>
#include <fstream>

namespace bf = boost::filesystem;

void
mouse_callback_models (const pcl::visualization::MouseEvent& mouse_event, void* cookie)
{
 MainWindow * main_w = (MainWindow *)cookie;

 if (mouse_event.getType() == pcl::visualization::MouseEvent::MouseDblClick && mouse_event.getButton() == pcl::visualization::MouseEvent::LeftButton)
 {
   int model_clicked = round(mouse_event.getX () / main_w->getModelXSize());
   cout << "Clicked in model window :: " << mouse_event.getX () << " , " << mouse_event.getY () << " , " << model_clicked << endl;
   //main_w->computeStablePosesAndDisplay(model_clicked);

   main_w->addSelectedModelCloud(model_clicked);
 }
}

void pp_callback (const pcl::visualization::PointPickingEvent& event, void* cookie)
{

  std::cout << "point picking call back" << std::endl;
  if (event.getPointIndex () == -1)
  {
    return;
  }

  std::cout << event.getPointIndex () << std::endl;
  MainWindow * main_w = (MainWindow *)cookie;

  pcl::PointXYZRGB current_point;
  event.getPoint(current_point.x, current_point.y, current_point.z);
  std::cout << current_point.getVector3fMap() << std::endl;

  main_w->initialPoseEstimate(current_point.getVector3fMap());
}

void
keyboard_callback (const pcl::visualization::KeyboardEvent& event, void* cookie)
{

  std::cout << "keyboard_callback" << std::endl;
  MainWindow * main_w = (MainWindow *)cookie;
  float step = 0.01;

  if (event.getKeyCode()) {
    //cout << "the key \'" << event.getKeyCode() << "\' (" << (int)event.getKeyCode() << ") was";
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

void MainWindow::model_list_clicked(const QModelIndex & idx) {
  std::cout << "model list clicked..." << static_cast<int>(idx.row()) << std::endl;
  enablePoseRefinmentButtons(true);
  /*
}

void MainWindow::model_list_dbClicked(const QModelIndex & idx) {
  std::cout << "model list double clicked..." << static_cast<int>(idx.row()) << std::endl;*/

  //std::vector<ModelTPtr> sequence_hypotheses_;
  //std::vector<Eigen::Matrix4f> hypotheses_poses_;

  //highlight the select hypothesis
  pviz_->removePointCloud("highlighted");
  selected_hypothesis_ = static_cast<int>(idx.row());
  ModelTPtr model = sequence_hypotheses_[ selected_hypothesis_ ];

  pcl::PointCloud<PointT>::ConstPtr model_cloud = model->getAssembled(0.003f);
  pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>(*model_cloud));
  pcl::transformPointCloud(*cloud, *cloud, hypotheses_poses_[ selected_hypothesis_ ]);

  pcl::visualization::PointCloudColorHandlerCustom<PointT> scene_handler(cloud, 0, 255, 0);
  pviz_->addPointCloud(cloud, scene_handler, "highlighted");
  pviz_->spinOnce(0.1, true);
}

void MainWindow::lock_with_icp()
{
    ModelTPtr model = sequence_hypotheses_[selected_hypothesis_];

    boost::shared_ptr < distance_field::PropagationDistanceField<pcl::PointXYZRGB> > dt;
    model->getVGDT (dt);

//    faat_pcl::rec_3d_framework::VoxelBasedCorrespondenceEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Ptr
//            est (
//                new faat_pcl::rec_3d_framework::VoxelBasedCorrespondenceEstimation<
//                pcl::PointXYZRGB,
//                pcl::PointXYZRGB> ());

//    pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGB>::Ptr
//            rej (
//                new pcl::registration::CorrespondenceRejectorSampleConsensus<
//                pcl::PointXYZRGB> ());

    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr model_cloud;
    dt->getInputCloud (model_cloud);

    pcl::PointCloud<PointT>::Ptr model_cloud_transformed(new pcl::PointCloud<PointT>(*model_cloud));
    pcl::transformPointCloud(*model_cloud, *model_cloud_transformed, hypotheses_poses_[selected_hypothesis_]);



//    pcl::visualization::PCLVisualizer viewer;
//    pcl::visualization::PointCloudColorHandlerRGBField<PointT> scene_handler(scene_merged_cloud_);
//    viewer.addPointCloud(scene_merged_cloud_, scene_handler, "scene");
//    pcl::visualization::PointCloudColorHandlerRGBField<PointT> scene_handler2(model_cloud_transformed);
//    viewer.addPointCloud(model_cloud_transformed, scene_handler2, "model");
//    viewer.spin();
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_voxelized_icp_cropped (new pcl::PointCloud<pcl::PointXYZRGB> ());
//    pcl::transformPointCloud(*merged_cloud_for_refinement_start_, *cloud_voxelized_icp_cropped, pose_refined_);

    /*est->setVoxelRepresentationTarget (dt);
    est->setInputSource (cloud_voxelized_icp_cropped);
    est->setInputTarget (cloud);
    est->setMaxCorrespondenceDistance (0.03f);
    est->setMaxColorDistance (-1, -1);

    rej->setInputTarget (cloud);
    rej->setMaximumIterations (1000);
    rej->setInlierThreshold (0.005f);
    rej->setInputSource (cloud_voxelized_icp_cropped);

    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> reg;
    reg.setCorrespondenceEstimation (est);
    reg.addCorrespondenceRejector (rej);
    reg.setInputTarget (cloud); //model
    reg.setInputSource (cloud_voxelized_icp_cropped); //scene
    reg.setMaximumIterations (50);
    reg.setEuclideanFitnessEpsilon (1e-12);
    reg.setTransformationEpsilon (0.0001f * 0.0001f);

    pcl::registration::DefaultConvergenceCriteria<float>::Ptr convergence_criteria;
    convergence_criteria = reg.getConvergeCriteria ();
    convergence_criteria->setAbsoluteMSE (1e-12);
    convergence_criteria->setMaximumIterationsSimilarTransforms (15);
    convergence_criteria->setFailureAfterMaximumIterations (false);

    pcl::PointCloud<pcl::PointXYZRGB> output;
    reg.align (output);*/

    pcl::PointCloud<pcl::PointXYZRGB> output;
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> reg;
    if(icp_scene_to_model_)
    {
        reg.setInputTarget (model_cloud_transformed); //model
//        reg.setInputSource (cloud_voxelized_icp_cropped); //scene
        reg.setInputSource (scene_merged_cloud_); //scene
    }
    else
    {
//        reg.setInputTarget (cloud_voxelized_icp_cropped); //scene
        reg.setInputTarget (scene_merged_cloud_); //scene
        reg.setInputSource (model_cloud_transformed); //scene
    }
    reg.setMaximumIterations (50);
    reg.setEuclideanFitnessEpsilon (1e-12);
    reg.setTransformationEpsilon (0.0001f * 0.0001f);
    reg.setMaxCorrespondenceDistance(0.005f);
    reg.align (output);
    //    pose_refined_ = reg.getFinalTransformation ().inverse() * pose_refined_;
    //    poseRefinementButtonPressed();

    hypotheses_poses_[selected_hypothesis_] = reg.getFinalTransformation() * hypotheses_poses_[selected_hypothesis_];
    updateSelectedHypothesis();
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

  pviz_->spinOnce(0.1, true);
}

void MainWindow::updateSelectedHypothesis()
{
    std::stringstream model_name;
    model_name << "hypotheses_" << selected_hypothesis_;

    pcl::PointCloud<PointT>::ConstPtr model_cloud = sequence_hypotheses_[selected_hypothesis_]->getAssembled(0.003f);
    pcl::PointCloud<PointT>::Ptr model_cloud_transformed(new pcl::PointCloud<PointT>(*model_cloud));
    pcl::transformPointCloud(*model_cloud, *model_cloud_transformed, hypotheses_poses_[selected_hypothesis_]);

    pviz_->removePointCloud(model_name.str(), pviz_v2_);
    pviz_->addPointCloud(model_cloud_transformed, model_name.str(), pviz_v2_);

    pviz_->removePointCloud("highlighted");
    pcl::visualization::PointCloudColorHandlerCustom<PointT> scene_handler(model_cloud_transformed, 0, 255, 0);
    pviz_->addPointCloud(model_cloud_transformed, scene_handler, "highlighted");

    pviz_->spinOnce(0.1, true);
}

void MainWindow::x_plus()
{
    std::cout << "x_plus() " << selected_hypothesis_ << std::endl;
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,1>(0,3) = Eigen::Vector3f(translation_step_, 0, 0);
    hypotheses_poses_[selected_hypothesis_] = hypotheses_poses_[selected_hypothesis_] * m4f;
    updateSelectedHypothesis();
}

void MainWindow::x_minus()
{
    std::cout << "x_minus()" << std::endl;
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,1>(0,3) = Eigen::Vector3f(-translation_step_, 0, 0);
    hypotheses_poses_[selected_hypothesis_] = hypotheses_poses_[selected_hypothesis_] * m4f;
    updateSelectedHypothesis();
}

void MainWindow::y_plus()
{
    std::cout << "y_plus()" << std::endl;
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,1>(0,3) = Eigen::Vector3f(0, translation_step_, 0);
    hypotheses_poses_[selected_hypothesis_] = hypotheses_poses_[selected_hypothesis_] * m4f;
    updateSelectedHypothesis();
}

void MainWindow::y_minus()
{
    std::cout << "y_minus()" << std::endl;
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,1>(0,3) = Eigen::Vector3f(0, -translation_step_, 0);
    hypotheses_poses_[selected_hypothesis_] = hypotheses_poses_[selected_hypothesis_] * m4f;
    updateSelectedHypothesis();
}


void MainWindow::z_plus()
{
    std::cout << "z_plus()" << std::endl;
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,1>(0,3) = Eigen::Vector3f(0, 0, translation_step_);
    hypotheses_poses_[selected_hypothesis_] = hypotheses_poses_[selected_hypothesis_] * m4f;
    updateSelectedHypothesis();
}

void MainWindow::z_minus()
{
    std::cout << "z_minus()" << std::endl;
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,1>(0,3) = Eigen::Vector3f(0, 0, -translation_step_);
    hypotheses_poses_[selected_hypothesis_] = hypotheses_poses_[selected_hypothesis_] * m4f;
    updateSelectedHypothesis();
}

void MainWindow::xr_plus()
{
    std::cout << "xr_plus()" << std::endl;
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(rotation_step_, Eigen::Vector3f::UnitX());
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,3>(0,0) = m;
    hypotheses_poses_[selected_hypothesis_] = hypotheses_poses_[selected_hypothesis_] * m4f;
    updateSelectedHypothesis();
}

void MainWindow::xr_minus()
{
    std::cout << "xr_minus()" << std::endl;
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(-rotation_step_, Eigen::Vector3f::UnitX());
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,3>(0,0) = m;
    hypotheses_poses_[selected_hypothesis_] = hypotheses_poses_[selected_hypothesis_] * m4f;
    updateSelectedHypothesis();
}

void MainWindow::yr_plus()
{
    std::cout << "yr_plus()" << std::endl;
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(rotation_step_, Eigen::Vector3f::UnitY());
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,3>(0,0) = m;
    hypotheses_poses_[selected_hypothesis_] = hypotheses_poses_[selected_hypothesis_] * m4f;
    updateSelectedHypothesis();
}

void MainWindow::yr_minus()
{
    std::cout << "yr_minus()" << std::endl;
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(-rotation_step_, Eigen::Vector3f::UnitY());
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,3>(0,0) = m;
    hypotheses_poses_[selected_hypothesis_] = hypotheses_poses_[selected_hypothesis_] * m4f;
    updateSelectedHypothesis();
}


void MainWindow::zr_plus()
{
    std::cout << "zr_plus()" << std::endl;
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(rotation_step_, Eigen::Vector3f::UnitZ());
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,3>(0,0) = m;
    hypotheses_poses_[selected_hypothesis_] = hypotheses_poses_[selected_hypothesis_] * m4f;
    updateSelectedHypothesis();
}

void MainWindow::zr_minus()
{
    std::cout << "zr_minus()" << std::endl;
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(-rotation_step_, Eigen::Vector3f::UnitZ());
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,3>(0,0) = m;
    hypotheses_poses_[selected_hypothesis_] = hypotheses_poses_[selected_hypothesis_] * m4f;
    updateSelectedHypothesis();
}

void MainWindow::save_model()
{

    std::string gt_or_ouput_dir = export_ground_truth_to_;
    if(gt_or_ouput_dir.compare("") != 0)
    {
        bf::path or_path = gt_or_ouput_dir;
        if(!bf::exists(or_path))
        {
            bf::create_directory(or_path);
        }

        for (size_t i=0; i < single_scenes_.size(); i++)
        {
            std::stringstream fn_str;
            fn_str << gt_or_ouput_dir << "/";
            std::map<std::string, int> id_count;
            std::map<std::string, int>::iterator id_c_it;

            std::string scene = scene_names_[i];
            boost::replace_all (scene, "cloud_", "");
            boost::replace_all (scene, ".pcd", "");

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

                Eigen::Matrix4f transform = single_clouds_to_global_[i].inverse() * hypotheses_poses_[k];

                std::stringstream pose_file;
                pose_file << fn_str.str() << scene << "_" << model_id_replaced << "_" << id_c_it->second << ".txt";
                std::cout << pose_file.str() << std::endl;
                faat_pcl::utils::writeMatrixToFile(pose_file.str(), transform);
            }
        }
    }

        std::stringstream results_3d_file_ss;
        results_3d_file_ss << pcd_file_ << "/results_3d_edited.txt";
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

  export_ground_truth_to_ = "";
  icp_scene_to_model_ = false;//true;

  pcl::console::parse_argument (argc, argv, "-models_dir", dir_models_);
  pcl::console::parse_argument (argc, argv, "-model_scale", model_scale_);
  pcl::console::parse_argument (argc, argv, "-pcd_file", pcd_file_);
  pcl::console::parse_argument (argc, argv, "-export_ground_truth_to_", export_ground_truth_to_);
  pcl::console::parse_argument (argc, argv, "-icp_scene_to_model", icp_scene_to_model_);

  std::vector<std::string> files;
  std::string start = "";
  std::string ext = std::string ("pcd");
  bf::path dir = pcd_file_;
  faat_pcl::utils::getFilesInDirectory( dir, start, files, ext);   // get models
  std::cout << "Number of scenes in directory is:" << files.size () << std::endl;
  for (size_t i = 0; i < files.size (); i++)
  {

    scene_names_.push_back(files[i]);
    std::cout << files[i] << std::endl;
    std::stringstream filestr;
    filestr << pcd_file_ << "/" << files[i];
    std::string file = filestr.str ();

    //read cloud
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::io::loadPCDFile(file, *cloud);
    single_scenes_.push_back(cloud);

    //read transform
    std::string trans (file);
    boost::replace_all (trans, "cloud", "transformation");
    boost::replace_all (trans, ".pcd", ".txt");
    Eigen::Matrix4f transform;
    faat_pcl::utils::readMatrixFromFile(trans, transform);

    single_clouds_to_global_.push_back(transform);
  }

  source_.reset (new faat_pcl::rec_3d_framework::ModelOnlySource<pcl::PointXYZRGBNormal, pcl::PointXYZRGB>);
  source_->setPath (dir_models_);
  source_->setLoadViews (false);
  source_->setLoadIntoMemory(false);
  std::string test = "irrelevant";
  source_->generate (test);
  source_->createVoxelGridAndDistanceTransform (0.005f);

  std::stringstream filestr;
  filestr << pcd_file_ << "/results_3d.txt";
  readResultsFile(filestr.str());

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
  //pviz_->addPointCloud<pcl::PointXYZ>(curr_cloud_,"current_cloud");
  //pviz_->resetCamera();

  pviz_->registerKeyboardCallback(&keyboard_callback, (void*)(this));
  pviz_->registerPointPickingCallback (pp_callback, (void*)(this));

  vtk_widget_->show();

  QVBoxLayout *main_vl = new QVBoxLayout;
  QVBoxLayout *control_vl = new QVBoxLayout;

  icp_button_ = new QPushButton("Lock with ICP");
  save_model_ = new QPushButton("Export ground truth");
//  refine_pose_ = new QCheckBox("Refine pose");
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
  model_list_ = new QListView;

  QObject::connect(icp_button_, SIGNAL(clicked(bool)), this, SLOT(lock_with_icp()));
  QObject::connect(save_model_, SIGNAL(clicked(bool)), this, SLOT(save_model()));
//  QObject::connect(refine_pose_, SIGNAL(stateChanged(int)), this, SLOT(refine_pose_changed(int)));
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
//  QObject::connect(model_list_, SIGNAL(doubleClicked(const QModelIndex&)), this, SLOT(model_list_dbClicked(const QModelIndex&)));
  QObject::connect(model_list_, SIGNAL(clicked(const QModelIndex&)), this, SLOT(model_list_clicked(const QModelIndex&)));

  QHBoxLayout *trans_x = new QHBoxLayout;
  trans_x->addWidget(x_minus_);
  trans_x->addWidget(x_plus_);

  QHBoxLayout *trans_y = new QHBoxLayout;
  trans_y->addWidget(y_minus_);
  trans_y->addWidget(y_plus_);

  QHBoxLayout *trans_z = new QHBoxLayout;
  trans_z->addWidget(z_minus_);
  trans_z->addWidget(z_plus_);

  QVBoxLayout * refine_pose_translation = new QVBoxLayout;
  refine_pose_translation->addLayout(trans_x);
  refine_pose_translation->addLayout(trans_y);
  refine_pose_translation->addLayout(trans_z);

  QHBoxLayout *rot_x = new QHBoxLayout;
  rot_x->addWidget(xr_minus_);
  rot_x->addWidget(xr_plus_);

  QHBoxLayout *rot_y = new QHBoxLayout;
  rot_y->addWidget(yr_minus_);
  rot_y->addWidget(yr_plus_);

  QHBoxLayout *rot_z = new QHBoxLayout;
  rot_z->addWidget(zr_minus_);
  rot_z->addWidget(zr_plus_);

  QVBoxLayout * refine_pose_rotation = new QVBoxLayout;
  refine_pose_rotation->addLayout(rot_x);
  refine_pose_rotation->addLayout(rot_y);
  refine_pose_rotation->addLayout(rot_z);

  //QFrame * qframe_refine_pose = new QFrame("pose refinement");

  QHBoxLayout *layout_refine_pose = new QHBoxLayout;
  layout_refine_pose->addLayout(refine_pose_translation);
  layout_refine_pose->addLayout(refine_pose_rotation);
  //qframe_refine_pose->

  QHBoxLayout *layout_vis = new QHBoxLayout;

  pviz_models_ = new pcl::visualization::PCLVisualizer("viz_models",true);
  pviz_models_->registerMouseCallback (mouse_callback_models, (void*)(this));

  boost::shared_ptr<std::vector<ModelTPtr> > models = source_->getModels();
  int models_size = models->size();

  vtk_widget_models_ = new QVTKWidget;
  vtk_widget_models_->resize(models_size*model_xsize_, model_xsize_);
  vtk_widget_models_->SetRenderWindow(pviz_models_->getRenderWindow());

  layout_vis->addLayout(control_vl);
  layout_vis->addWidget(vtk_widget_);

  QVBoxLayout * model_list_layout = new QVBoxLayout;
  model_list_->setMaximumWidth(500);

  model_list_layout->setSizeConstraint(QLayout::SetMaximumSize);

  QLabel * label1 = new QLabel("Hypotheses:");

  model_list_layout->addWidget(label1);
  model_list_layout->addWidget(model_list_);
  model_list_layout->addWidget(remove_highlighted_hypotheses_);

  //control_vl->addLayout(scene_list_layout);
  control_vl->addLayout(model_list_layout);
//  control_vl->addWidget(refine_pose_);
  control_vl->addLayout(layout_refine_pose);
  control_vl->addWidget(icp_button_);
  control_vl->addWidget(save_model_);
  control_vl->setSizeConstraint(QLayout::SetMaximumSize);

  //layout_vis->addLayout();

  main_vl->addLayout(layout_vis);

  QScrollArea * scroll_models_ = new QScrollArea();
  scroll_models_->setWidget(vtk_widget_models_);
  scroll_models_->setFixedHeight(model_xsize_ + 20);
  scroll_models_->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Fixed);
  main_vl->addWidget(scroll_models_);
  fillModels ();
  fillScene ();
  fillHypotheses ();

  enablePoseRefinmentButtons(false);
  selected_hypothesis_ = -1;

  //pviz_->registerKeyboardCallback(&keyboard_callback, (void*)(this));

  //pviz_models_->resetCamera();
  mainWindow_->setLayout(main_vl);
  mainWindow_->show();
  app.exec();
}
