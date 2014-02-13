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
#include <faat_pcl/3d_rec_framework/registration/voxel_based_correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>

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
  std::cout << "model list clicqued..." << static_cast<int>(idx.row()) << std::endl;

  //std::vector<ModelTPtr> sequence_hypotheses_;
  //std::vector<Eigen::Matrix4f> hypotheses_poses_;

  //highlight the select hypothesis
  pviz_->removePointCloud("highlighted");
  int i = static_cast<int>(idx.row());
  ModelTPtr model = sequence_hypotheses_[i];

  pcl::PointCloud<PointT>::ConstPtr model_cloud = model->getAssembled(0.003f);
  pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>(*model_cloud));
  pcl::transformPointCloud(*cloud, *cloud, hypotheses_poses_[i]);

  pcl::visualization::PointCloudColorHandlerCustom<PointT> scene_handler(cloud, 0, 255, 0);
  pviz_->addPointCloud(cloud, scene_handler, "highlighted");
  pviz_->spinOnce(0.1, true);

  selected_hypothesis_ = i;
}

void MainWindow::lock_with_icp()
{

    ModelTPtr model = sequence_hypotheses_[selected_hypothesis_];

    boost::shared_ptr < distance_field::PropagationDistanceField<pcl::PointXYZRGB> > dt;
    model->getVGDT (dt);

    faat_pcl::rec_3d_framework::VoxelBasedCorrespondenceEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Ptr
            est (
                new faat_pcl::rec_3d_framework::VoxelBasedCorrespondenceEstimation<
                pcl::PointXYZRGB,
                pcl::PointXYZRGB> ());

    pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGB>::Ptr
            rej (
                new pcl::registration::CorrespondenceRejectorSampleConsensus<
                pcl::PointXYZRGB> ());

    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud;
    dt->getInputCloud (cloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_voxelized_icp_cropped (new pcl::PointCloud<pcl::PointXYZRGB> ());
    pcl::transformPointCloud(*merged_cloud_for_refinement_start_, *cloud_voxelized_icp_cropped, pose_refined_);

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
    if(icp_scene_to_model_)
    {
        pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> reg;
        reg.setInputTarget (cloud); //model
        reg.setInputSource (cloud_voxelized_icp_cropped); //scene
        reg.setMaximumIterations (50);
        reg.setEuclideanFitnessEpsilon (1e-12);
        reg.setTransformationEpsilon (0.0001f * 0.0001f);
        reg.setMaxCorrespondenceDistance(0.005f);
        reg.align (output);
        pose_refined_ = reg.getFinalTransformation () * pose_refined_;
    }
    else
    {
        pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> reg;
        reg.setInputTarget (cloud_voxelized_icp_cropped); //scene
        reg.setInputSource (cloud); //scene
        reg.setMaximumIterations (50);
        reg.setEuclideanFitnessEpsilon (1e-12);
        reg.setTransformationEpsilon (0.0001f * 0.0001f);
        reg.setMaxCorrespondenceDistance(0.005f);
        reg.align (output);
        pose_refined_ = reg.getFinalTransformation ().inverse() * pose_refined_;
    }
    poseRefinementButtonPressed();

}

void
MainWindow::remove_selected()
{

  std::cout << "Going to remove selected hypothesis " << selected_hypothesis_ << " " << sequence_hypotheses_.size() << std::endl;

  if(selected_hypothesis_ < 0)
      return;

  sequence_hypotheses_.erase(sequence_hypotheses_.begin() + selected_hypothesis_);
  hypotheses_poses_.erase(hypotheses_poses_.begin() + selected_hypothesis_);
  pviz_->removePointCloud("highlighted");

  selected_hypothesis_ = -1;
  fillHypotheses();

  pviz_->spinOnce(0.1, true);

}

void MainWindow::refine_pose_changed(int state)
{
    std::cout << "refine pose state:" << state << std::endl;
    if(state == 0)
    {
        //disabled
        pviz_->removePointCloud("merged_cloud_refine_pose");
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> scene_handler(scene_merged_cloud_);
        pviz_->addPointCloud(scene_merged_cloud_, scene_handler, "merged_cloud", pviz_v1_);
        pviz_->removePointCloud("canonical_hypothesis_v2");
        //pviz_->removePointCloud("canonical_hypothesis_v1");
        pviz_->spinOnce(0.1, true);
        icp_button_->setEnabled(false);

        //pose_refined_ is the incremental scene transformation... modify pose hyoptheses accordingly
        Eigen::Matrix4f trans = pose_refined_ * hypotheses_poses_[selected_hypothesis_].inverse();
        hypotheses_poses_[selected_hypothesis_] = trans.inverse();
        //update hypotheses
        fillHypotheses();
    }
    else
    {
        //enabled, transform scene to the coordinate system of the selected_hypothesis_

        pcl::PointCloud<PointT>::ConstPtr model_cloud = sequence_hypotheses_[selected_hypothesis_]->getAssembled(0.003f);

        pviz_->removePointCloud("merged_cloud");

        pcl::PointCloud<PointT>::Ptr merged_cloud (new pcl::PointCloud<PointT>);

        merged_cloud_for_refinement_.reset(new pcl::PointCloud<PointT>);
        Eigen::Matrix4f trans = hypotheses_poses_[selected_hypothesis_].inverse();
        pcl::transformPointCloud(*scene_merged_cloud_, *merged_cloud, trans);

        float threshold = 0.3f;
        PointT minPoint, maxPoint;
        pcl::getMinMax3D<PointT>(*model_cloud, minPoint, maxPoint);
        minPoint.x -= threshold;
        minPoint.y -= threshold;
        minPoint.z -= threshold;

        maxPoint.x += threshold;
        maxPoint.y += threshold;
        maxPoint.z += threshold;

        pcl::CropBox<PointT> cropFilter;
        cropFilter.setInputCloud (merged_cloud);
        cropFilter.setMin(minPoint.getVector4fMap());
        cropFilter.setMax(maxPoint.getVector4fMap());
        cropFilter.filter (*merged_cloud_for_refinement_);

        pcl::visualization::PointCloudColorHandlerRGBField<PointT> scene_handler(merged_cloud_for_refinement_);
        pviz_->addPointCloud(merged_cloud_for_refinement_, scene_handler, "merged_cloud_refine_pose", pviz_v1_);
        pviz_->removePointCloud("highlighted");

        //visualize model in canonical coordinate frame
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> model_handler(model_cloud);
        pviz_->addPointCloud(model_cloud, model_handler, "canonical_hypothesis_v2");

        /*{
            pcl::visualization::PointCloudColorHandlerCustom<PointT> model_handler(model_cloud, 0, 255, 0);
            pviz_->addPointCloud(model_cloud, model_handler, "canonical_hypothesis_v1", pviz_v1_);
        }*/

        pviz_->spinOnce(0.1, true);
        icp_button_->setEnabled(true);
        pose_refined_ = Eigen::Matrix4f::Identity();
        merged_cloud_for_refinement_start_.reset(new pcl::PointCloud<PointT>(*merged_cloud_for_refinement_));
    }
}

void MainWindow::poseRefinementButtonPressed()
{
    //pviz_->removePointCloud("merged_cloud_refine_pose");

    pcl::transformPointCloud(*merged_cloud_for_refinement_start_, *merged_cloud_for_refinement_, pose_refined_);

    //pcl::visualization::PointCloudColorHandlerRGBField<PointT> scene_handler(merged_cloud_for_refinement_);
    //pviz_->addPointCloud(merged_cloud_for_refinement_, scene_handler, "merged_cloud_refine_pose", pviz_v1_);

    //pviz_->removePointCloud("canonical_hypothesis_v1");
    pviz_->removePointCloud("canonical_hypothesis_v2");

    Eigen::Matrix4f pose_refined_inverse = pose_refined_.inverse();
    pcl::PointCloud<PointT>::ConstPtr model_cloud = sequence_hypotheses_[selected_hypothesis_]->getAssembled(0.003f);
    pcl::PointCloud<PointT>::Ptr merged_cloud (new pcl::PointCloud<PointT>);
    pcl::transformPointCloud(*model_cloud, *merged_cloud, pose_refined_inverse);

    pcl::visualization::PointCloudColorHandlerRGBField<PointT> model_handler(merged_cloud);
    pviz_->addPointCloud(merged_cloud, model_handler, "canonical_hypothesis_v2");

    /*{
        pcl::visualization::PointCloudColorHandlerCustom<PointT> model_handler(merged_cloud, 0, 255, 0);
        pviz_->addPointCloud(merged_cloud, model_handler, "canonical_hypothesis_v1", pviz_v1_);
    }*/

    pviz_->spinOnce(0.1, true);

}

void MainWindow::x_plus()
{
    std::cout << "x_plus() " << selected_hypothesis_ << std::endl;
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,1>(0,3) = Eigen::Vector3f(translation_step_, 0, 0);
    pose_refined_ = m4f * pose_refined_;
    poseRefinementButtonPressed();
}

void MainWindow::x_minus()
{
    std::cout << "x_minus()" << std::endl;
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,1>(0,3) = Eigen::Vector3f(-translation_step_, 0, 0);
    pose_refined_ = m4f * pose_refined_;
    poseRefinementButtonPressed();
}

void MainWindow::y_plus()
{
    std::cout << "y_plus()" << std::endl;
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,1>(0,3) = Eigen::Vector3f(0, translation_step_, 0);
    pose_refined_ = m4f * pose_refined_;
    poseRefinementButtonPressed();
}

void MainWindow::y_minus()
{
    std::cout << "y_minus()" << std::endl;
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,1>(0,3) = Eigen::Vector3f(0, -translation_step_, 0);
    pose_refined_ = m4f * pose_refined_;
    poseRefinementButtonPressed();
}


void MainWindow::z_plus()
{
    std::cout << "z_plus()" << std::endl;
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,1>(0,3) = Eigen::Vector3f(0, 0, translation_step_);
    pose_refined_ = m4f * pose_refined_;
    poseRefinementButtonPressed();
}

void MainWindow::z_minus()
{
    std::cout << "z_minus()" << std::endl;
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,1>(0,3) = Eigen::Vector3f(0, 0, -translation_step_);
    pose_refined_ = m4f * pose_refined_;
    poseRefinementButtonPressed();
}

void MainWindow::xr_plus()
{
    std::cout << "xr_plus()" << std::endl;
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(rotation_step_, Eigen::Vector3f::UnitX());
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,3>(0,0) = m;
    pose_refined_ = m4f * pose_refined_;
    poseRefinementButtonPressed();
}

void MainWindow::xr_minus()
{
    std::cout << "xr_minus()" << std::endl;
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(-rotation_step_, Eigen::Vector3f::UnitX());
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,3>(0,0) = m;
    pose_refined_ = m4f * pose_refined_;
    poseRefinementButtonPressed();
}

void MainWindow::yr_plus()
{
    std::cout << "yr_plus()" << std::endl;
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(rotation_step_, Eigen::Vector3f::UnitY());
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,3>(0,0) = m;
    pose_refined_ = m4f * pose_refined_;
    poseRefinementButtonPressed();
}

void MainWindow::yr_minus()
{
    std::cout << "yr_minus()" << std::endl;
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(-rotation_step_, Eigen::Vector3f::UnitY());
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,3>(0,0) = m;
    pose_refined_ = m4f * pose_refined_;
    poseRefinementButtonPressed();
}


void MainWindow::zr_plus()
{
    std::cout << "zr_plus()" << std::endl;
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(rotation_step_, Eigen::Vector3f::UnitZ());
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,3>(0,0) = m;
    pose_refined_ = m4f * pose_refined_;
    poseRefinementButtonPressed();
}

void MainWindow::zr_minus()
{
    std::cout << "zr_minus()" << std::endl;
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(-rotation_step_, Eigen::Vector3f::UnitZ());
    Eigen::Matrix4f m4f = Eigen::Matrix4f::Identity();
    m4f.block<3,3>(0,0) = m;
    pose_refined_ = m4f * pose_refined_;
    poseRefinementButtonPressed();
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
                faat_pcl::rec_3d_framework::PersistenceUtils::writeMatrixToFile(pose_file.str(), transform);

            }
        }
    }
}

inline void
MainWindow::getModelsInDirectoryNonRecursive (bf::path & dir, std::string & rel_path_so_far, std::vector<std::string> & relative_paths, std::string & ext)
{
  bf::directory_iterator end_itr;
  for (bf::directory_iterator itr (dir); itr != end_itr; ++itr)
  {
    //check if its a directory, then get models in it
    if (bf::is_directory (*itr))
    {

    }
    else
    {
      //check that it is a ply file and then add, otherwise ignore..
      std::vector<std::string> strs;
#if BOOST_FILESYSTEM_VERSION == 3
      std::string file = (itr->path ().filename ()).string ();
#else
      std::string file = (itr->path ()).filename ();
#endif

      boost::split (strs, file, boost::is_any_of ("."));
      std::string extension = strs[strs.size () - 1];

      if (extension.compare (ext) == 0)
      {
#if BOOST_FILESYSTEM_VERSION == 3
        std::string path = rel_path_so_far + (itr->path ().filename ()).string ();
#else
        std::string path = rel_path_so_far + (itr->path ()).filename ();
#endif

        relative_paths.push_back (path);
      }
    }
  }
}

MainWindow::MainWindow(int argc, char *argv[])
{

  translation_step_ = 0.01f;
  rotation_step_ = pcl::deg2rad(5.f);

  model_xsize_ = 200;
  pose_xsize_ = 300;

  export_ground_truth_to_ = "";
  icp_scene_to_model_ = true;

  pcl::console::parse_argument (argc, argv, "-models_dir", dir_models_);
  pcl::console::parse_argument (argc, argv, "-model_scale", model_scale_);
  pcl::console::parse_argument (argc, argv, "-pcd_file", pcd_file_);
  pcl::console::parse_argument (argc, argv, "-export_ground_truth_to_", export_ground_truth_to_);
  pcl::console::parse_argument (argc, argv, "-icp_scene_to_model", icp_scene_to_model_);

  std::vector<std::string> files;
  std::string start = "";
  std::string ext = std::string ("pcd");
  bf::path dir = pcd_file_;
  getModelsInDirectoryNonRecursive (dir, start, files, ext);
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
    faat_pcl::rec_3d_framework::PersistenceUtils::readMatrixFromFile2(trans, transform);

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

  QObject::connect(icp_button_, SIGNAL(clicked(bool)),
                            this, SLOT(lock_with_icp()));

  QObject::connect(save_model_, SIGNAL(clicked(bool)),
                            this, SLOT(save_model()));

  refine_pose_ = new QCheckBox("Refine pose");
  QObject::connect(refine_pose_, SIGNAL(stateChanged(int)),
                   this, SLOT(refine_pose_changed(int)));

  QPushButton * x_plus = new QPushButton("X+");
  QPushButton * x_minus = new QPushButton("X-");

  QObject::connect(x_plus, SIGNAL(clicked(bool)),
                   this, SLOT(x_plus()));

  QObject::connect(x_minus, SIGNAL(clicked(bool)),
                   this, SLOT(x_minus()));

  QHBoxLayout *trans_x = new QHBoxLayout;
  trans_x->addWidget(x_minus);
  trans_x->addWidget(x_plus);

  QPushButton * y_plus = new QPushButton("Y+");
  QPushButton * y_minus = new QPushButton("Y-");

  QObject::connect(y_plus, SIGNAL(clicked(bool)),
                   this, SLOT(y_plus()));

  QObject::connect(y_minus, SIGNAL(clicked(bool)),
                   this, SLOT(y_minus()));


  QHBoxLayout *trans_y = new QHBoxLayout;
  trans_y->addWidget(y_minus);
  trans_y->addWidget(y_plus);

  QPushButton * z_plus = new QPushButton("Z+");
  QPushButton * z_minus = new QPushButton("Z-");

  QObject::connect(z_plus, SIGNAL(clicked(bool)),
                   this, SLOT(z_plus()));

  QObject::connect(z_minus, SIGNAL(clicked(bool)),
                   this, SLOT(z_minus()));

  QHBoxLayout *trans_z = new QHBoxLayout;
  trans_z->addWidget(z_minus);
  trans_z->addWidget(z_plus);

  QVBoxLayout * refine_pose_translation = new QVBoxLayout;
  refine_pose_translation->addLayout(trans_x);
  refine_pose_translation->addLayout(trans_y);
  refine_pose_translation->addLayout(trans_z);

  QPushButton * xr_plus = new QPushButton("XR+");
  QPushButton * xr_minus = new QPushButton("XR-");
  QObject::connect(xr_plus, SIGNAL(clicked(bool)),
                   this, SLOT(xr_plus()));

  QObject::connect(xr_minus, SIGNAL(clicked(bool)),
                   this, SLOT(xr_minus()));

  QHBoxLayout *rot_x = new QHBoxLayout;
  rot_x->addWidget(xr_minus);
  rot_x->addWidget(xr_plus);

  QPushButton * yr_plus = new QPushButton("YR+");
  QPushButton * yr_minus = new QPushButton("YR-");

  QObject::connect(yr_plus, SIGNAL(clicked(bool)),
                   this, SLOT(yr_plus()));

  QObject::connect(yr_minus, SIGNAL(clicked(bool)),
                   this, SLOT(yr_minus()));

  QHBoxLayout *rot_y = new QHBoxLayout;
  rot_y->addWidget(yr_minus);
  rot_y->addWidget(yr_plus);

  QPushButton * zr_plus = new QPushButton("ZR+");
  QPushButton * zr_minus = new QPushButton("ZR-");

  QObject::connect(zr_plus, SIGNAL(clicked(bool)),
                   this, SLOT(zr_plus()));

  QObject::connect(zr_minus, SIGNAL(clicked(bool)),
                   this, SLOT(zr_minus()));

  QHBoxLayout *rot_z = new QHBoxLayout;
  rot_z->addWidget(zr_minus);
  rot_z->addWidget(zr_plus);

  QVBoxLayout * refine_pose_rotation = new QVBoxLayout;
  refine_pose_rotation->addLayout(rot_x);
  refine_pose_rotation->addLayout(rot_y);
  refine_pose_rotation->addLayout(rot_z);

  //QFrame * qframe_refine_pose = new QFrame("pose refinement");

  QHBoxLayout *layout_refine_pose = new QHBoxLayout;
  layout_refine_pose->addLayout(refine_pose_translation);
  layout_refine_pose->addLayout(refine_pose_rotation);
  //qframe_refine_pose->

  icp_button_->setEnabled(false);

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


  QPushButton * remove_highlighted_hypotheses_ = new QPushButton("Remove selected hypotheses");
  QObject::connect(remove_highlighted_hypotheses_, SIGNAL(clicked(bool)),
                              this, SLOT(remove_selected()));

  QVBoxLayout * model_list_layout = new QVBoxLayout;
  model_list_ = new QListView;
  model_list_->setMaximumWidth(500);

  QObject::connect(model_list_, SIGNAL(doubleClicked(const QModelIndex&)),
                            this, SLOT(model_list_clicked(const QModelIndex&)));

  model_list_layout->setSizeConstraint(QLayout::SetMaximumSize);

  QLabel * label1 = new QLabel("Hypotheses:");

  model_list_layout->addWidget(label1);
  model_list_layout->addWidget(model_list_);
  model_list_layout->addWidget(remove_highlighted_hypotheses_);

  //control_vl->addLayout(scene_list_layout);
  control_vl->addLayout(model_list_layout);
  control_vl->addWidget(refine_pose_);
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

  //pviz_->registerKeyboardCallback(&keyboard_callback, (void*)(this));

  //pviz_models_->resetCamera();
  mainWindow_->setLayout(main_vl);
  mainWindow_->show();
  app.exec();
}
