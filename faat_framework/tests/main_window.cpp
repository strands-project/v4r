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
#include <faat_pcl/3d_rec_framework/pc_source/mesh_source.h>
#include <faat_pcl/registration/icp_with_gc.h>
#include <faat_pcl/3d_rec_framework/feature_wrapper/normal_estimator.h>

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

void
keyboard_callback (const pcl::visualization::KeyboardEvent& event, void* cookie)
{

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

void MainWindow::scene_list_clicked(const QModelIndex & idx) {
  std::cout << "scene list clicqued..." << static_cast<int>(idx.row()) << std::endl;
  pviz_->removeAllPointClouds();
  pviz_->removeAllShapes();

  //Load cloud of the selected scene
  orig_cloud_.reset (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile(scenes_list_[idx.row()], *orig_cloud_);
  curr_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>(*orig_cloud_));

  pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> scene_handler(curr_cloud_, "z");
  pviz_->addPointCloud<pcl::PointXYZ>(curr_cloud_, scene_handler, "current_cloud");
  pviz_->resetCamera();
  pviz_->addCoordinateSystem(0.1);
  pviz_->spinOnce(0.1,true);

  //Load GT models and visualize them

  std::vector<std::string> strs1;
  boost::split (strs1, scenes_list_[idx.row()], boost::is_any_of ("/"));
  std::string id_1 = strs1[strs1.size () - 1];
  size_t pos1 = id_1.find (".pcd");
  id_1 = id_1.substr (0, pos1);
  current_scene_id_ = id_1;
  or_eval_.visualizeGroundTruth(*pviz_, id_1, 0, false);

  pviz_->spinOnce(0.1,true);

}

void MainWindow::lock_with_gc_icp() {
  //Sample aligned current model
  pcl::PointCloud<pcl::PointXYZ>::Ptr sampled_(new  pcl::PointCloud<pcl::PointXYZ>());
  pcl::transformPointCloud(*model_clouds_[current_model_], *sampled_, last_transform_);

  typedef pcl::PointNormal PointTInternal;
  pcl::PointCloud<PointTInternal>::Ptr curr_cloud_normals(new  pcl::PointCloud<PointTInternal>());
  pcl::PointCloud<PointTInternal>::Ptr model_with_normals(new  pcl::PointCloud<PointTInternal>());
  pcl::PointCloud<PointTInternal>::Ptr output(new  pcl::PointCloud<PointTInternal>());


  boost::shared_ptr<faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<pcl::PointXYZ, pcl::Normal> > normal_estimator;
  normal_estimator.reset (new faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<pcl::PointXYZ, pcl::Normal>);
  normal_estimator->setCMR (false);
  normal_estimator->setDoVoxelGrid (true);
  normal_estimator->setRemoveOutliers (true);
  normal_estimator->setMinNRadius (27);
  normal_estimator->setValuesForCMRFalse (0.003f, 0.018f);

  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr processed (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr est_normals(new pcl::PointCloud<pcl::Normal>);

    normal_estimator->estimate (curr_cloud_, processed, est_normals);
    pcl::copyPointCloud (*est_normals, *curr_cloud_normals);
    pcl::copyPointCloud (*processed, *curr_cloud_normals);
  }

  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr processed (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr est_normals(new pcl::PointCloud<pcl::Normal>);

    normal_estimator->estimate (sampled_, processed, est_normals);
    pcl::copyPointCloud (*est_normals, *model_with_normals);
    pcl::copyPointCloud (*processed, *model_with_normals);
  }

  typename pcl::registration::TransformationEstimationPointToPlaneLLS<PointTInternal, PointTInternal>::Ptr
                                                                                                           trans_lls (
                                                                                                                      new pcl::registration::TransformationEstimationPointToPlaneLLS<
                                                                                                                          PointTInternal,
                                                                                                                          PointTInternal>);

  typename pcl::registration::CorrespondenceRejectorSampleConsensus<PointTInternal>::Ptr
                                                                                         rej (
                                                                                              new pcl::registration::CorrespondenceRejectorSampleConsensus<
                                                                                                  PointTInternal> ());

  rej->setMaximumIterations (1000);
  rej->setInlierThreshold (0.01f);

  float max_corresp_dist = std::numeric_limits<float>::infinity();
  float ov_percentage_ = 0.5f;
  Eigen::Matrix4f icp_trans;
  faat_pcl::IterativeClosestPointWithGC<PointTInternal, PointTInternal> icp;
  icp.setTransToCentroid(false);
  icp.setTransformationEpsilon (0.000001 * 0.000001);
  icp.setMinNumCorrespondences (3);
  icp.setMaxCorrespondenceDistance (max_corresp_dist);
  icp.setUseCG (true);
  icp.setSurvivalOfTheFittest (false);
  icp.setMaximumIterations(5);
  icp.setOverlapPercentage(ov_percentage_);
  icp.setVisFinal(true);
  icp.setDtVxSize(0.01f);
  pcl::registration::DefaultConvergenceCriteria<float>::Ptr convergence_criteria;
  convergence_criteria = icp.getConvergeCriteria ();
  convergence_criteria->setAbsoluteMSE (1e-9);
  convergence_criteria->setMaximumIterationsSimilarTransforms (15);
  convergence_criteria->setFailureAfterMaximumIterations (false);
  icp.setInputTarget (curr_cloud_normals);
  rej->setInputTarget (curr_cloud_normals);
  icp.setInputSource (model_with_normals);
  rej->setInputSource (model_with_normals);
  icp.align (*output);
  icp_trans = icp.getFinalTransformation ();

  std::cout << "ICP finished..." << std::endl;
  //Concatenate transfrom
  last_transform_ = icp_trans * last_transform_;

  {
    //Transform and show model model
    pviz_->removeShape("curr_mesh");
    pcl::transformPointCloud(*model_clouds_[current_model_], *sampled_, last_transform_);
    pviz_->addPointCloud(sampled_, "curr_mesh");
    pviz_->spinOnce(0.1,true);
  }
}

void MainWindow::lock_with_icp() {
  //Sample aligned current model
  pcl::PointCloud<pcl::PointXYZ>::Ptr sampled_(new  pcl::PointCloud<pcl::PointXYZ>());
  pcl::transformPointCloud(*model_clouds_[current_model_], *sampled_, last_transform_);

  //Perform ICP and add ICP transform to last_transform_
  Eigen::Matrix4f icp_trans;
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> reg;
  reg.setInputSource (sampled_);
  reg.setInputTarget (curr_cloud_);

  //Perform ICP and save the final transformation
  reg.setMaximumIterations (20);
  reg.setMaxCorrespondenceDistance (0.01);

  reg.align (*sampled_);
  icp_trans = reg.getFinalTransformation ();

  /*typedef pcl::PointNormal PointTInternal;
  pcl::PointCloud<PointTInternal>::Ptr curr_cloud_normals(new  pcl::PointCloud<PointTInternal>());
  pcl::PointCloud<PointTInternal>::Ptr model_with_normals(new  pcl::PointCloud<PointTInternal>());
  pcl::PointCloud<PointTInternal>::Ptr output(new  pcl::PointCloud<PointTInternal>());


  boost::shared_ptr<faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<pcl::PointXYZ, pcl::Normal> > normal_estimator;
  normal_estimator.reset (new faat_pcl::rec_3d_framework::PreProcessorAndNormalEstimator<pcl::PointXYZ, pcl::Normal>);
  normal_estimator->setCMR (false);
  normal_estimator->setDoVoxelGrid (true);
  normal_estimator->setRemoveOutliers (true);
  normal_estimator->setMinNRadius (27);
  normal_estimator->setValuesForCMRFalse (0.003f, 0.018f);

  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr processed (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr est_normals(new pcl::PointCloud<pcl::Normal>);

    normal_estimator->estimate (curr_cloud_, processed, est_normals);
    pcl::copyPointCloud (*est_normals, *curr_cloud_normals);
    pcl::copyPointCloud (*processed, *curr_cloud_normals);
  }

  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr processed (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr est_normals(new pcl::PointCloud<pcl::Normal>);

    normal_estimator->estimate (sampled_, processed, est_normals);
    pcl::copyPointCloud (*est_normals, *model_with_normals);
    pcl::copyPointCloud (*processed, *model_with_normals);
  }

  typename pcl::registration::TransformationEstimationPointToPlaneLLS<PointTInternal, PointTInternal>::Ptr
                                                                                                           trans_lls (
                                                                                                                      new pcl::registration::TransformationEstimationPointToPlaneLLS<
                                                                                                                          PointTInternal,
                                                                                                                          PointTInternal>);

  typename pcl::registration::CorrespondenceRejectorSampleConsensus<PointTInternal>::Ptr
                                                                                         rej (
                                                                                              new pcl::registration::CorrespondenceRejectorSampleConsensus<
                                                                                                  PointTInternal> ());

  rej->setMaximumIterations (1000);
  rej->setInlierThreshold (0.01f);

  float max_corresp_dist = std::numeric_limits<float>::infinity();
  float ov_percentage_ = 0.5f;
  faat_pcl::IterativeClosestPointWithGC<PointTInternal, PointTInternal> icp;
  icp.setTransToCentroid(false);
  icp.setTransformationEpsilon (0.000001 * 0.000001);
  icp.setMinNumCorrespondences (3);
  icp.setMaxCorrespondenceDistance (max_corresp_dist);
  icp.setUseCG (true);
  icp.setSurvivalOfTheFittest (false);
  icp.setMaximumIterations(10);
  icp.setOverlapPercentage(ov_percentage_);
  icp.setVisFinal(false);
  icp.setDtVxSize(0.01f);
  pcl::registration::DefaultConvergenceCriteria<float>::Ptr convergence_criteria;
  convergence_criteria = icp.getConvergeCriteria ();
  convergence_criteria->setAbsoluteMSE (1e-9);
  convergence_criteria->setMaximumIterationsSimilarTransforms (15);
  convergence_criteria->setFailureAfterMaximumIterations (false);
  icp.setInputTarget (curr_cloud_normals);
  rej->setInputTarget (curr_cloud_normals);
  icp.setInputSource (model_with_normals);
  rej->setInputSource (model_with_normals);
  icp.align (*output);
  icp_trans = icp.getFinalTransformation ();*/

  std::cout << "ICP finished..." << std::endl;
  //Concatenate transfrom
  last_transform_ = icp_trans * last_transform_;

  {
    //Transform and show model model
    pviz_->removeShape("curr_mesh");
    pcl::transformPointCloud(*model_clouds_[current_model_], *sampled_, last_transform_);
    pviz_->addPointCloud(sampled_, "curr_mesh");
    pviz_->spinOnce(0.1,true);
  }
}

void MainWindow::cut_z() {

  for(size_t i=0; i < addedToVis_.size(); i++) {
    pviz_->removePointCloud(addedToVis_[i]);
  }

  std::cout << input_z_->text().toDouble() << std::endl;

  pcl::PassThrough<pcl::PointXYZ> pass_;
  pass_.setKeepOrganized(true);

  pass_.setFilterLimits (0, input_z_->text().toDouble());
  pass_.setFilterFieldName ("z");
  pass_.setInputCloud (orig_cloud_);
  pass_.filter (*curr_cloud_);

  pviz_->removePointCloud("current_cloud");
  pviz_->addPointCloud<pcl::PointXYZ>(curr_cloud_,"current_cloud");
  pviz_->spinOnce(0.1,true);

  chop_at_z_ = input_z_->text().toDouble();
}

void
MainWindow::save_to_disk ()
{

  Eigen::Vector4f centroid = centroids_[current_model_] * -1.f;
  Eigen::Matrix4f transform;
  transform.setIdentity ();
  transform (0, 3) = centroid[0];
  transform (1, 3) = centroid[1];
  transform (2, 3) = centroid[2];

  std::vector<std::string> strs1;
  boost::split (strs1, loaded_models_[current_model_], boost::is_any_of ("/"));
  std::string id_1 = strs1[strs1.size () - 1];
  size_t pos1 = id_1.find (".ply");
  std::string m_id = id_1.substr (0, pos1);
  Eigen::Matrix4f to_save = last_transform_ * transform;
  or_eval_.updateGT (current_scene_id_, m_id, to_save);
  or_eval_.loadGTData();
}

void MainWindow::save_model() {
  //Remove curr_mesh
  pviz_->removeShape("curr_mesh");

  //Add model again with a specific ID!

  vtkMatrix4x4 * matrix = vtkMatrix4x4::New();
  for(size_t i=0; i < 4; i++) {
   for(size_t j=0; j < 4; j++) {
     matrix->SetElement(i,j,last_transform_(i,j));
   }
  }

  vtkTransform * trans = vtkTransform::New();
  trans->SetMatrix(matrix);
  trans->Modified ();

  std::stringstream name;
  name << "mesh_" << model_poses_.size();
  pviz_->addModelFromPLYFile(loaded_models_[selected_model_], trans, name.str());
  pviz_->spinOnce(0.1,true);

  //Save transform and model ID
  ModelPose mp;
  mp.nid_ = id_save_models_;
  mp.id_ = loaded_models_[selected_model_];
  mp.transform_  = last_transform_;
  model_poses_.push_back(mp);

  id_save_models_++;

  QStringList list;
  for(size_t i=0; i < model_poses_.size(); i++) {

    std::stringstream id;
    id << model_poses_[i].nid_ << " - " << model_poses_[i].id_;
    list << QString(id.str().c_str());
  }

  model_list_->setModel(new QStringListModel(list));

  last_transform_.setIdentity();
  counter_.setZero();

  save_model_->setEnabled(false);
  icp_button_->setEnabled(false);

}

MainWindow::MainWindow(int argc, char *argv[])
{
  counter_.resize(6);
  model_xsize_ = 200;
  pose_xsize_ = 300;
  /*dir_models_ = "/home/aitor/wg/cvfh_models";
  dir_output_ = "/home/aitor/wg/scenes_gt_data";*/

  pcl::console::parse_argument (argc, argv, "-models_dir_vis", dir_models_vis_);
  pcl::console::parse_argument (argc, argv, "-models_dir", dir_models_);
  pcl::console::parse_argument (argc, argv, "-model_scale", model_scale_);
  pcl::console::parse_argument (argc, argv, "-pcd_file", pcd_file_);
  pcl::console::parse_argument (argc, argv, "-GT_DIR", GT_DIR_);
  pcl::console::parse_argument (argc, argv, "-training_dir", training_dir_);

  std::vector<std::string> files;
  std::string start = "";
  std::string ext = std::string ("pcd");
  bf::path dir = pcd_file_;
  faat_pcl::rec_3d_framework::or_evaluator::getModelsInDirectory (dir, start, files, ext);
  std::cout << "Number of scenes in directory is:" << files.size () << std::endl;
  for (size_t i = 0; i < files.size (); i++)
  {
    std::cout << files[i] << std::endl;
    std::stringstream filestr;
    filestr << pcd_file_ << "/" << files[i];
    std::string file = filestr.str ();
    scenes_list_.push_back (file);
  }

  boost::shared_ptr<faat_pcl::rec_3d_framework::MeshSource<pcl::PointXYZ> > mesh_source (new faat_pcl::rec_3d_framework::MeshSource<pcl::PointXYZ>);
  mesh_source->setPath (dir_models_);
  mesh_source->setResolution (250);
  mesh_source->setTesselationLevel (1);
  mesh_source->setViewAngle (57.f);
  mesh_source->setRadiusSphere (1.5f);
  mesh_source->setModelScale (model_scale_);

  mesh_source->generate (training_dir_);

  boost::shared_ptr<faat_pcl::rec_3d_framework::Source<pcl::PointXYZ> > cast_source;
  cast_source = boost::static_pointer_cast<faat_pcl::rec_3d_framework::MeshSource<pcl::PointXYZ> > (mesh_source);

  or_eval_.setGTDir(GT_DIR_);
  or_eval_.setModelsDir(dir_models_);
  or_eval_.setScenesDir(pcd_file_);
  or_eval_.setDataSource(cast_source);
  or_eval_.loadGTData();

  id_save_models_ = 0;

  counter_.setZero();

  /*orig_cloud_.reset (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile(argv[1],*orig_cloud_);
  curr_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>(*orig_cloud_));*/

  QApplication app(argc,argv);
  mainWindow_ = new QWidget;
  mainWindow_->resize(1680,1000);

  vtk_widget_ = new QVTKWidget;
  vtk_widget_->resize(800, 256);

  pviz_ = new pcl::visualization::PCLVisualizer("test_viz",true);
  vtk_widget_->SetRenderWindow(pviz_->getRenderWindow());
  pviz_->addCoordinateSystem(0.1);
  //pviz_->addPointCloud<pcl::PointXYZ>(curr_cloud_,"current_cloud");
  //pviz_->resetCamera();

  pviz_->registerKeyboardCallback(&keyboard_callback, (void*)(this));

  vtk_widget_->show();

  QVBoxLayout *main_vl = new QVBoxLayout;
  QVBoxLayout *control_vl = new QVBoxLayout;

  icp_button_ = new QPushButton("Lock with ICP");

  gc_icp_button_ = new QPushButton("Lock with GC-ICP");

  save_model_ = new QPushButton("Save model & pose");

  QObject::connect(icp_button_, SIGNAL(clicked(bool)),
                            this, SLOT(lock_with_icp()));

  QObject::connect(gc_icp_button_, SIGNAL(clicked(bool)),
                              this, SLOT(lock_with_gc_icp()));

  QObject::connect(save_model_, SIGNAL(clicked(bool)),
                            this, SLOT(save_model()));

  save_model_->setEnabled(false);
  icp_button_->setEnabled(false);
  gc_icp_button_->setEnabled(false);

  QVBoxLayout * scene_list_layout = new QVBoxLayout;
  scene_list_ = new QListView;
  scene_list_->setMaximumWidth(500);
  {
    QStringList list;
    for(size_t i=0; i < scenes_list_.size(); i++) {
      list << QString(scenes_list_[i].c_str());
    }

    scene_list_->setModel(new QStringListModel(list));
  }

  QObject::connect(scene_list_, SIGNAL(doubleClicked(const QModelIndex&)),
                            this, SLOT(scene_list_clicked(const QModelIndex&)));

  scene_list_layout->setSizeConstraint(QLayout::SetMaximumSize);

  QLabel * label_scenes = new QLabel("Scenes:");
  scene_list_layout->addWidget(label_scenes);
  scene_list_layout->addWidget(scene_list_);
  QHBoxLayout *layout_vis = new QHBoxLayout;

  pviz_models_ = new pcl::visualization::PCLVisualizer("viz_models",true);
  pviz_models_->registerMouseCallback (mouse_callback_models, (void*)(this));

  vtk_widget_models_ = new QVTKWidget;
  vtk_widget_models_->resize(getNumberOfModels(dir_models_)*model_xsize_, model_xsize_);
  vtk_widget_models_->SetRenderWindow(pviz_models_->getRenderWindow());

  layout_vis->addLayout(control_vl);
  layout_vis->addWidget(vtk_widget_);


  QPushButton * save_scene_ = new QPushButton("Update GT with current model");
  QObject::connect(save_scene_, SIGNAL(clicked(bool)),
                              this, SLOT(save_to_disk()));

  QVBoxLayout * model_list_layout = new QVBoxLayout;
  model_list_ = new QListView;
  model_list_->setMaximumWidth(500);
  model_list_layout->setSizeConstraint(QLayout::SetMaximumSize);

  QLabel * label1 = new QLabel("Models added:");

  model_list_layout->addWidget(label1);
  model_list_layout->addWidget(model_list_);
  model_list_layout->addWidget(save_scene_);

  control_vl->addLayout(scene_list_layout);
  control_vl->addLayout(model_list_layout);
  control_vl->addWidget(gc_icp_button_);
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
  fillModels(dir_models_);

  //pviz_models_->resetCamera();
  mainWindow_->setLayout(main_vl);
  mainWindow_->show();
  app.exec();
}
