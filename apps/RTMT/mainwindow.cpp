/**
 * $Id$
 * 
 * Software License Agreement (GNU General Public License)
 *
 *  Copyright (C) 2015:
 *
 *    Johann Prankl, prankl@acin.tuwien.ac.at
 *    Aitor Aldoma, aldoma@acin.tuwien.ac.at
 *
 *      Automation and Control Institute
 *      Vienna University of Technology
 *      Gusshausstraße 25-29
 *      1170 Vienn, Austria
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @author Johann Prankl, Aitor Aldoma
 *
 */

/**
 * TODO:
 * - parameter config of roi
 * - consistent poses
 * - use roi for segmentation?
 * - use qt5
 */

#ifndef Q_MOC_RUN
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <v4r/keypoints/impl/toString.hpp>

#include <QFileDialog>
#include <QMessageBox>
#endif

Q_DECLARE_METATYPE(cv::Mat)

using namespace std;

MainWindow::MainWindow(QWidget *parent) :
  QMainWindow(parent),
  m_ui(new Ui::MainWindow),
  m_params(new Params(this)),
  m_sensor(new Sensor()),
  m_segmentation(new ObjectSegmentation()),
  m_store_tracking_model(new StoreTrackingModel()),
  m_ba(new BundleAdjustment()),
  m_multi_session(new MultiSession()),
  have_multi_session(false),
  m_num_saves_disp(0),
  m_num_saves_pcd(0),
  idx_seg(-1),
  num_clouds(0),
  bbox_active(false)
{
  m_ui->setupUi(this);

  m_glviewer = new GLViewer(this);
  m_glview = new GLGraphicsView(m_ui->centralWidget);
  m_ui->glView = m_glview;
  m_glview->setGeometry(10,0,640,480);
  m_glview->setViewport(m_glviewer);

  // input signals
  connect(m_glview, SIGNAL(mouse_moved(QMouseEvent*)),
          m_glviewer, SLOT(mouse_moved(QMouseEvent*)));
  connect(m_glview, SIGNAL(mouse_pressed(QMouseEvent*)),
          m_glviewer, SLOT(mouse_pressed(QMouseEvent*)));
  connect(m_glview, SIGNAL(key_pressed(QKeyEvent*)),
          m_glviewer, SLOT(key_pressed(QKeyEvent*)));
  connect(m_glview, SIGNAL(wheel_event(QWheelEvent*)),
          m_glviewer, SLOT(wheel_event(QWheelEvent*)));

  // param signals
  connect(m_params, SIGNAL(cam_params_changed(const RGBDCameraParameter)),
          m_glviewer, SLOT(cam_params_changed(const RGBDCameraParameter)));
  connect(m_params, SIGNAL(cam_params_changed(const RGBDCameraParameter)),
          m_sensor, SLOT(cam_params_changed(const RGBDCameraParameter)));
  connect(m_params, SIGNAL(cam_tracker_params_changed(const CamaraTrackerParameter)),
          m_sensor, SLOT(cam_tracker_params_changed(const CamaraTrackerParameter)));
  connect(m_params, SIGNAL(bundle_adjustment_parameter_changed(const BundleAdjustmentParameter)),
          m_sensor, SLOT(bundle_adjustment_parameter_changed(const BundleAdjustmentParameter)));
  connect(m_params, SIGNAL(cam_params_changed(const RGBDCameraParameter)),
          m_segmentation, SLOT(cam_params_changed(const RGBDCameraParameter)));
  connect(m_params, SIGNAL(segmentation_parameter_changed(const SegmentationParameter)),
          m_segmentation, SLOT(segmentation_parameter_changed(const SegmentationParameter)));
  connect(m_params, SIGNAL(object_modelling_parameter_changed(const ObjectModelling)),
          m_segmentation, SLOT(object_modelling_parameter_changed(const ObjectModelling)));
  connect(m_params, SIGNAL(cam_params_changed(const RGBDCameraParameter)),
          m_store_tracking_model, SLOT(cam_params_changed(const RGBDCameraParameter)));
  connect(m_params, SIGNAL(cam_tracker_params_changed(const CamaraTrackerParameter)),
          m_ba, SLOT(cam_tracker_params_changed(const CamaraTrackerParameter)));
  connect(m_params, SIGNAL(set_roi_params(const double, const double, const double)),
          m_sensor, SLOT(set_roi_params(const double, const double, const double)));
  connect(m_params, SIGNAL(object_modelling_parameter_changed(const ObjectModelling)),
          m_multi_session, SLOT(object_modelling_parameter_changed(const ObjectModelling)));


  // sensor signals
  qRegisterMetaType< pcl::PointCloud<pcl::PointXYZRGB>::Ptr >("pcl::PointCloud<pcl::PointXYZRGB>::Ptr");
  qRegisterMetaType< cv::Mat_<cv::Vec3b> >("cv::Mat_<cv::Vec3b>");
  qRegisterMetaType< boost::shared_ptr< std::vector<Sensor::CameraLocation> > >("boost::shared_ptr< std::vector<Sensor::CameraLocation> >");
  qRegisterMetaType< boost::shared_ptr< Sensor::AlignedPointXYZRGBVector > >("boost::shared_ptr< Sensor::AlignedPointXYZRGBVector >");
  qRegisterMetaType< std::string >("std::string");
  qRegisterMetaType< std::vector<Eigen::Vector3f> >("std::vector<Eigen::Vector3f>");
  qRegisterMetaType< Eigen::Matrix4f >("Eigen::Matrix4f");
  qRegisterMetaType< Eigen::Vector3f >("Eigen::Vector3f");

  connect(m_sensor, SIGNAL(new_image(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr, const cv::Mat_<cv::Vec3b>)),
          m_glviewer, SLOT(new_image(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr, const cv::Mat_<cv::Vec3b>)));
  connect(m_sensor, SIGNAL(update_model_cloud(const boost::shared_ptr< Sensor::AlignedPointXYZRGBVector >)),
          m_glviewer, SLOT(update_model_cloud(const boost::shared_ptr< Sensor::AlignedPointXYZRGBVector >)));
  connect(m_segmentation, SIGNAL(update_model_cloud(const boost::shared_ptr< Sensor::AlignedPointXYZRGBVector >)),
          m_glviewer, SLOT(update_model_cloud(const boost::shared_ptr< Sensor::AlignedPointXYZRGBVector >)));
  connect(m_sensor, SIGNAL(update_cam_trajectory(const boost::shared_ptr< std::vector<Sensor::CameraLocation> > )),
          m_glviewer, SLOT(update_cam_trajectory(const boost::shared_ptr< std::vector<Sensor::CameraLocation> > )));
  connect(m_sensor, SIGNAL(update_visualization()),
          m_glviewer, SLOT(update_visualization()));
  connect(m_sensor, SIGNAL(printStatus(const std::string)),
          this, SLOT(printStatus(const std::string)));
  connect(m_sensor, SIGNAL(finishedOptimizeCameras(int)),
          this, SLOT(finishedOptimizeCameras(int)));
  connect(m_glviewer, SIGNAL(select_roi(int, int)),
          m_sensor, SLOT(select_roi(int, int)));

  // object segmentation
  connect(this, SIGNAL(set_image(int)),
          m_segmentation, SLOT(set_image(int)));
  connect(m_glviewer, SIGNAL(segment_image(int, int)),
          m_segmentation, SLOT(segment_image(int, int)));
  connect(m_segmentation, SIGNAL(new_image(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr, const cv::Mat_<cv::Vec3b>)),
          m_glviewer, SLOT(new_image(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr, const cv::Mat_<cv::Vec3b>)));
  connect(m_segmentation, SIGNAL(printStatus(const std::string)),
          this, SLOT(printStatus(const std::string)));
  connect(m_segmentation, SIGNAL(update_visualization()),
          m_glviewer, SLOT(update_visualization()));
  connect(m_sensor, SIGNAL(update_boundingbox(const std::vector<Eigen::Vector3f>, const Eigen::Matrix4f)),
          m_glviewer, SLOT(update_boundingbox(const std::vector<Eigen::Vector3f>, const Eigen::Matrix4f)));
  connect(m_sensor, SIGNAL(set_roi(const Eigen::Vector3f, const Eigen::Vector3f, const Eigen::Matrix4f)),
          m_segmentation, SLOT(set_roi(const Eigen::Vector3f, const Eigen::Vector3f, const Eigen::Matrix4f)));
  connect(m_params, SIGNAL(set_segmentation_params(bool, const double, bool, const double)),
          m_segmentation, SLOT(set_segmentation_params(bool, const double, bool, const double)));
  connect(m_segmentation, SIGNAL(finishedObjectSegmentation()),
          this, SLOT(finishedObjectSegmentation()));

  // StoreTrackingModel
  connect(m_store_tracking_model, SIGNAL(printStatus(const std::string)),
          this, SLOT(printStatus(const std::string)));
  connect(m_store_tracking_model, SIGNAL(finishedStoreTrackingModel()),
          this, SLOT(finishedStoreTrackingModel()));
  connect(m_segmentation, SIGNAL(set_object_base_transform(const Eigen::Matrix4f)),
          m_store_tracking_model, SLOT(set_object_base_transform(const Eigen::Matrix4f)));
  connect(m_params, SIGNAL(set_cb_param(bool, float)),
          m_store_tracking_model, SLOT(set_cb_param(bool, float)));

  // bundle adjustment
  connect(m_ba, SIGNAL(update_model_cloud(const boost::shared_ptr< Sensor::AlignedPointXYZRGBVector >)),
          m_glviewer, SLOT(update_model_cloud(const boost::shared_ptr< Sensor::AlignedPointXYZRGBVector >)));
  connect(m_ba, SIGNAL(update_cam_trajectory(const boost::shared_ptr< std::vector<Sensor::CameraLocation> > )),
          m_glviewer, SLOT(update_cam_trajectory(const boost::shared_ptr< std::vector<Sensor::CameraLocation> > )));
  connect(m_ba, SIGNAL(update_visualization()),
          m_glviewer, SLOT(update_visualization()));
  connect(m_ba, SIGNAL(printStatus(const std::string)),
          this, SLOT(printStatus(const std::string)));
  connect(m_ba, SIGNAL(finishedOptimizeCameras(int)),
          this, SLOT(finishedOptimizeCameras(int)));

  // multi session
  connect(m_multi_session, SIGNAL(finishedAlignment(bool)),
          this, SLOT(finishedAlignment(bool)));
  connect(m_multi_session, SIGNAL(printStatus(const std::string)),
          this, SLOT(printStatus(const std::string)));
  connect(m_multi_session, SIGNAL(update_model_cloud(const boost::shared_ptr< Sensor::AlignedPointXYZRGBVector >)),
          m_glviewer, SLOT(update_model_cloud(const boost::shared_ptr< Sensor::AlignedPointXYZRGBVector >)));
  connect(m_multi_session, SIGNAL(update_visualization()),
          m_glviewer, SLOT(update_visualization()));


  m_params->apply_cam_params();
  m_params->apply_params();

  setWindowTitle(tr("RTM-Toolbox"));
}

MainWindow::~MainWindow()
{
  delete m_ui;
  delete m_params;
  delete m_glviewer;
  delete m_glview;
  delete m_sensor;
}

/**
 * @brief MainWindow::activateAllButtons
 */
void MainWindow::activateAllButtons()
{
  m_ui->CamStart->setEnabled(true);
  m_ui->TrackerStart->setEnabled(true);
  m_ui->CamStop->setEnabled(true);
  m_ui->TrackerStop->setEnabled(true);

  m_ui->OptimizePoses->setEnabled(true);
  m_ui->SegmentObject->setEnabled(true);
  m_ui->SavePointClouds->setEnabled(true);
  m_ui->SaveTrackerModel->setEnabled(true);
  m_ui->undoOptimize->setEnabled(true);
  m_ui->okSegmentation->setEnabled(true);
  m_ui->OptimizeObject->setEnabled(true);
  m_ui->ResetTracker->setEnabled(true);
  m_ui->setROI->setEnabled(true);
  m_ui->ActivateROI->setEnabled(true);
  m_ui->ResetROI->setEnabled(true);
  m_ui->SessionAdd->setEnabled(true);
  m_ui->SessionAlign->setEnabled(true);
  m_ui->SessionClear->setEnabled(true);
  m_ui->SessionOptimize->setEnabled(true);
}

/**
 * @brief MainWindow::deactivateAllButtons
 */
void MainWindow::deactivateAllButtons()
{
  m_ui->CamStart->setEnabled(false);
  m_ui->TrackerStart->setEnabled(false);
  m_ui->CamStop->setEnabled(false);
  m_ui->TrackerStop->setEnabled(false);

  m_ui->OptimizePoses->setEnabled(false);
  m_ui->SegmentObject->setEnabled(false);
  m_ui->SavePointClouds->setEnabled(false);
  m_ui->SaveTrackerModel->setEnabled(false);
  m_ui->undoOptimize->setEnabled(false);
  m_ui->okSegmentation->setEnabled(false);
  m_ui->OptimizeObject->setEnabled(false);
  m_ui->ResetTracker->setEnabled(false);
  m_ui->setROI->setEnabled(false);
  m_ui->ActivateROI->setEnabled(false);
  m_ui->ResetROI->setEnabled(false);
  m_ui->SessionAdd->setEnabled(false);
  m_ui->SessionAlign->setEnabled(false);
  m_ui->SessionClear->setEnabled(false);
  m_ui->SessionOptimize->setEnabled(false);
}

/**
 * @brief MainWindow::activateTrackingButtons
 */
void MainWindow::activateTrackingButtons()
{
  m_ui->CamStart->setEnabled(true);
  m_ui->TrackerStart->setEnabled(true);
  m_ui->CamStop->setEnabled(true);
  m_ui->TrackerStop->setEnabled(true);

  m_ui->OptimizePoses->setEnabled(false);
  m_ui->SegmentObject->setEnabled(false);
  m_ui->SavePointClouds->setEnabled(false);
  m_ui->SaveTrackerModel->setEnabled(false);
  m_ui->undoOptimize->setEnabled(false);
  m_ui->okSegmentation->setEnabled(false);
  m_ui->OptimizeObject->setEnabled(false);
  m_ui->ResetTracker->setEnabled(false);
  //m_ui->setROI->setEnabled(false);
  //m_ui->ActivateROI->setEnabled(false);
  //m_ui->ResetROI->setEnabled(false);
  m_ui->SessionAdd->setEnabled(false);
  m_ui->SessionAlign->setEnabled(false);
  m_ui->SessionClear->setEnabled(false);
  m_ui->SessionOptimize->setEnabled(false);


  if (bbox_active) m_glviewer->drawBoundingBox(true);
}

/**
 * @brief MainWindow::activateModellingButtons
 */
void MainWindow::activateModellingButtons()
{
  m_ui->CamStart->setEnabled(false);
  m_ui->TrackerStart->setEnabled(false);
  m_ui->CamStop->setEnabled(false);
  m_ui->TrackerStop->setEnabled(false);

  m_ui->OptimizePoses->setEnabled(true);
  m_ui->SegmentObject->setEnabled(true);
  m_ui->SavePointClouds->setEnabled(true);
  m_ui->SaveTrackerModel->setEnabled(true);
  m_ui->undoOptimize->setEnabled(true);
  m_ui->okSegmentation->setEnabled(true);
  m_ui->OptimizeObject->setEnabled(true);
  m_ui->ResetTracker->setEnabled(true);
  //m_ui->setROI->setEnabled(true);
  //m_ui->ActivateROI->setEnabled(true);
  //m_ui->ResetROI->setEnabled(true);
  m_ui->SessionAdd->setEnabled(true);
  m_ui->SessionAlign->setEnabled(true);
  m_ui->SessionClear->setEnabled(true);
  m_ui->SessionOptimize->setEnabled(true);
}

/**
 * @brief MainWindow::setStartVis
 */
void MainWindow::setStartVis()
{
  m_ui->ShowImage->setChecked(true);
  m_ui->ShowDepthMask->setChecked(true);
  m_ui->ShowCameras->setChecked(false);
  m_ui->ShowPointCloud->setChecked(false);
  m_ui->ShowObjectModel->setChecked(false);

  m_glviewer->showImage(m_ui->ShowImage->isChecked());
  m_glviewer->showCameras(m_ui->ShowCameras->isChecked());
  m_glviewer->showCloud(m_ui->ShowPointCloud->isChecked());
  m_glviewer->showObject(m_ui->ShowObjectModel->isChecked());
  m_sensor->showDepthMask(m_ui->ShowDepthMask->isChecked());

  m_glviewer->resetView();
}

void MainWindow::printStatus(const std::string &_txt)
{
  m_ui->statusLabel->setText(_txt.c_str());
}

void MainWindow::on_actionPreferences_triggered()
{
  m_params->show();
}


void MainWindow::on_actionExit_triggered()
{
  QApplication::exit();
}

void MainWindow::on_CamStart_clicked()
{
  activateTrackingButtons();
  setStartVis();

  m_sensor->start(0);
  //m_glviewer->drawBoundingBox(false);
  m_ui->statusLabel->setText("Status: Started camera");
}

void MainWindow::on_CamStop_clicked()
{
  m_sensor->stop();

  activateAllButtons();

  m_glviewer->drawBoundingBox(false);

  m_ui->statusLabel->setText("Status: Stopped camera");
}

void MainWindow::on_TrackerStart_clicked()
{
  activateTrackingButtons();

  if (!m_sensor->isRunning()) setStartVis();

  m_sensor->startTracker(0);


  //m_glviewer->drawBoundingBox(false);
  m_ui->statusLabel->setText("Status: Started tracker");
}

void MainWindow::on_TrackerStop_clicked()
{
  m_sensor->stopTracker();

  activateAllButtons();
  m_glviewer->drawBoundingBox(false);

  m_ui->statusLabel->setText("Status: Stopped tracker");
}

void MainWindow::on_OptimizePoses_clicked()
{
  //int num_cameras;

  deactivateAllButtons();

  //m_ui->statusLabel->setText("Status: Optimizing camera locations ...");
  //m_sensor->optimizeCameras(num_cameras);
  m_ba->optimizeCamStructProj(m_sensor->getModel(),m_sensor->getTrajectory(),m_sensor->getClouds(),m_sensor->getAlignedCloud());
}

void MainWindow::finishedOptimizeCameras(int num_cameras)
{
  //std::string txt = std::string("Status: Optimized ")+v4r::toString(num_cameras,0)+std::string(" cameras");
  //m_ui->statusLabel->setText(txt.c_str());

  activateAllButtons();
}


void MainWindow::on_undoOptimize_clicked()
{
  if (m_ba->restoreCameras())
    m_ui->statusLabel->setText("Status: Undo camera optimization");
  else m_ui->statusLabel->setText("Status: Nothing to restore");
}

void MainWindow::on_SegmentObject_clicked()
{
  // stop camera otherwise it will occupy the window
  m_sensor->stop();
  m_ui->statusLabel->setText("Status: Stopped camera");

  idx_seg = 0;
  num_clouds = m_sensor->getClouds()->size();
  m_segmentation->setData(m_sensor->getCameras(), m_sensor->getClouds() );
  m_ui->imBackward->setEnabled(true);
  m_ui->imForward->setEnabled(true);

  deactivateAllButtons();

  m_ui->okSegmentation->setEnabled(true);
  m_ui->OptimizeObject->setEnabled(true);

  setStartVis();

  //m_glviewer->drawBoundingBox(true);
  m_glviewer->segmentObject(true);
  m_ui->statusLabel->setText("Activate only the image display and click on the object to segment...");
}

void MainWindow::on_okSegmentation_clicked()
{
  m_ui->statusLabel->setText("Status: Postprocessing the object...");
  m_glviewer->segmentObject(false);
  m_ui->okSegmentation->setEnabled(false);
  m_segmentation->finishSegmentation();
}

/**
 * @brief MainWindow::finishedObjectSegmentation
 */
void MainWindow::finishedObjectSegmentation()
{
  m_glviewer->drawBoundingBox(false);
  m_segmentation->drawObjectCloud();

  activateAllButtons();

  m_ui->ShowImage->setChecked(false);
  m_ui->ShowDepthMask->setChecked(false);
  m_ui->ShowCameras->setChecked(false);
  m_ui->ShowPointCloud->setChecked(false);
  m_ui->ShowObjectModel->setChecked(true);

  m_glviewer->showImage(m_ui->ShowImage->isChecked());
  m_glviewer->showCameras(m_ui->ShowCameras->isChecked());
  m_glviewer->showCloud(m_ui->ShowPointCloud->isChecked());
  m_glviewer->showObject(m_ui->ShowObjectModel->isChecked());
  m_sensor->showDepthMask(m_ui->ShowDepthMask->isChecked());
  m_glviewer->resetView(-1.);

  m_ui->imBackward->setEnabled(false);
  m_ui->imForward->setEnabled(false);
  m_ui->statusLabel->setText("Status: Finised segmentation");
}

void MainWindow::on_imForward_clicked()
{
  if (idx_seg<num_clouds-1) idx_seg++;
  emit set_image(idx_seg);
}

void MainWindow::on_imBackward_clicked()
{
  if (idx_seg>0) idx_seg--;
  emit set_image(idx_seg);
}

void MainWindow::on_SavePointClouds_clicked()
{
  bool ok_save;
  QString text = QString::fromStdString(m_params->get_object_name());

  QString model_name = QInputDialog::getText(this, tr("Store point clouds for recognition"), tr("Model name:"), QLineEdit::Normal, text, &ok_save);

  if ( ok_save && model_name.isNull() == false )
  {
    if (boost::filesystem::exists( m_params->get_rgbd_path() + "/" + model_name.toStdString()+ "/views/") )
    {
      int ret = QMessageBox::warning(this, tr("Store point clouds for recognition"),
                                     tr("The directory exists!\n"
                                        "Do you want to overwrite the files?"), QMessageBox::Save, QMessageBox::Cancel);
      if (ret!=QMessageBox::Save)
        ok_save = false;
    }

    if (ok_save)
    {
      m_params->set_object_name(model_name);
      m_ui->statusLabel->setText("Status: Save point clouds...");

      bool ok=false;

      if (have_multi_session)
        ok = m_multi_session->savePointClouds(m_params->get_rgbd_path(), model_name.toStdString());
      else
        ok = m_segmentation->savePointClouds(m_params->get_rgbd_path(), model_name.toStdString());

      if (ok) m_ui->statusLabel->setText("Status: Saved point clouds (recognition model)");
      else m_ui->statusLabel->setText("Status: No segmented data available!");
    }
  }
}

void MainWindow::on_SaveTrackerModel_clicked()
{
  bool ok;
  QString text = QString::fromStdString(m_params->get_object_name());

  QString object_name = QInputDialog::getText(this, tr("Store tracking model"), tr("Object name:"), QLineEdit::Normal, text, &ok);

  if ( ok && object_name.isNull() == false )
  {
    if (boost::filesystem::exists(m_params->get_rgbd_path() + "/" + object_name.toStdString() + "/tracking_model.ao"))
    {
      int ret = QMessageBox::warning(this, tr("Store tracking model"),
                                     tr("The object file exists!\n"
                                        "Do you want to overwrite the file?"), QMessageBox::Save, QMessageBox::Cancel);
      if (ret!=QMessageBox::Save)
        ok = false;
    }

    if (ok)
    {
      m_params->set_object_name(object_name);
      deactivateAllButtons();

      if (have_multi_session)
        m_store_tracking_model->storeTrackingModel(m_params->get_rgbd_path(), object_name.toStdString(), m_multi_session->getCameras(),m_multi_session->getClouds(),m_multi_session->getMasks(), Eigen::Matrix4f::Identity());
      else m_store_tracking_model->storeTrackingModel(m_params->get_rgbd_path(), object_name.toStdString(), m_segmentation->getCameras(),m_sensor->getClouds(),m_segmentation->getMasks(), m_segmentation->getObjectBaseTransform());
    }
  }
}

void  MainWindow::finishedStoreTrackingModel()
{
  activateAllButtons();
}

void MainWindow::on_ResetView_clicked()
{
  m_glviewer->resetView();
}

void MainWindow::on_ShowImage_clicked()
{
  m_glviewer->showImage(m_ui->ShowImage->isChecked());
}

void MainWindow::on_ShowCameras_clicked()
{
  m_glviewer->showCameras(m_ui->ShowCameras->isChecked());
}

void MainWindow::on_ShowPointCloud_clicked()
{
  m_glviewer->showCloud(m_ui->ShowPointCloud->isChecked());
}

void MainWindow::on_ShowObjectModel_clicked()
{
  m_glviewer->showObject(m_ui->ShowObjectModel->isChecked());
}

void MainWindow::on_ResetTracker_clicked()
{
  m_sensor->reset();
  m_ui->statusLabel->setText("Status: Reset tracker");
}

void MainWindow::on_ShowDepthMask_clicked()
{
  m_sensor->showDepthMask(m_ui->ShowDepthMask->isChecked());
}

void MainWindow::on_setROI_clicked()
{
  bbox_active = true;
  activateTrackingButtons();
  m_sensor->start(0);
  m_glviewer->drawBoundingBox(true);
  m_glviewer->selectROI(true);
  m_ui->statusLabel->setText("For automatic ROI generation click to the supporting surface (e.g. top surface of a turntable)");
}

void MainWindow::on_ActivateROI_clicked()
{
  bbox_active = true;
  m_glviewer->selectROI(false);
  m_sensor->activateROI(true);
  m_glviewer->drawBoundingBox(true);
  m_segmentation->activateROI(true);
}

void MainWindow::on_ResetROI_clicked()
{
  bbox_active = false;
  m_glviewer->drawBoundingBox(false);
  m_glviewer->selectROI(false);
  m_sensor->activateROI(false);
  m_segmentation->activateROI(false);
}



void MainWindow::on_OptimizeObject_clicked()
{
  m_ui->statusLabel->setText("Status: Postprocessing the object...");
  m_glviewer->segmentObject(false);
  m_ui->okSegmentation->setEnabled(false);
  m_ui->OptimizeObject->setEnabled(false);
  if (!m_segmentation->optimizeMultiview())
  {
    finishedObjectSegmentation();
  }
}

void MainWindow::on_SessionAdd_clicked()
{
  if (m_segmentation->getCameras().size()==0 || m_segmentation->getObjectIndices().size()==0)
  {
    m_ui->statusLabel->setText("No data available! Did you click 'Segment'?");
    return;
  }

  m_multi_session->addSequences(m_segmentation->getCameras(),m_sensor->getClouds(),m_segmentation->getObjectIndices(), m_segmentation->getObjectBaseTransform());
}

void MainWindow::on_SessionAlign_clicked()
{
  m_ui->SessionOptimize->setEnabled(false);
  m_multi_session->alignSequences();
}

void MainWindow::on_SessionClear_clicked()
{
  m_multi_session->clear();
  have_multi_session = false;
}

void MainWindow::finishedAlignment(bool ok)
{
  m_ui->SessionOptimize->setEnabled(true);
  if (ok) have_multi_session = true;
}

void MainWindow::on_SessionOptimize_clicked()
{
  if (have_multi_session)
    m_multi_session->optimizeSequences();
}


