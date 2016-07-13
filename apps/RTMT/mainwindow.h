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

#ifndef _GRAB_PCD_QT_MAINWINDOW_H_
#define _GRAB_PCD_QT_MAINWINDOW_H_

#ifndef Q_MOC_RUN
#include <QMainWindow>
#include <QInputDialog>
#include "glviewer.h"
#include "params.h"
#include "sensor.h"
#include "ObjectSegmentation.h"
#include "StoreTrackingModel.h"
#include "BundleAdjustment.h"
#include "MultiSession.h"

#undef Success
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/octree/octree.h>
#include "OctreeVoxelCentroidContainerXYZRGB.hpp"
#endif

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
  Q_OBJECT
  
public:
  explicit MainWindow(QWidget *parent = 0);
  ~MainWindow();

signals:
  void set_image(int idx);

private slots:

  void on_actionPreferences_triggered();
  void on_actionExit_triggered();

  void on_CamStart_clicked();
  void on_CamStop_clicked();
  void on_TrackerStart_clicked();
  void on_TrackerStop_clicked();
  void on_OptimizePoses_clicked();
  void on_SegmentObject_clicked();
  void on_SavePointClouds_clicked();
  void on_SaveTrackerModel_clicked();
  void on_ResetView_clicked();
  void on_ResetTracker_clicked();

  void on_ShowImage_clicked();
  void on_ShowCameras_clicked();
  void on_ShowPointCloud_clicked();
  void on_ShowObjectModel_clicked();
  void on_undoOptimize_clicked();
  void on_okSegmentation_clicked();
  void on_imForward_clicked();
  void on_imBackward_clicked();

  void finishedOptimizeCameras(int num_cameras);
  void finishedStoreTrackingModel();
  void finishedObjectSegmentation();
  void finishedAlignment(bool ok);

  void on_ShowDepthMask_clicked();
  void on_setROI_clicked();
  void on_ActivateROI_clicked();
  void on_ResetROI_clicked();
  void on_OptimizeObject_clicked();
  void on_SessionAdd_clicked();
  void on_SessionAlign_clicked();
  void on_SessionClear_clicked();

  void on_SessionOptimize_clicked();

public slots:
  void printStatus(const std::string &_txt);

private:
//  void make_extension(std::string& filename, std::string ext);

  Ui::MainWindow *m_ui;
  GLViewer *m_glviewer;
  GLGraphicsView *m_glview;
  Params* m_params;
  Sensor* m_sensor;
  ObjectSegmentation* m_segmentation;
  StoreTrackingModel* m_store_tracking_model;
  BundleAdjustment* m_ba;
  MultiSession* m_multi_session;

  bool have_multi_session;

  size_t m_num_saves_disp;
  size_t m_num_saves_pcd;
  int idx_seg, num_clouds;
  bool bbox_active;

  void activateAllButtons();
  void deactivateAllButtons();
  void activateTrackingButtons();
  void activateModellingButtons();
  void setStartVis();

};


#endif // MAINWINDOW_H
