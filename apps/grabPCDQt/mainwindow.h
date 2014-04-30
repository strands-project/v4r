/*
 * Software License Agreement (GNU General Public License)
 *
 *  Copyright (c) 2011, Thomas Mörwald
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
 * @author thomas.moerwald
 *
 */
#ifndef _GRAB_PCD_QT_MAINWINDOW_H_
#define _GRAB_PCD_QT_MAINWINDOW_H_

#include <QMainWindow>
#include "glviewer.h"
#include "params.h"
#include "sensor.h"

#undef Success
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

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

private slots:

  void rgbd_path_changed();
  void pcd_path_changed();

  void on_actionPreferences_triggered();
  void on_actionExit_triggered();
  void on_ButtonStart_pressed();
  void on_ButtonStop_pressed();
  void on_checkColor_clicked();
  void on_checkPoints_clicked();
  void on_pushSaveRGBD_pressed();
  void on_pushSavePCD_pressed();
  void on_pushReset_pressed();

private:
  void make_extension(std::string& filename, std::string ext);

  Ui::MainWindow *m_ui;
  GLViewer *m_glviewer;
  GLGraphicsView *m_glview;
  Params* m_params;
  Sensor* m_sensor;

  size_t m_num_saves_disp;
  size_t m_num_saves_pcd;

};

std::string ZeroPadNumber(int num, int width);

void cv_to_pcd(cv::Mat& points, cv::Mat& color, pcl::PointCloud<pcl::PointXYZRGB>& cloud);

#endif // MAINWINDOW_H
