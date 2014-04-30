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
#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QMessageBox>

Q_DECLARE_METATYPE(cv::Mat)

using namespace std;

MainWindow::MainWindow(QWidget *parent) :
  QMainWindow(parent),
  m_ui(new Ui::MainWindow),
  m_params(new Params(this)),
  m_sensor(new Sensor()),
  m_num_saves_disp(0),
  m_num_saves_pcd(0)
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
  connect(m_params, SIGNAL(cam_params_changed(CDepthColorCam)),
          m_glviewer, SLOT(cam_params_changed(CDepthColorCam)));
  connect(m_params, SIGNAL(cam_params_changed(CDepthColorCam)),
          m_sensor, SLOT(cam_params_changed(CDepthColorCam)));

  connect(m_params, SIGNAL(rgbd_path_changed()),
          this, SLOT(rgbd_path_changed()));
  connect(m_params, SIGNAL(pcd_path_changed()),
          this, SLOT(pcd_path_changed()));

  // sensor signals
  qRegisterMetaType<cv::Mat>("cv::Mat");
  connect(m_sensor, SIGNAL(new_image(cv::Mat,cv::Mat)),
          m_glviewer, SLOT(new_image(cv::Mat,cv::Mat)));

  m_params->apply_params();

  setWindowTitle(tr("Grab PCD Qt"));
}

MainWindow::~MainWindow()
{
  delete m_ui;
  delete m_params;
  delete m_glviewer;
  delete m_glview;
  delete m_sensor;
}

void MainWindow::rgbd_path_changed()
{
  m_num_saves_disp = 0;
}

void MainWindow::pcd_path_changed()
{
  m_num_saves_pcd = 0;
}

void MainWindow::on_actionPreferences_triggered()
{
  m_params->show();
}


void MainWindow::on_actionExit_triggered()
{
  QApplication::exit();
}

void MainWindow::make_extension(string& filename, string ext)
{
  if(ext.find_first_of(".")!=0)
    ext.insert(0, ".");

  std::string::size_type idx_dot = filename.rfind('.');
  std::string::size_type idx_sl = filename.rfind('/');
  std::string::size_type idx_bsl = filename.rfind('\\');
  if((idx_dot == std::string::npos) ||
     ((idx_dot < idx_sl) && (idx_sl != std::string::npos)) ||
     ((idx_dot < idx_bsl) && (idx_bsl != std::string::npos)))
    filename.append(ext);
}

void MainWindow::on_ButtonStart_pressed()
{
  m_sensor->start(0);
  m_ui->pushSavePCD->setEnabled(true);
  m_ui->pushSaveRGBD->setEnabled(true);
}

void MainWindow::on_ButtonStop_pressed()
{
  m_sensor->stop();
  m_ui->pushSavePCD->setEnabled(false);
  m_ui->pushSaveRGBD->setEnabled(false);
}

void MainWindow::on_checkColor_clicked()
{
  m_glviewer->show_color(m_ui->checkColor->isChecked());
}

void MainWindow::on_checkPoints_clicked()
{
  m_glviewer->show_points(m_ui->checkPoints->isChecked());
}

void MainWindow::on_pushSaveRGBD_pressed()
{
  std::string path = m_params->get_rgpd_path();

  // create file names
  std::string file_color, file_disparity;
  file_color = path + ZeroPadNumber(m_num_saves_disp, 4) + "-c1.jpg";
  file_disparity = path + ZeroPadNumber(m_num_saves_disp, 4) + "-d.pgm";

  cv::Mat color, disparity;
  m_sensor->get_image(color, disparity);

  if(color.empty() || disparity.empty())
    return;

  // write color image
  cv::Mat rgb;
  cv::cvtColor(color,rgb,CV_BGR2RGB);
  cv::imwrite(file_color.c_str(),rgb);

  // write disparity image
  std::vector<int> params;
  params.push_back(CV_IMWRITE_PXM_BINARY);
  params.push_back(1);
  cv::imwrite(file_disparity.c_str(), disparity, params);

  cout << "Saved: " << file_color << endl;
  cout << "Saved: " << file_disparity << endl;

  m_num_saves_disp++;
}

void MainWindow::on_pushSavePCD_pressed()
{
  std::string path = m_params->get_pcd_path();

  cv::Mat color, points;
  m_sensor->get_undistorted(color, points);

  if(color.empty() || points.empty())
    return;

  // convert to pcl::PointCloud
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  cv_to_pcd(points, color, cloud);

  // save
  std::string file = path + "cloud_" + ZeroPadNumber(m_num_saves_pcd, 4) + ".pcd";

  cout << "Saved: " << file << endl;

  if(m_params->get_save_pcd_binary())
    pcl::io::savePCDFileBinary(file, cloud);
  else
    pcl::io::savePCDFileASCII(file, cloud);
  cout << "Saved: " << file << endl;

  m_num_saves_pcd++;
}

void MainWindow::on_pushReset_pressed()
{
  m_glviewer->reset_view();
}

#include <iomanip>
std::string ZeroPadNumber(int num, int width)
{
  std::ostringstream ss;
  ss << std::setw( width ) << std::setfill( '0' ) << num;
  return ss.str();
}

void cv_to_pcd(cv::Mat& points, cv::Mat& color, pcl::PointCloud<pcl::PointXYZRGB>& cloud)
{
  cv::Point3f p;
  cv::Vec3b c;

  cloud.width = points.cols;
  cloud.height = points.rows;
  cloud.points.resize(cloud.width*cloud.height);
  cloud.is_dense = false;

  float nan = std::numeric_limits<float>::quiet_NaN();

  // collect points
  for(size_t i=0; i<(size_t)points.rows; i++)
  {
    for(size_t j=0; j<(size_t)points.cols; j++)
    {
      c = color.at<cv::Vec3b>(i,j);
      p = points.at<cv::Point3f>(i,j);

      pcl::PointXYZRGB& pt = cloud.at(j,i);

      pt.r = c[0];
      pt.g = c[1];
      pt.b = c[2];

      if(cv::norm(p)>0)
      {
        pt.x = p.x;
        pt.y = p.y;
        pt.z = p.z;
      }else{
        pt.x = nan;
        pt.y = nan;
        pt.z = nan;
      }

    }
  }
}
