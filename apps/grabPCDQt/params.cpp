#include "params.h"
#include "ui_params.h"

#include <opencv2/opencv.hpp>

#include <vector>
#include <fstream>

#include <QFileDialog>

using namespace cv;
using namespace std;

Params::Params(QWidget *parent):
  QDialog(parent),
  ui(new Ui::Params),
  m_D(Mat::zeros(480,640,CV_32FC1))
{
  ui->setupUi(this);
}

Params::~Params()
{
  delete ui;
}

CDepthColorCam Params::get_depth_color_cam()
{
  // get params
  vector<size_t> srgb, sd;
  vector<float> frgb, fd, crgb, cd, krgb, kd, range, dd, da;
  srgb.push_back(640);
  srgb.push_back(480);
  frgb.push_back(ui->fuRgbEdit->text().toFloat());
  frgb.push_back(ui->fvRgbEdit->text().toFloat());
  crgb.push_back(ui->cuRgbEdit->text().toFloat());
  crgb.push_back(ui->cvRgbEdit->text().toFloat());
  krgb.push_back(ui->k0Edit->text().toFloat());
  krgb.push_back(ui->k1Edit->text().toFloat());
  krgb.push_back(ui->k2Edit->text().toFloat());
  krgb.push_back(ui->k3Edit->text().toFloat());
  krgb.push_back(ui->k4Edit->text().toFloat());

  Mat F = Mat::eye(4,4,CV_32FC1);
  F.at<float>(0,0) = ui->r11Edit->text().toFloat();
  F.at<float>(0,1) = ui->r12Edit->text().toFloat();
  F.at<float>(0,2) = ui->r13Edit->text().toFloat();
  F.at<float>(0,3) = ui->txEdit->text().toFloat();
  F.at<float>(1,0) = ui->r21Edit->text().toFloat();
  F.at<float>(1,1) = ui->r22Edit->text().toFloat();
  F.at<float>(1,2) = ui->r23Edit->text().toFloat();
  F.at<float>(1,3) = ui->tyEdit->text().toFloat();
  F.at<float>(2,0) = ui->r31Edit->text().toFloat();
  F.at<float>(2,1) = ui->r32Edit->text().toFloat();
  F.at<float>(2,2) = ui->r33Edit->text().toFloat();
  F.at<float>(2,3) = ui->tzEdit->text().toFloat();

  CCam rgb(srgb,frgb,crgb,0,krgb,F);

  sd.push_back(640);
  sd.push_back(480);
  fd.push_back(ui->fuEdit->text().toFloat());
  fd.push_back(ui->fvEdit->text().toFloat());
  cd.push_back(ui->cuEdit->text().toFloat());
  cd.push_back(ui->cvEdit->text().toFloat());
  kd.push_back(ui->k0DEdit->text().toFloat());
  kd.push_back(ui->k1DEdit->text().toFloat());
  kd.push_back(ui->k2DEdit->text().toFloat());
  kd.push_back(ui->k3DEdit->text().toFloat());
  kd.push_back(ui->k4DEdit->text().toFloat());
  range.push_back(ui->zminEdit->text().toFloat());
  range.push_back(ui->zmaxEdit->text().toFloat());
  dd.push_back(ui->c0Edit->text().toFloat());
  dd.push_back(ui->c1Edit->text().toFloat());
  da.push_back(ui->alpha0Edit->text().toFloat());
  da.push_back(ui->alpha1Edit->text().toFloat());

  CDepthCam dcam(sd,fd,cd,0,vector<float>(5,0),Mat::eye(4,4,CV_32FC1),range,dd,m_D,da);

  return CDepthColorCam(rgb,dcam);
}

void Params::apply_params()
{
  emit cam_params_changed(get_depth_color_cam());
}


std::string Params::get_rgpd_path()
{
  std::string path = ui->editRGBDPath->text().toStdString();

  if(path.empty())
    path += ".";

  return (path + "/");
}

string Params::get_pcd_path()
{
  std::string path =  ui->editPCDPath->text().toStdString();

  if(path.empty())
    path += ".";

  return (path + "/");
}

bool Params::get_save_pcd_binary()
{
  return ui->checkPCDbinary->isChecked();
}

void Params::on_applyButton_clicked()
{
  apply_params();
}

void Params::on_saveButton_clicked()
{
  QString filename = QFileDialog::getSaveFileName(this, tr("Save file..."),".",tr("*.txt"));

  ofstream out(filename.toStdString().c_str());

  if(!out.is_open())
    return;

  CDepthColorCam cam = get_depth_color_cam();

  out << cam;
  out.close();

  cout << "Saved: " << filename.toStdString() << endl;
}

void Params::on_loadButton_clicked()
{

  QString filename = QFileDialog::getOpenFileName(this, tr("Load file..."),".",tr("*.txt"));

  ifstream in(filename.toStdString().c_str());

  if(!in.is_open())
    return;

  CDepthColorCam cam;

  in >> cam;
  in.close();

  emit cam_params_changed(cam);

  ui->fuRgbEdit->setText(QString::number(cam.m_rgb_cam.m_f[0]));
  ui->fvRgbEdit->setText(QString::number(cam.m_rgb_cam.m_f[1]));
  ui->cuRgbEdit->setText(QString::number(cam.m_rgb_cam.m_c[0]));
  ui->cvRgbEdit->setText(QString::number(cam.m_rgb_cam.m_c[1]));
  ui->k0Edit->setText(QString::number(cam.m_rgb_cam.m_k[0]));
  ui->k1Edit->setText(QString::number(cam.m_rgb_cam.m_k[1]));
  ui->k2Edit->setText(QString::number(cam.m_rgb_cam.m_k[2]));
  ui->k3Edit->setText(QString::number(cam.m_rgb_cam.m_k[3]));
  ui->k4Edit->setText(QString::number(cam.m_rgb_cam.m_k[4]));

  Mat& F = cam.m_rgb_cam.GetExtrinsics();
  ui->r11Edit->setText(QString::number(F.at<float>(0,0)));
  ui->r12Edit->setText(QString::number(F.at<float>(0,1)));
  ui->r13Edit->setText(QString::number(F.at<float>(0,2)));
  ui->txEdit->setText(QString::number(F.at<float>(0,3)));
  ui->r21Edit->setText(QString::number(F.at<float>(1,0)));
  ui->r22Edit->setText(QString::number(F.at<float>(1,1)));
  ui->r23Edit->setText(QString::number(F.at<float>(1,2)));
  ui->tyEdit->setText(QString::number(F.at<float>(1,3)));
  ui->r31Edit->setText(QString::number(F.at<float>(2,0)));
  ui->r32Edit->setText(QString::number(F.at<float>(2,1)));
  ui->r33Edit->setText(QString::number(F.at<float>(2,2)));
  ui->tzEdit->setText(QString::number(F.at<float>(2,3)));

  ui->fuEdit->setText(QString::number(cam.m_depth_cam.m_f[0]));
  ui->fvEdit->setText(QString::number(cam.m_depth_cam.m_f[1]));
  ui->cuEdit->setText(QString::number(cam.m_depth_cam.m_c[0]));
  ui->cvEdit->setText(QString::number(cam.m_depth_cam.m_c[1]));
  ui->c0Edit->setText(QString::number(cam.m_depth_cam.m_d[0]));
  ui->c1Edit->setText(QString::number(cam.m_depth_cam.m_d[1]));
  ui->zminEdit->setText(QString::number(cam.m_depth_cam.m_range[0]));
  ui->zmaxEdit->setText(QString::number(cam.m_depth_cam.m_range[1]));

}

void Params::on_loadErrorPatternButton_clicked()
{

  QString filename = QFileDialog::getOpenFileName(this, tr("Save file..."),".",tr("*.txt"));

  float* buffer = new float[640*480];

  ifstream in(filename.toStdString().c_str());

  if(!in.is_open())
    return;

  in.read((char*)buffer,sizeof(float)*640*480);

  in.close();

  m_D = Mat(480,640,CV_32FC1,buffer);

  // does the mat constructor copy this?
  // delete [] buffer;

}

void Params::on_okButton_pressed()
{

}

void Params::on_pushFindRGBDPath_pressed()
{
  QString filename = QFileDialog::getExistingDirectory(this, tr("RGBD Path"), tr("./"));
  if(filename.size()!=0)
  {
    ui->editRGBDPath->setText(filename);
    emit rgbd_path_changed();
  }
}

void Params::on_pushFindPCDPath_pressed()
{
  QString filename = QFileDialog::getExistingDirectory(this, tr("PCD Path"), tr("./"));
  if(filename.size()!=0)
  {
    ui->editPCDPath->setText(filename);
    emit pcd_path_changed();
  }
}
