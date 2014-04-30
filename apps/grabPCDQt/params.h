/*////////////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2013, Jonathan Balzer
//
// All rights reserved.
//
// This file is part of the R4R library.
//
// The R4R library is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The R4R library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with the R4R library. If not, see <http://www.gnu.org/licenses/>.
//
////////////////////////////////////////////////////////////////////////////////*/

#ifndef _GRAB_PCD_PARAMS_H_
#define _GRAB_PCD_PARAMS_H_

#include <QDialog>
#include "v4r/CalibrationRGBD/dccam.h"

namespace Ui {
class Params;
}

class Params : public QDialog
{
  Q_OBJECT

public:

  //! Constructor.
  explicit Params(QWidget *parent = 0);

  //! Destructor.
  ~Params();

  CDepthColorCam get_depth_color_cam();

  void apply_params();
  std::string get_rgpd_path();
  std::string get_pcd_path();
  bool get_save_pcd_binary();

signals:

  void cam_params_changed(const CDepthColorCam& cam);
  void rgbd_path_changed();
  void pcd_path_changed();


private slots:
  void on_applyButton_clicked();

  void on_saveButton_clicked();

  void on_loadButton_clicked();

  void on_loadErrorPatternButton_clicked();

  void on_okButton_pressed();

  void on_pushFindRGBDPath_pressed();

  void on_pushFindPCDPath_pressed();

private:

  Ui::Params* ui;
  cv::Mat m_D;

};

#endif // PARAMS_H
