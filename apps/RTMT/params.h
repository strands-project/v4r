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

#ifndef _GRAB_PCD_PARAMS_H_
#define _GRAB_PCD_PARAMS_H_

#ifndef Q_MOC_RUN
#include <QDialog>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <v4r/reconstruction/KeypointSlamRGBD2.h>
#endif


namespace Ui {
class Params;
}

class RGBDCameraParameter
{
public:
  double f[2];
  double c[2];
  int width, height;
  double range[2];
  RGBDCameraParameter() {
    f[0] = 525; f[1] = 525;
    c[0] = 320; c[1] = 240;
    width = 640;
    height = 480;
    range[0] = 0.4; range[1] = 3.14;
  }
};

class CamaraTrackerParameter
{
public:
  bool log_point_clouds;
  bool create_prev_cloud;
  double min_delta_angle;
  double min_delta_distance;
  double prev_voxegrid_size;
  double prev_z_cutoff;
  CamaraTrackerParameter() : log_point_clouds(true), create_prev_cloud(true), min_delta_angle(20.), min_delta_distance(1.), prev_voxegrid_size(0.01), prev_z_cutoff(2.) {}
};

class BundleAdjustmentParameter
{
public:
  double dist_cam_add_projections;
  BundleAdjustmentParameter() : dist_cam_add_projections(0.1) {}
};

class SegmentationParameter
{
public:
  double inl_dist_plane;
  double thr_angle;
  double min_points_plane;
  SegmentationParameter() : inl_dist_plane(0.01), thr_angle(45), min_points_plane(5000) {}
};

class ObjectModelling
{
public:
  double vx_size_object;
  double edge_radius_px;

  ObjectModelling() : vx_size_object(0.001), edge_radius_px(3) {}
};

/**
 * @brief The Params class
 */
class Params : public QDialog
{
  Q_OBJECT

public:

  //! Constructor.
  explicit Params(QWidget *parent = 0);

  //! Destructor.
  ~Params();


  void apply_params();
  void apply_cam_params();
  std::string get_rgbd_path();
  std::string get_object_name();
  void set_object_name(const QString &txt);


signals:

  void cam_params_changed(const RGBDCameraParameter& cam);
  void cam_tracker_params_changed(const CamaraTrackerParameter& param);
  void rgbd_path_changed();
  void bundle_adjustment_parameter_changed(const BundleAdjustmentParameter& param);
  void segmentation_parameter_changed(const SegmentationParameter& param);
  void object_modelling_parameter_changed(const ObjectModelling& param);
  void set_roi_params(const double &_bbox_scale_xy, const double &_bbox_scale_height, const double &_seg_offs);
  void set_segmentation_params(bool use_roi_segm, const double &offs, bool _use_dense_mv, const double &_edge_radius_px);
  void set_cb_param(bool create_cb, float rnn_thr);



private slots:

  void on_okButton_clicked();
  void on_pushFindRGBDPath_pressed();

  void on_applyButton_clicked();

private:

  Ui::Params* ui;

  RGBDCameraParameter cam_params;
  CamaraTrackerParameter cam_tracker_params;
  BundleAdjustmentParameter ba_params;
  SegmentationParameter seg_params;
  ObjectModelling om_params;

};

#endif // PARAMS_H
