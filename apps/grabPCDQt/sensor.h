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
#ifndef _GRAB_PCD_SENSOR_H_
#define _GRAB_PCD_SENSOR_H_

#include <QThread>
#include <QMutex>
#include <opencv2/opencv.hpp>
#include "v4r/CalibrationRGBD/dsensor.h"

class Sensor : public QThread
{
  Q_OBJECT

public:
  Sensor();
  ~Sensor();

  void start(int cam_id=0);
  void stop();
  bool is_running();
  void get_image(cv::Mat &color, cv::Mat &disparity);
  void get_undistorted(cv::Mat& color, cv::Mat& points);

public slots:
  void cam_params_changed(const CDepthColorCam& cam);

signals:
  void new_image(cv::Mat color, cv::Mat disparity);

private:
  void run();

  CDepthColorCam m_cam;

  cv::Mat m_color;
  cv::Mat m_disparity;

  cv::Mat m_points, m_color_undistorted;

  bool m_run;
  int m_cam_id;

  QMutex raw_mutex;
  QMutex undistorted_mutex;

};

#endif // _GRAB_PCD_SENSOR_H_
