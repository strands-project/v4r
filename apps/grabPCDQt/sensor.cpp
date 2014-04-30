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
#include "sensor.h"

using namespace std;

Sensor::Sensor() :
  m_cam_id(0)
{

}

Sensor::~Sensor()
{
  stop();
}

void Sensor::start(int cam_id)
{
  m_cam_id = cam_id;
  QThread::start();
}

void Sensor::stop()
{
  if(m_run)
  {
    m_run = false;
    this->wait();
  }
}

bool Sensor::is_running()
{
  return m_run;
}

void Sensor::get_image(cv::Mat& color, cv::Mat& disparity)
{
  if(!m_run)
    return;

  QMutexLocker ml(&raw_mutex);
  m_color.copyTo(color);
  m_disparity.copyTo(disparity);
}

void Sensor::get_undistorted(cv::Mat& color, cv::Mat& points)
{
  if(!m_run)
    return;

  QMutexLocker ml(&undistorted_mutex);
  m_color_undistorted.copyTo(color);
  m_points.copyTo(points);
}

void Sensor::cam_params_changed(const CDepthColorCam& cam)
{
  m_cam = cam;
}

void Sensor::run()
{
  CDepthColorSensor sensor(m_cam);
  if(sensor.OpenDevice(m_cam_id))
  {
    cout << "[Sensor::run()] Error, can not start capturing device " << m_cam_id << endl;
    return;
  }

  cout << "[Sensor::run()] Start broadcasting" << endl;

  m_run = true;
  while(m_run)
  {
    {
      {
        QMutexLocker ml(&raw_mutex);
        m_color = sensor.GetRGB();
        m_disparity = sensor.GetDisparity();
      }
      {
        QMutexLocker ml(&undistorted_mutex);
        m_color_undistorted = sensor.GetCam().m_rgb_cam.GetUndistorted(m_color);
        m_points = sensor.GetCam().WarpDisparityToRGBPointsUndistorted(m_disparity, m_color);
      }
    }

    emit new_image(m_color_undistorted, m_points);
  }

  cout << "[Sensor::run()] Stop broadcasting" << endl;
  sensor.CloseDevice();
  cout << "[Sensor::run()] stopped" << endl;
}
