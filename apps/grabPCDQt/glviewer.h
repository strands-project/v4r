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
#ifndef _GRAB_PCD_GL_VIEWER_H_
#define _GRAB_PCD_GL_VIEWER_H_

#include <qgraphicswidget.h>
#include <QGLWidget>
#include <QMouseEvent>
#include <QTimer>
#include <QElapsedTimer>
#include <QThread>

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "v4r/TomGine5/Camera.h"
#include "v4r/CalibrationRGBD/dsensor.h"

#include <QtGui/QGraphicsView>
class GLGraphicsView : public QGraphicsView
{
  Q_OBJECT

signals:
  void mouse_moved(QMouseEvent *event);
  void mouse_pressed(QMouseEvent *event);
  void key_pressed(QKeyEvent *event);
  void wheel_event(QWheelEvent *event);

public:
  GLGraphicsView(QWidget* widget=0) : QGraphicsView(widget) { }

  void mouseMoveEvent(QMouseEvent *event)
  {
    emit mouse_moved(event);
  }

  void mousePressEvent(QMouseEvent *event)
  {
    emit mouse_pressed(event);
  }

  void keyPressEvent(QKeyEvent *event)
  {
    emit key_pressed(event);
  }

  void wheelEvent(QWheelEvent *event)
  {
    emit wheel_event(event);
  }


};

class GLViewer : public QGLWidget
{
  Q_OBJECT

public:
  //! Default constructor.
  GLViewer(QWidget* _parent=0);

  //! Destructor.
  virtual ~GLViewer();

  void show_color(bool enable);
  void show_points(bool enable);
  void reset_view();

signals:

public slots:
  void draw();

  void new_image(cv::Mat color, cv::Mat points);

  void cam_params_changed(const CDepthColorCam& cam);

  void mouse_moved(QMouseEvent *event);
  void mouse_pressed(QMouseEvent *event);
  void key_pressed(QKeyEvent *event);
  void wheel_event(QWheelEvent *event);

private:

  //! Initializes OpenGL states (triggered by Qt).
  void initializeGL();

  //! Draws a coordinate frame at the origin (0,0,0).
  void drawCoordinates(float length=1.0);

  //! Grabs an Image and draws it.
  void drawImage();

  //! Draws the scene (triggered by Qt).
  void paintGL();

  //! Handle resize events (triggered by Qt).
  void resizeGL(int w, int h);

  tg::Camera m_cam_origin;
  tg::Camera m_cam_perspective;
  tg::Camera m_cam_ortho;

  CDepthColorCam m_dcam;

  size_t m_width;
  size_t m_height;

  QPoint m_last_point_2d;
  QElapsedTimer m_elapsed;
  QTimer m_timer;

  cv::Mat m_color;
  cv::Mat m_points;

  bool m_show_color;
  bool m_show_points;

  GLuint m_texture_id;

protected:

  // Qt mouse events
  virtual void mousePressEvent(QMouseEvent* event);
  virtual void mouseMoveEvent(QMouseEvent* event);
  virtual void wheelEvent(QWheelEvent* event);
  virtual void keyPressEvent(QKeyEvent *event);

};



#endif // GLWIDGET_H
