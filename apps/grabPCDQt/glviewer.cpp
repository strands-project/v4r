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
#include "glviewer.h"

using namespace std;

GLViewer::GLViewer(QWidget* _parent)
  : QGLWidget(_parent),
    m_timer(this),
    m_show_color(true),
    m_show_points(true)
{

  setAttribute(Qt::WA_NoSystemBackground,true);
  setFocusPolicy(Qt::StrongFocus);
  setAcceptDrops(true);
  setCursor(Qt::PointingHandCursor);

  connect(&m_timer, SIGNAL(timeout()), this, SLOT(draw()));
  m_timer.start(200);
  //  m_elapsed.start();
}

GLViewer::~GLViewer()
{

}

void GLViewer::show_color(bool enable)
{
  m_show_color = enable;
}

void GLViewer::show_points(bool enable)
{
  m_show_points = enable;
}

void GLViewer::reset_view()
{
  m_cam_perspective = m_cam_origin;
  updateGL();
}

void GLViewer::new_image(cv::Mat color, cv::Mat points)
{
  m_color = color; // TODO: don't know if this is save
  m_points = points;
  //  cv::Mat mat(img);
  //  mat.copyTo(m_image);


  updateGL();
}

void GLViewer::draw()
{
  updateGL();
}

void GLViewer::cam_params_changed(const CDepthColorCam &cam)
{
  m_dcam = cam;

  size_t size[2];
  float f[2];
  float c[2];
  float alpha;
  float k[5];
  float range[2];

  m_dcam.m_rgb_cam.GetIntrinsics(size, f, c, alpha, k);
  m_dcam.m_depth_cam.GetRange(range);
  cv::Mat ext = m_dcam.m_depth_cam.GetExtrinsics();

  m_width = size[0];
  m_height = size[1];

  this->setGeometry(0,0, m_width, m_height);

  glm::mat4 E;
  E[0][0]=ext.at<float>(0,0); E[0][1]=ext.at<float>(0,1); E[0][2]=ext.at<float>(0,2); E[0][3]=ext.at<float>(0,3);
  E[1][0]=ext.at<float>(1,0); E[1][1]=ext.at<float>(1,1); E[1][2]=ext.at<float>(1,2); E[1][3]=ext.at<float>(1,3);
  E[2][0]=ext.at<float>(2,0); E[2][1]=ext.at<float>(2,1); E[2][2]=ext.at<float>(2,2); E[2][3]=ext.at<float>(2,3);
  E[3][0]=ext.at<float>(3,0); E[3][1]=ext.at<float>(3,1); E[3][2]=ext.at<float>(3,2); E[3][3]=ext.at<float>(3,3);

  m_cam_perspective.SetPerspective(f[0],f[1], c[0],c[1], size[0],size[1], 0.5f*range[0],2.0f*range[1]);
  m_cam_perspective.SetExtrinsic(tg::Camera::cv2gl(E));
  m_cam_origin = m_cam_perspective;
  m_cam_ortho.SetOrtho(size[0], size[1], 0.1f, 2.0f);
  updateGL();
}

void GLViewer::mouse_moved(QMouseEvent *event)
{
  this->mouseMoveEvent(event);
}

void GLViewer::mouse_pressed(QMouseEvent *event)
{
  this->mousePressEvent(event);
}

void GLViewer::key_pressed(QKeyEvent *event)
{
  this->keyPressEvent(event);
}


void GLViewer::wheel_event(QWheelEvent *event)
{
  this->wheelEvent(event);
}



void GLViewer::initializeGL()
{ 
  glGenTextures(1, &m_texture_id);
  glBindTexture(GL_TEXTURE_2D, m_texture_id);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);

  glClearColor(0.5,0.5,0.5,0);
}

void GLViewer::resizeGL(int w, int h)
{
  cout << "[GLViewer::resizeGL] w,h: " << width() << ", " << height() << endl;
  glViewport(0,0,width(),height());
  updateGL();
}


void GLViewer::drawCoordinates(float length)
{
  m_cam_perspective.Activate();

  glDisable(GL_LIGHTING);
  glLineWidth(2.0);
  glBegin(GL_LINES);
  glColor3ub(255,0,0); glVertex3f(0.0f,0.0f,0.0f); glVertex3f(length,0.0f,0.0f);
  glColor3ub(0,255,0); glVertex3f(0.0f,0.0f,0.0f); glVertex3f(0.0f,length,0.0f);
  glColor3ub(0,0,255); glVertex3f(0.0f,0.0f,0.0f); glVertex3f(0.0f,0.0f,length);
  glEnd();
}


void GLViewer::drawImage()
{
  if(m_show_color && !m_color.empty())
  {
    m_cam_ortho.Activate();

    glBindTexture(GL_TEXTURE_2D, m_texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_color.cols, m_color.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, m_color.data);

    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    float w = float(m_width);
    float h = float(m_height);
    glColor3f(1,1,1);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 1.0f);
    glVertex3f(0, 0, 0.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex3f(w, 0, 0.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex3f(w, h, 0.0f);
    glTexCoord2f(0.0f, 0.0f);
    glVertex3f(0, h, 0.0f);
    glEnd();

    glEnable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

    glBindTexture(GL_TEXTURE_2D, 0);
  }

  if(m_show_points && !m_points.empty())
  {
    //    cout << "[drawImage] p: " << m_points.cols << ", " << m_points.rows << endl;

    m_cam_perspective.Activate();

    glDisable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);

    glPointSize(1.0f);

    glBegin(GL_POINTS);
    for(int i=0; i<m_points.rows; i++)
    {
      for(int j=0; j<m_points.cols; j++)
      {
        const cv::Point3f& p = m_points.at<cv::Point3f>(i,j);
        const cv::Vec3b& c = m_color.at<cv::Vec3b>(i,j);
        glColor3ub(c[0],c[1],c[2]);
        glVertex3f(p.x,p.y,p.z);
      }
    }
    glEnd();

    glEnable(GL_LIGHTING);

  }
}


void GLViewer::paintGL()
{
  // enable GL context
  makeCurrent();

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  drawImage();

  drawCoordinates(0.25f);
}


void GLViewer::mousePressEvent(QMouseEvent* event)
{
  //  cout << "GLWidget::mousePressEvent" << endl;
  m_last_point_2d = event->pos();
  updateGL();
}

void GLViewer::mouseMoveEvent(QMouseEvent* event)
{
  // enable GL context
  makeCurrent();

  QPoint newPoint2D = event->pos();

  float dx = newPoint2D.x() - m_last_point_2d.x();
  float dy = newPoint2D.y() - m_last_point_2d.y();

  float far = m_cam_perspective.GetFar();
  float near = m_cam_perspective.GetNear();

  // move in z direction
  if ( (event->buttons() == Qt::MidButton) )
  {
    m_cam_perspective.TranslateForward(0.001f * (far - near) * dx);
    m_cam_perspective.TranslateForward(0.001f * (far - near) * dy);
  }  // move in x,y direction
  else if ( (event->buttons() == Qt::RightButton) )
  {
    m_cam_perspective.TranslateSideward(0.0001f * (far - near) * dx);
    m_cam_perspective.TranslateUpward(-0.0001f * (far - near) * dy);
  } // rotate
  else if (event->buttons() == Qt::LeftButton)
  {
    glm::vec3 cor(0, 0, 0);
    m_cam_perspective.Orbit(cor, m_cam_perspective.GetUpward(), -0.05f * dx);
    m_cam_perspective.Orbit(cor, m_cam_perspective.GetSideward(), -0.05f * dy);
  }

  // remember this point
  m_last_point_2d = newPoint2D;

  // trigger redraw
  updateGL();
}

void GLViewer::wheelEvent(QWheelEvent* event)
{
  float d = -(float)event->delta() / 120.0;

  float far = m_cam_perspective.GetFar();
  float near = m_cam_perspective.GetNear();
  m_cam_perspective.TranslateForward(0.001f * (far - near) * d);
  m_cam_perspective.TranslateForward(0.001f * (far - near) * d);

  updateGL();
  event->accept();
}

void GLViewer::keyPressEvent(QKeyEvent *event)
{
  if(event->key() == Qt::Key_Z)
  {
    m_cam_perspective = m_cam_origin;
    m_cam_perspective.Activate();
  }
  if(event->key() == Qt::Key_O)
  {
    m_cam_perspective.Print();
    std::cout << std::endl;
  }
  updateGL();
}

