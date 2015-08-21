/**
 * $Id$
 *
 * Software License Agreement (GNU General Public License)
 *
 *  Copyright (c) 2014, Thomas Mörwald, Johann Prankl
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
 * @author thomas.moerwald, Johann Prankl (prankl@acin.tuwien.ac.at)
 */

#include "glviewer.h"

using namespace std;

GLViewer::GLViewer(QWidget* _parent)
  : QGLWidget(_parent),
    m_width(640),
    m_height(480),
    m_point_size(1.),
    m_timer(this),
    bbox_pose(Eigen::Matrix4f::Identity()),
    m_show_image(true),
    m_show_cloud(false),
    m_show_cameras(false),
    m_show_object(false),
    m_draw_bbox(false),
    m_segment_object(false),
    m_select_roi(false),
    cor(glm::vec3(0.,0.,0.))
{
  setAttribute(Qt::WA_NoSystemBackground,true);
  setFocusPolicy(Qt::StrongFocus);
  setAcceptDrops(true);
  setCursor(Qt::PointingHandCursor);

  connect(&m_timer, SIGNAL(timeout()), this, SLOT(draw()));
  m_timer.start(200);
  //  m_elapsed.start();

  pt00=Eigen::Vector4f(0.,0.,0.,1.);
  pt0x=Eigen::Vector4f(1.,0.,0.,1.);
  pt0y=Eigen::Vector4f(0.,1.,0.,1.);
  pt0z=Eigen::Vector4f(0.,0.,1.,1.);

  cam_params_changed(cam_params);
}

GLViewer::~GLViewer()
{

}

void GLViewer::resetView(float fw)
{
  m_cam_perspective = m_cam_origin;
  if (fabs(fw)>0.001) {m_cam_perspective.TranslateForward(fw);}
  updateGL();
}

void GLViewer::showImage(bool enable)
{
  m_show_image = enable;
}

void GLViewer::showCloud(bool enable)
{
  m_show_cloud = enable;
}

void GLViewer::showCameras(bool enable)
{
  m_show_cameras = enable;
  //cout<<"[GLViewer::showCameras] "<<m_show_cameras<<endl;
}

void GLViewer::showObject(bool enable)
{
  m_show_object = enable;
  //cout<<"[GLViewer::showObject] "<<m_show_object<<endl;
}

void GLViewer::drawBoundingBox(bool enable)
{
  m_draw_bbox = enable;
}

void GLViewer::segmentObject(bool enable)
{
  m_segment_object = enable;
}

void GLViewer::selectROI(bool enable)
{
  m_select_roi = enable;
}

void GLViewer::new_image(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_cloud, const cv::Mat_<cv::Vec3b> &_image)
{
  m_cloud = _cloud;
  m_image = _image;
}

void GLViewer::update_model_cloud(const boost::shared_ptr< Sensor::AlignedPointXYZRGBVector > &_oc_cloud)
{
  oc_mutex.lock();
  oc_cloud = *_oc_cloud;
  oc_mutex.unlock();
  //cout<<"[GLViewer::update_model_cloud] "<<oc_cloud.size()<<endl;
}

void GLViewer::update_cam_trajectory(const boost::shared_ptr< std::vector<Sensor::CameraLocation> > &_cam_trajectory)
{
  cam_mutex.lock();
  cam_trajectory = *_cam_trajectory;
  cam_mutex.unlock();
}

void GLViewer::update_boundingbox(const std::vector<Eigen::Vector3f> &edges, const Eigen::Matrix4f &pose)
{
  Eigen::Matrix3f R = pose.topLeftCorner<3,3>(0,0);
  Eigen::Vector3f t = pose.block<3,1>(0,3);

  bb_mutex.lock();
  bbox.resize(edges.size());
  for (unsigned i=0; i<bbox.size(); i++)
    bbox[i] = R*edges[i]+t;
  bbox_pose = pose;
  bb_mutex.unlock();
}

void GLViewer::update_visualization()
{
  updateGL();
}

void GLViewer::draw()
{
  updateGL();
}

void GLViewer::cam_params_changed(const RGBDCameraParameter &_cam_params)
{
  cam_params = _cam_params;

  size_t size[2];
  float f[2];
  float c[2];
  float range[2];

  cv::Mat_<float> ext = cv::Mat_<float>::eye(4,4);
  m_width = size[0] = cam_params.width;
  m_height = size[1] = cam_params.height;
  f[0] = cam_params.f[0]; f[1] = cam_params.f[1];
  c[0] = cam_params.c[0]; c[1] = cam_params.c[1];
  range[0] = cam_params.range[0]; range[1] = cam_params.range[1];
  
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
  //cout << "[GLViewer::resizeGL] w,h: " << width() << ", " << height() << endl;
  glViewport(0,0,width(),height());
  updateGL();
}


void GLViewer::drawCoordinates(float length, const Eigen::Matrix4f &pose)
{
  pt10 = pose*pt00;
  pt1x = pose*pt0x;
  pt1y = pose*pt0y;
  pt1z = pose*pt0z;

  m_cam_perspective.Activate();

  glDisable(GL_LIGHTING);
  glLineWidth(2.0);
  glBegin(GL_LINES);
  glColor3ub(255,0,0); glVertex3f(pt10[0],pt10[1],pt10[2]); glVertex3f(pt1x[0],pt1x[1],pt1x[2]);
  glColor3ub(0,255,0); glVertex3f(pt10[0],pt10[1],pt10[2]); glVertex3f(pt1y[0],pt1y[1],pt1y[2]);
  glColor3ub(0,0,255); glVertex3f(pt10[0],pt10[1],pt10[2]); glVertex3f(pt1z[0],pt1z[1],pt1z[2]);
  glEnd();
}


void GLViewer::drawImage()
{
  // draw image
  if(m_show_image && !m_image.empty())
  {
    m_cam_ortho.Activate();

    glBindTexture(GL_TEXTURE_2D, m_texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_image.cols, m_image.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, m_image.data);

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

  // draw point cloud
  if(m_show_cloud && m_cloud.get()!=0)
  {
    m_cam_perspective.Activate();

    glDisable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
    glPointSize(m_point_size);

    const pcl::PointCloud<pcl::PointXYZRGB> &_cloud = *m_cloud;

    glBegin(GL_POINTS);
    for (unsigned i=0; i<_cloud.points.size(); i++)
    {
      const pcl::PointXYZRGB &pt = _cloud.points[i];
      if (!std::isnan(pt.x))
      {
        glColor3ub(pt.r,pt.g,pt.b);
        glVertex3f(pt.x,pt.y,pt.z);
      }
    }
    glEnd();

    glEnable(GL_LIGHTING);
  }

  // draw object model
  if(m_show_object)
  {
    oc_mutex.lock();
    m_cam_perspective.Activate();

    glDisable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
    glPointSize(m_point_size);

    const Sensor::AlignedPointXYZRGBVector &oc = oc_cloud;
    oc_center.setZero();
    unsigned cnt=0;

    glBegin(GL_POINTS);
    for (unsigned i=0; i<oc.size(); i++)
    {
      const pcl::PointXYZRGB &pt = oc[i];
      if (!std::isnan(pt.x))
      {
        glColor3ub(pt.r,pt.g,pt.b);
        glVertex3f(pt.x,pt.y,pt.z);
        oc_center += pt.getVector3fMap();
        cnt++;
      }
    }
    glEnd();

    glEnable(GL_LIGHTING);
    oc_mutex.unlock();

    oc_center/=float(cnt);
    cor = glm::vec3(oc_center[0],oc_center[1],oc_center[2]);
  }

  // draw camera trajectory
  if(m_show_cameras)
  {
    cam_mutex.lock();
    Eigen::Vector3f pt;
    m_cam_perspective.Activate();

    glDisable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);

    const std::vector<Sensor::CameraLocation> &traj = cam_trajectory;

    glLineWidth(1);

    glBegin(GL_LINE_STRIP);
    glColor3f(.8,.8,.8);
    for (unsigned i=0; i<traj.size(); i++)
    {
      const Sensor::CameraLocation &t0 = traj[i];
      glVertex3f(t0.pt[0], t0.pt[1], t0.pt[2]);
    }
    glEnd();


    for (unsigned i=0; i<traj.size(); i++)
    {
      const Sensor::CameraLocation &t = traj[i];

      if (t.type ==0)
      {
        glPointSize(2.0f);
        glBegin(GL_POINTS);
        glColor3ub(255,255,0);
        glVertex3f(t.pt[0], t.pt[1], t.pt[2]);
        glEnd();
      }
      else if (t.type==1)
      {
        glPointSize(2.0f);
        glBegin(GL_POINTS);
        glColor3ub(0,0,255);
        glVertex3f(t.pt[0], t.pt[1], t.pt[2]);
        glEnd();
      }
      else if (t.type==2)
      {
        pt = t.pt + 0.05*t.vr;

        glPointSize(6.0f);
        glBegin(GL_POINTS);
        glColor3ub(0,255,0);
        glVertex3f(t.pt[0], t.pt[1], t.pt[2]);
        glEnd();

        glBegin(GL_LINES);
        glColor3f(1.,1.,1.);
        glVertex3f(t.pt[0],t.pt[1],t.pt[2]);
        glVertex3f(pt[0],pt[1],pt[2]);
        glEnd();
      }
    }

    glEnable(GL_LIGHTING);
    cam_mutex.unlock();
  }

  if (m_draw_bbox)
  {
    drawCoordinates(0.5, bbox_pose);
    bb_mutex.lock();
    m_cam_perspective.Activate();

    glDisable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);

    glLineWidth(1);

    glBegin(GL_LINES);
    glColor3f(0.,0.,1.);
    for (unsigned i=0; i<bbox.size(); i++)
    {
      const Eigen::Vector3f &pt = bbox[i];
      glVertex3f(pt[0], pt[1], pt[2]);
    }
    glEnd();
    bb_mutex.unlock();
  }
}


void GLViewer::paintGL()
{
  // enable GL context
  makeCurrent();

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  drawImage();

  //drawCoordinates(0.25f);
}


void GLViewer::mousePressEvent(QMouseEvent* event)
{
  //  cout << "GLWidget::mousePressEvent" << endl;
  m_last_point_2d = event->pos();

  if (!m_show_cloud && m_segment_object)
  {
    emit segment_image(m_last_point_2d.x(),m_last_point_2d.y());
  }
  else if (m_select_roi)
  {
    emit select_roi(m_last_point_2d.x(),m_last_point_2d.y());
    m_select_roi = false;
  }

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
  if(event->key() == Qt::Key_Plus)
  {
    if (m_point_size<50) m_point_size+=1.;
  }
  if(event->key() == Qt::Key_Minus)
  {
    if (m_point_size>0) m_point_size-=1.;
  }

  updateGL();
}

