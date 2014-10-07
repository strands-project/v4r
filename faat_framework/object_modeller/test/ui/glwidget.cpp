#include "glwidget.h"

#include <v4r/TomGine/tgTomGineThread.h>

GlWidget::GlWidget(QWidget *parent) :
    QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
}

void GlWidget::initializeGL()
{
    qglClearColor(QColor::fromCmykF(0.00, 0.5, 0.5, 0.0));

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glShadeModel(GL_SMOOTH);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_MULTISAMPLE);
    static GLfloat lightPosition[4] = { 0.5, 5.0, 7.0, 1.0 };
    glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
}

void GlWidget::paintGL()
{
}

void GlWidget::resizeGL(int width, int height)
{
    int side = qMin(width, height);
    glViewport((width - side) / 2, (height - side) / 2, side, side);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-0.5, 0.5, -0.5, 0.5, 4.0, 15.0);
    glMatrixMode(GL_MODELVIEW);
}

QSize GlWidget::minimumSizeHint() const
{
    return QSize(50, 50);
}

QSize GlWidget::sizeHint() const
{
    return QSize(400, 400);
}
