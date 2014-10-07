#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QGLWidget>

class GlWidget : public QGLWidget
{
    Q_OBJECT
public:
    explicit GlWidget(QWidget *parent = 0);

    QSize minimumSizeHint() const;
    QSize sizeHint() const;

protected:
    void initializeGL();
    void paintGL();
    void resizeGL(int w, int h);
    
signals:
    
public slots:
    
};

#endif // GLWIDGET_H
