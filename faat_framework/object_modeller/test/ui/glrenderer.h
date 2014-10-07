#ifndef GLRENDERER_H
#define GLRENDERER_H

#include <QVTKWidget.h>
#include <QLabel>
#include <vtkRenderWindow.h>

#include "output/pclRenderer.h"

class MainWindow;

class QPclRenderer : public object_modeller::output::PclRenderer
{
private:
    QVTKWidget *widget;
    QLabel *img;
    MainWindow *mainWindow;
public:
    QPclRenderer(MainWindow *mainWindow, QLabel *img, QVTKWidget *widget);

    virtual bool onEvent(Event e);

    virtual void updateImage()
    {
        if (image != NULL)
        {
            //widget->setVisible(false);
            img->setVisible(true);


            QImage qimage(image->width, image->height, QImage::Format_RGB888);
            for(int y(0);y < image->height;++y)
            {
                for(int x(0);x < image->width;++x)
                {
                    int index((y*image->width + x) * 3);
                    qimage.setPixel(x, y, qRgb(image->data[index], image->data[index+1], image->data[index+2]));
                }
            }

            img->setPixmap(QPixmap::fromImage(qimage));
            //img->update();
        }
        else
        {
            img->setVisible(false);
        }
    }
};

/*
class GlRenderer : public object_modeller::output::Renderer//, public TomGine::tgTomGineThreadPCL
{
public:

    GlRenderer()// : tgTomGineThreadPCL(800, 600)
    {
        std::cout << "creating gl renderer" << std::endl;
        //StartEventListener(1);
    }

    virtual void applyConfig(object_modeller::Config &config, std::string base_path) {}

    virtual std::string getName()
    {
        return "OpenGL Renderer";
    }

    virtual Event waitForEvent()
    {
        return STEP;
    }

    virtual void renderImage(unsigned char *data, int width, int height) {}

    virtual void renderPointClouds(std::vector<typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr> point_clouds, std::string name, bool step) {

        for (unsigned k = 0; k < point_clouds.size(); k++)
        {
            TomGine::tgModel* tg_cloud = new TomGine::tgModel;
            tg_cloud->m_point_size = point_clouds[k]->size();

            for (size_t i = 0; i < point_clouds[k]->size (); i++)
            {
              const pcl::PointXYZRGB &pt = point_clouds[k]->at (i);
              TomGine::tgColorPoint cpt;
              cpt.color[0] = pt.r;
              cpt.color[1] = pt.g;
              cpt.color[2] = pt.b;
              cpt.pos = TomGine::vec3 (pt.x, pt.y, pt.z);
              tg_cloud->m_colorpoints.push_back (cpt);
            }

            tg_cloud->DrawColorPoints();
        }
    }
    virtual void renderPointClouds(std::vector<typename pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> point_clouds, std::string name, bool step) {}

    virtual void renderMesh(pcl::PolygonMesh::Ptr, std::string, bool) {}
    virtual void renderTexturedMesh(object_modeller::output::TexturedMesh, std::string, bool) {}
};
*/

#endif // GLRENDERER_H
