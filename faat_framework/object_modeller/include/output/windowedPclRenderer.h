
#include "outputModule.h"
#include "module.h"
#include "pclRenderer.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/image_viewer.h>

namespace object_modeller
{
namespace output
{

class WindowedPclRenderer : public PclRenderer
{
private:
    bool show_help;
    bool m_updateRenderer;
    bool m_updateRenderControls;
    bool m_updateImage;
    bool m_quit;

    pcl::visualization::ImageViewer::Ptr imgViewer;
public:
    WindowedPclRenderer();

    void loop();

    virtual bool onEvent(Event e);

    void updateRenderControls();

    vtkSmartPointer<vtkRenderWindow> getRenderWindow()
    {
        return vis->getRenderWindow();
    }

    virtual void updateImage();

    std::string getName()
    {
        return "Pcl Renderer";
    }
};

}
}
