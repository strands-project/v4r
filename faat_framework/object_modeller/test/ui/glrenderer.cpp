#include "glrenderer.h"


#include "mainwindow.h"

QPclRenderer::QPclRenderer(MainWindow *mainWindow, QLabel *img, QVTKWidget *widget) : PclRenderer(false)
{
    this->mainWindow = mainWindow;
    this->img = img;
    this->widget = widget;
    vtkSmartPointer<vtkRenderWindow> localRenderWindow = vis->getRenderWindow();

    widget->SetRenderWindow(localRenderWindow);

    vis->setupInteractor (widget->GetInteractor (), widget->GetRenderWindow ());
    vis->getInteractorStyle()->setKeyboardModifier (pcl::visualization::INTERACTOR_KB_MOD_SHIFT);
    widget->update();
}

bool QPclRenderer::onEvent(Event e)
{
    if (e == EventManager::UPDATE_RENDERER)
    {
        mainWindow->updateRenderer();
        return true;
    }

    if (e == EventManager::UPDATE_IMAGE)
    {
        mainWindow->updateImage();
        return true;
    }

    if (e == EventManager::UPDATE_RENDERCONTROLS)
    {
        mainWindow->updateRenderControls();
        return true;
    }

    bool result = PclRenderer::onEvent(e);

    //mainWindow->updateRenderControls();
    //mainWindow->forceUpdateRenderer();

    return result;
}
