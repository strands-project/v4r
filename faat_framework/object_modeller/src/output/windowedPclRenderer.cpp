#include "output/windowedPclRenderer.h"

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

namespace object_modeller
{
namespace output
{

void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void)
{
    if (event.keyDown())
    {
        WindowedPclRenderer *renderer = static_cast<WindowedPclRenderer*> (viewer_void);
        std::cout << event.getKeySym() << std::endl;

        if (event.getKeySym () == "space") renderer->trigger(EventManager::STEP);
        if (event.getKeySym () == "Right") renderer->trigger(EventManager::NEXT_OBJECT);
        if (event.getKeySym () == "Left") renderer->trigger(EventManager::PREVIOUS_OBJECT);
        if (event.getKeySym () == "Up") renderer->trigger(EventManager::NEXT_SEQUENCE);
        if (event.getKeySym () == "Down") renderer->trigger(EventManager::PREVIOUS_SEQUENCE);
        if (event.getKeySym () == "e") renderer->trigger(EventManager::RUN);
        if (event.getKeySym () == "p") renderer->trigger(EventManager::PAUSE);
        if (event.getKeySym () == "Escape") renderer->trigger(EventManager::CLOSE);
        if (event.getKeySym () == "g") renderer->trigger(EventManager::GRAB_FRAME);
        if (event.getKeySym () == "c") renderer->trigger(EventManager::CONTINUE);
        if (event.getKeySym () == "n") renderer->trigger(EventManager::ADD_SEQUENCE);
        if (event.getKeySym () == "F1") renderer->trigger(EventManager::TOGGLE_HELP);
        if (event.getKeySym () == "F2") renderer->trigger(EventManager::START_RECORDING);
        if (event.getKeySym () == "F3") renderer->trigger(EventManager::END_RECORDING);
    }
}

WindowedPclRenderer::WindowedPclRenderer() : PclRenderer(true) {
    //vis->registerKeyboardCallback(&keyboardEventOccurred, this);
    vis->registerKeyboardCallback(keyboardEventOccurred, (void*) this);

    show_help = true;
    m_updateRenderer = true;
    m_updateImage = false;
    m_quit = false;
    imgViewer.reset();
}

void WindowedPclRenderer::loop()
{
    while (!m_quit)
    {
        vis->spinOnce(100);

        if (m_updateRenderer)
        {
            std::cout << " EXIT LOOP " << std::endl;
            update();
            updateImage();
            updateRenderControls();
            m_updateRenderer = false;
        }

        if (m_updateImage)
        {
            m_updateImage = false;

            updateImage();
        }

        if (m_updateRenderControls)
        {
            m_updateRenderControls = false;

            updateRenderControls();
        }
    }
}

void WindowedPclRenderer::updateImage()
{
    if (image != NULL)
    {
        if (!imgViewer)
        {
            imgViewer.reset(new pcl::visualization::ImageViewer("Camera"));
        }

        imgViewer->removeLayer("img");
        imgViewer->showRGBImage(image->data, image->width, image->height, "img");
    }
    else
    {
        if (imgViewer)
        {
            imgViewer->close();
            imgViewer.reset();
        }
    }
}

void WindowedPclRenderer::updateRenderControls()
{
    vis->removeShape("help");
    vis->removeShape("processStep");
    vis->removeShape("sequenceText");
    vis->removeShape("objectText");

    if (show_help)
    {
        std::string help_text;

        if (isEventAvailable(EventManager::TOGGLE_HELP)) help_text.append("F1 - Toggle Help\n");
        if (isEventAvailable(EventManager::PREVIOUS_OBJECT)) help_text.append("Left Arrow - Previous Object\n");
        if (isEventAvailable(EventManager::NEXT_OBJECT)) help_text.append("Right Arrow - Next Object\n");
        if (isEventAvailable(EventManager::PREVIOUS_SEQUENCE)) help_text.append("Down Arrow - Previous Sequence\n");
        if (isEventAvailable(EventManager::NEXT_SEQUENCE)) help_text.append("Up Arrow - Next Sequence\n");
        if (isEventAvailable(EventManager::STEP)) help_text.append("Space - Process one step\n");
        if (isEventAvailable(EventManager::RUN)) help_text.append("E - Process\n");
        if (isEventAvailable(EventManager::CONTINUE)) help_text.append("C - Continue\n");
        if (isEventAvailable(EventManager::ADD_SEQUENCE)) help_text.append("N - Add Sequence\n");
        if (isEventAvailable(EventManager::GRAB_FRAME)) help_text.append("G - Snapshot\n");
        if (isEventAvailable(EventManager::START_RECORDING)) help_text.append("F2 - Start Recording\n");
        if (isEventAvailable(EventManager::END_RECORDING)) help_text.append("F3 - End Recording\n");
        if (isEventAvailable(EventManager::PAUSE)) help_text.append("P - Pause\n");
        if (isEventAvailable(EventManager::CLOSE)) help_text.append("Esc - Quit\n");

        vis->addText(help_text, 10, 10, "help");

    }

    if (getActiveStepName() != "")
    {
        std::string renderStep("Processing module: ");
        renderStep.append(getActiveStepName());

        vis->addText(renderStep, 10, 100, "processStep");
    }



    // update sequence text
    int nrSequences = getNrSequences();
    int sequenceId = getActiveSequenceId();

    if (nrSequences > 0)
    {
        std::string sequenceText("Sequence ");
        sequenceText.append(boost::lexical_cast<std::string>(sequenceId + 1));
        sequenceText.append(" / ");
        sequenceText.append(boost::lexical_cast<std::string>(nrSequences));
        vis->addText(sequenceText, 10, 200, "sequenceText");

        // update object selection text
        int objectId = getActiveObjectId();
        int nrObjects = getNrObjects(sequenceId);
        std::string objectText;

        if (objectId == -1)
        {
            objectText.append("All Objects");
        }
        else
        {
            objectText.append("Object ");
            objectText.append(boost::lexical_cast<std::string>(objectId + 1));
            objectText.append(" / ");
            objectText.append(boost::lexical_cast<std::string>(nrObjects));
        }

        vis->addText(objectText, 10, 300, "objectText");
    }
}


bool WindowedPclRenderer::onEvent(Event e)
{
    if (e == EventManager::TOGGLE_HELP)
    {
        show_help = !show_help;

        m_updateRenderControls = true;

        return true;
    }

    if (e == EventManager::UPDATE_RENDERER)
    {
        std::cout << "update windowed renderer" << std::endl;
        m_updateRenderer = true;
        return true;
    }


    if (e == EventManager::UPDATE_IMAGE)
    {
        m_updateImage = true;
        return true;
    }


    if (e == EventManager::UPDATE_RENDERCONTROLS)
    {
        m_updateRenderControls = true;
        return true;
    }

    if (e == EventManager::CLOSE)
    {
        m_quit = true;
    }

    return PclRenderer::onEvent(e);
}

/*
void PclRenderer::renderTexturedMesh(TexturedMesh mesh, std::string text, bool step)
{
    renderMesh(mesh.mesh, text, step);
}

void PclRenderer::renderMesh(pcl::PolygonMesh::Ptr mesh, std::string text, bool step)
{
    int v;
    vis->createViewPort(0,0,1,1,v);
    vis->removeAllPointClouds();
    vis->removeAllShapes();

    vis->addPolygonMesh(*mesh, text, v);

    if (step)
    {
        vis->spin();
    }
    else
    {
        vis->spinOnce();
    }
}
*/

}
}
