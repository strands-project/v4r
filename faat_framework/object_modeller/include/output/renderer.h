#pragma once

#include "outputModule.h"
#include "module.h"
#include "rendererArgs.h"
#include "eventManager.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace object_modeller
{
namespace output
{

struct TexturedMesh
{
    pcl::PolygonMesh::Ptr mesh;
    std::vector<cv::Mat3b> textures;
    std::vector<Eigen::Vector2f> textureCoordinates;
    std::vector<int> textureIndex;
    std::vector<int> textureIndex2;

    bool hasTexture()
    {
        return textures.size() > 0;
    }

    typedef boost::shared_ptr<TexturedMesh> Ptr;
};

class Roi;

struct ImageData
{
    unsigned char *data;
    int width;
    int height;
};

class Renderer : public EventManager
{
public:
    typedef boost::shared_ptr<Renderer> Ptr;
protected:
    std::map<int, std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > pointclouds_rgb;
    std::map<int, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> > pointclouds_rgbnormal;
    std::map<int, std::vector<TexturedMesh::Ptr> > textured_meshs;
    std::map<int, std::vector<pcl::PolygonMesh::Ptr> > meshs;
    ImageData *image;
    Roi *roi;

    std::string renderName;
    std::string m_activeStepName;

    int nrSequences;
    int activeSequence;
    int activeObject;

public:

    Renderer()
    {
        availableEvents.insert(EventManager::NEXT_OBJECT);
        availableEvents.insert(EventManager::PREVIOUS_OBJECT);
        availableEvents.insert(EventManager::NEXT_SEQUENCE);
        availableEvents.insert(EventManager::PREVIOUS_SEQUENCE);
        availableEvents.insert(EventManager::UPDATE_RENDERER);
        availableEvents.insert(EventManager::UPDATE_RENDERCONTROLS);
        availableEvents.insert(EventManager::UPDATE_IMAGE);
        //availableEvents.insert(EventManager::UPDATE_COMPLETE);
        availableEvents.insert(EventManager::CLOSE);

        m_activeStepName = "";

        clear();
    }

    std::string getActiveStepName()
    {
        return m_activeStepName;
    }

    virtual void loop()
    {

    }

    virtual bool onEvent(Event e)
    {
        std::cout << "on event" << std::endl;
        if (e == NEXT_OBJECT)
        {
            nextObject();
            trigger(EventManager::UPDATE_RENDERER);
            return true;
        }

        if (e == PREVIOUS_OBJECT)
        {
            previousObject();
            trigger(EventManager::UPDATE_RENDERER);
            return true;
        }

        if (e == NEXT_SEQUENCE)
        {
            nextSequence();
            trigger(EventManager::UPDATE_RENDERER);
            return true;
        }

        if (e == PREVIOUS_SEQUENCE)
        {
            previousSequence();
            trigger(EventManager::UPDATE_RENDERER);
            return true;
        }

        if (e == CONTINUE)
        {
            removeAvailableEvent(CONTINUE);
            disableRoiMode();
        }

        if (e == RESET)
        {
            clear();
        }


        if (e == PAUSE)
        {
            removeAvailableEvent(PAUSE);
        }

        return EventManager::onEvent(e);
    }

    bool hasImage()
    {
        return image != NULL;
    }

    void clearImage()
    {
        image = NULL;
    }

    void clear()
    {
        activeSequence = 0;
        activeObject = -1;
        pointclouds_rgb.clear();
        pointclouds_rgbnormal.clear();
        textured_meshs.clear();
        meshs.clear();
        image = NULL;

        updateNavEvents();
    }

    void addPointCloud(int sequenceId, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointclouds)
    {
        pointclouds_rgb[sequenceId].push_back(pointclouds);

        updateNavEvents();
    }

    void addPointClouds(int sequenceId, std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointclouds)
    {
        for (int i=0;i<pointclouds.size();i++)
        {
            pointclouds_rgb[sequenceId].push_back(pointclouds[i]);
        }

        updateNavEvents();
    }

    void addPointCloud(int sequenceId, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pointclouds)
    {
        pointclouds_rgbnormal[sequenceId].push_back(pointclouds);

        updateNavEvents();
    }

    void addPointClouds(int sequenceId, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> pointclouds)
    {
        for (int i=0;i<pointclouds.size();i++)
        {
            pointclouds_rgbnormal[sequenceId].push_back(pointclouds[i]);
        }

        updateNavEvents();
    }

    void addTexturedMesh(int sequenceId, TexturedMesh::Ptr mesh)
    {
        textured_meshs[sequenceId].push_back(mesh);


        ImageData *data = new ImageData();
        data->width = mesh->textures[0].cols;
        data->height = mesh->textures[0].rows;
        data->data = mesh->textures[0].data;

        this->addImage(data);

        updateNavEvents();
    }

    void addMesh(int sequenceId, pcl::PolygonMesh::Ptr mesh)
    {
       meshs[sequenceId].push_back(mesh);

       updateNavEvents();
    }

    virtual std::string getName() = 0;

    virtual void addImage(ImageData *image)
    {
        this->image = image;
    }

    virtual void update() = 0;

    virtual void updateImage() = 0;

    std::string getRenderName()
    {
        return renderName;
    }

    void setRenderName(std::string name)
    {
        renderName = name;
    }

    virtual void enableRoiMode(Eigen::Vector3f *dim, Eigen::Vector3f *translation, Eigen::Quaternionf *rotation) = 0;

    virtual void disableRoiMode()
    {
        roi = NULL;
    }

    int getNrSequences()
    {
        int nrSequences = 0;

        nrSequences = std::max(nrSequences, (int) pointclouds_rgb.size());
        nrSequences = std::max(nrSequences, (int) pointclouds_rgbnormal.size());
        nrSequences = std::max(nrSequences, (int) meshs.size());
        nrSequences = std::max(nrSequences, (int) textured_meshs.size());

        return nrSequences;
    }

    int getActiveSequenceId()
    {
        return activeSequence;
    }


    int getActiveObjectId()
    {
        return activeObject;
    }

    void nextObject()
    {
        if (activeObject + 1 < getNrObjects(activeSequence))
        {
            activeObject++;
        }

        updateNavEvents();
    }

    void previousObject()
    {
        if (activeObject > -1)
        {
            activeObject--;
        }

        updateNavEvents();
    }

    void nextSequence()
    {
        if (activeSequence + 1 < getNrSequences())
        {
            activeSequence++;
        }

        updateNavEvents();
    }

    void previousSequence()
    {
        if (activeSequence > 0)
        {
            activeSequence--;
        }

        updateNavEvents();
    }

    void updateNavEvents()
    {
        if (activeObject + 1 == getNrObjects(activeSequence))
        {
            removeAvailableEvent(EventManager::NEXT_OBJECT);
        }
        else
        {
            availableEvents.insert(EventManager::NEXT_OBJECT);
        }

        if (activeObject == -1)
        {
            removeAvailableEvent(EventManager::PREVIOUS_OBJECT);
        }
        else
        {
            availableEvents.insert(EventManager::PREVIOUS_OBJECT);
        }

        if (activeSequence + 1 == getNrSequences())
        {
            removeAvailableEvent(EventManager::NEXT_SEQUENCE);
        }
        else
        {
            availableEvents.insert(EventManager::NEXT_SEQUENCE);
        }

        if (activeSequence == 0)
        {
            removeAvailableEvent(EventManager::PREVIOUS_SEQUENCE);
        }
        else
        {
            availableEvents.insert(EventManager::PREVIOUS_SEQUENCE);
        }
    }

    void pipelineStateChanged(State state, std::string activeStepName="")
    {
        m_activeStepName = activeStepName;

        if (state == EventManager::RUNNING)
        {
            removeAvailableEvent(EventManager::STEP);
            removeAvailableEvent(EventManager::RUN);
            removeAvailableEvent(EventManager::RESET);
            availableEvents.insert(EventManager::PAUSE);
        }
        else
        {
            removeAvailableEvent(EventManager::PAUSE);

            if (state != EventManager::INITIALIZED)
            {
                availableEvents.insert(EventManager::RESET);
            }
        }

        if (state == EventManager::PAUSED || state == EventManager::INITIALIZED)
        {
            availableEvents.insert(EventManager::STEP);
            availableEvents.insert(EventManager::RUN);

            //trigger(EventManager::UPDATE_RENDERER);
        }

        if (state == EventManager::CONFIGURE)
        {
            removeAvailableEvent(EventManager::STEP);
            removeAvailableEvent(EventManager::RUN);

            availableEvents.insert(EventManager::CONTINUE);

            //trigger(EventManager::UPDATE_RENDERER);
        }

        trigger(EventManager::UPDATE_RENDERCONTROLS);
    }

    int getNrObjects(int sequenceId)
    {
        int nrObjects = 0;

        if (pointclouds_rgb.count(sequenceId) > 0)
            nrObjects += pointclouds_rgb[sequenceId].size();

        if (pointclouds_rgbnormal.count(sequenceId) > 0)
            nrObjects += pointclouds_rgbnormal[sequenceId].size();

        if (meshs.count(sequenceId) > 0)
            nrObjects += meshs[sequenceId].size();

        if (textured_meshs.count(sequenceId) > 0)
            nrObjects += textured_meshs[sequenceId].size();

        return nrObjects;
    }

    //virtual void renderPointClouds(std::vector<typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr> point_clouds, std::string name, bool step) = 0;
    //virtual void renderPointClouds(std::vector<typename pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> point_clouds, std::string name, bool step) = 0;

    //virtual void renderMesh(pcl::PolygonMesh::Ptr, std::string, bool) = 0;
    //virtual void renderTexturedMesh(TexturedMesh, std::string, bool) = 0;
};

class RenderDelegateBase
{
public:
    virtual bool canRender()
    {
        return true;
    }
};

template<class T>
class RenderDelegate : public RenderDelegateBase
{
public:
    virtual bool canRender()
    {
        return false;
    }
    virtual void render(Renderer::Ptr renderer, int sequenceId, T result)
    {
        std::cout << "Render called for result which cannot be rendered" << std::endl;
    }
};

template<>
class RenderDelegate<std::vector<typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr> > : public RenderDelegateBase
{
public:
    virtual void render(Renderer::Ptr renderer, int sequenceId, std::vector<typename pcl::PointCloud<pcl::PointXYZRGB>::Ptr> result)
    {
        return renderer->addPointClouds(sequenceId, result);
    }
};

template<>
class RenderDelegate<std::vector<typename pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> > : public RenderDelegateBase
{
public:
    virtual void render(Renderer::Ptr renderer, int sequenceId, std::vector<typename pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> result)
    {
        return renderer->addPointClouds(sequenceId, result);
    }
};

template<>
class RenderDelegate<pcl::PolygonMesh::Ptr> : public RenderDelegateBase
{
public:
    virtual void render(Renderer::Ptr renderer, int sequenceId, pcl::PolygonMesh::Ptr result)
    {
        return renderer->addMesh(sequenceId, result);
    }
};


template<>
class RenderDelegate<TexturedMesh::Ptr> : public RenderDelegateBase
{
public:
    virtual void render(Renderer::Ptr renderer, int sequenceId, TexturedMesh::Ptr result)
    {
        if (result->hasTexture())
        {
            return renderer->addTexturedMesh(sequenceId, result);
        }

        return renderer->addMesh(sequenceId, result->mesh);
    }
};

}
}
