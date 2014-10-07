
#include "outputModule.h"
#include "module.h"
#include "rendererArgs.h"
#include "renderer.h"

#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/image_viewer.h>

#include "roi.h"

namespace object_modeller
{
namespace output
{

class Roi;
class PclRenderer;

class Interactor : public pcl::visualization::PCLVisualizerInteractorStyle
{
private:
    Roi *roi;
    PclRenderer *renderer;
public:
    Interactor()
    {
        roi = NULL;
        renderer = NULL;
    }

    void setRoi(Roi *roi)
    {
        this->roi = roi;
    }

    void setRenderer(PclRenderer *renderer)
    {
        this->renderer = renderer;
    }

    virtual void OnMouseMove();

    virtual void OnLeftButtonDown();

    virtual void OnLeftButtonUp();
};

/** \author Khai Tran */
  struct TexMaterial
  {
    TexMaterial () : tex_name (), tex_Ka (), tex_Kd (), tex_Ks (), tex_d (), tex_Ns (), tex_illum () {}

    struct RGB
    {
      float r;
      float g;
      float b;
    }; //RGB

    cv::Mat3b data;

    /** \brief Texture name. */
    std::string tex_name;

    /** \brief Texture file. */
    //std::string tex_file;

    /** \brief Defines the ambient color of the material to be (r,g,b). */
    RGB         tex_Ka;

    /** \brief Defines the diffuse color of the material to be (r,g,b). */
    RGB         tex_Kd;

    /** \brief Defines the specular color of the material to be (r,g,b). This color shows up in highlights. */
    RGB         tex_Ks;

    /** \brief Defines the transparency of the material to be alpha. */
    float       tex_d;

    /** \brief Defines the shininess of the material to be s. */
    float       tex_Ns;

    /** \brief Denotes the illumination model used by the material.
      *
      * illum = 1 indicates a flat material with no specular highlights, so the value of Ks is not used.
      * illum = 2 denotes the presence of specular highlights, and so a specification for Ks is required.
      */
    int         tex_illum;
  }; // TexMaterial

  /** \author Khai Tran */
  struct TextureMesh
  {
    TextureMesh () :
      cloud (), tex_polygons (), tex_coordinates (), tex_materials () {}

    pcl::PCLPointCloud2  cloud;
    pcl::PCLHeader  header;


    std::vector<std::vector<pcl::Vertices> >    tex_polygons;     // polygon which is mapped with specific texture defined in TexMaterial
    std::vector<std::vector<Eigen::Vector2f> >  tex_coordinates;  // UV coordinates
    std::vector<TexMaterial>               tex_materials;    // define texture material

    public:
      typedef boost::shared_ptr<TextureMesh> Ptr;
      typedef boost::shared_ptr<TextureMesh const> ConstPtr;
   }; // struct TextureMesh

class AdvancedPclVisualizer : public pcl::visualization::PCLVisualizer
{
public:
    AdvancedPclVisualizer(int &argc, char **argv, const std::string &name, pcl::visualization::PCLVisualizerInteractorStyle* style, const bool create_interactor)
        : PCLVisualizer (argc, argv, name, style, create_interactor) {}

    int textureFromTexMaterial (const TexMaterial& tex_mat, vtkTexture* vtk_tex) const;
    bool addTextureMesh (TexturedMesh::Ptr mesh, const std::string &id, int viewport);
    void addActorToRenderer2 (const vtkSmartPointer<vtkProp> &actor, int viewport);
private:
    bool addTextureMesh (const TextureMesh &mesh, const std::string &id, int viewport);
};

class PclRenderer : public Renderer
{
public:
    Interactor *interactor;

    Roi *getRoi()
    {
        return roi;
    }

    void updateRoi();

    virtual void disableRoiMode();

    virtual bool onEvent(Event e)
    {
        return Renderer::onEvent(e);
    }

    void updateRoiRotCylinder(std::string name, Eigen::Vector3f normal, Eigen::Vector3f color);

    pcl::visualization::Camera getCamera()
    {
        std::vector<pcl::visualization::Camera> cameras;
        vis->getCameras(cameras);
        return cameras.front();
    }

protected:
    boost::shared_ptr<AdvancedPclVisualizer> vis;

    PclRenderer(bool createWindow);

public:

    vtkSmartPointer<vtkRenderWindow> getRenderWindow()
    {
        return vis->getRenderWindow();
    }

    virtual void update();

    virtual void enableRoiMode(Eigen::Vector3f *dim, Eigen::Vector3f *translation, Eigen::Quaternionf *rotation);

    std::string getName()
    {
        return "Pcl Renderer";
    }
};

}
}
